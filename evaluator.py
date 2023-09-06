import os
import ray
import time
import random
import argparse
import torch.nn
import numpy as np
from torch import Tensor
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from mappo.DHGN.mappo_parallel import DHGN, SharedActor
from pursuit_evasion_game.env_2d.pursuit_env import Pursuit_Env
from pursuit_evasion_game.env_2d.gif_plotting import sim_moving


@ray.remote(num_cpus=1, num_gpus=0.001)
class EvaluatorProc(object):
    def __init__(self, args, num_cpus_eval):
        self.env = args.env_class()  # the env for Evaluator, `eval_env = env` in default
        self.agent = args.agent_class(args, batch_size=None, mini_batch_size=None, agent_type='Evaluator')
        self.agent_id = 0
        self.total_step = 0  # the total training step
        self.start_time = time.time()  # `used_time = time.time() - self.start_time`
        self.eval_times = args.evaluate_times  # number of times that get episodic cumulative return
        
        self.args = args
        self.num_cpus_eval = num_cpus_eval

        self.recorder = []  # total_step, r_avg, r_std, obj_c, ...
        self.max_r = -np.inf
        self.max_win_rate = 0.0
        self.min_collision_rate = 1.0
        print("| Evaluator:"
              "\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
              "\n| `time`: Time spent from the start of training to this moment."
              "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
              "\n| `avgS`: Average of steps in an episode."
              "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
              "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
              f"\n{'#' * 80}\n"
              f"{'ID':<3}{'Step':>8}{'Time':>8} |"
              f"{'avgR':>8}{'stdR':>7}{'Win':>7}{'Coll':>7}{'avgS':>7}{'stdS':>6} |"
              f"{'expR':>8}{'expW':>7}{'expColl':>7}{'objC':>7}{'objA':>7}{'etc.':>7}")
        
    def run(self, actor_weights, critic_weights, total_step, exp_r, exp_win_rate, exp_collision_rate, logging_tuple):
        args = self.args
        torch.set_grad_enabled(False)

        '''loop'''
        break_step = args.max_train_steps
        self.agent.actor.set_weights(actor_weights)
        self.agent.critic.set_weights(critic_weights)

        ref_list = self.evaluate_and_save(total_step, exp_r, exp_win_rate, exp_collision_rate, logging_tuple)
        
        '''Evaluator send the training signal to Learner'''
        if_train = self.total_step <= break_step
        print(f'| TrainingTime: {time.time() - self.start_time:>7.0f}')
        return [if_train, ref_list]
    
    def evaluate_and_save(self, new_total_step: int, exp_r: float, logging_tuple: tuple):
        self.total_step = new_total_step
        rewards_step_ten = self.get_rewards_and_step()
        returns = rewards_step_ten[:, 0]  # episodic cumulative returns of an
        steps = rewards_step_ten[:, 1]  # episodic step number
        avg_r = returns.mean().item()
        std_r = returns.std().item()
        avg_s = steps.mean().item()
        std_s = steps.std().item()

        train_time = int(time.time() - self.start_time)
        '''record the training information'''
        self.recorder.append((self.total_step, avg_r, std_r, exp_r, *logging_tuple))  # update recorder
        '''print some information to Terminal'''
        prev_r = self.max_r
        self.max_r = max(self.max_r, avg_r)  # update max average cumulative rewards
        print(f"{self.agent_id:<3}{self.total_step:8.2e}{train_time:8.0f} |"
              f"{avg_r:8.2f}{std_r:7.1f}{avg_s:7.0f}{std_s:6.0f} |"
              f"{exp_r:8.2f}{''.join(f'{n:7.2f}' for n in logging_tuple)}")
            
        if_save = avg_r >= prev_r
        if if_save:
            actor_ref = ray.put(self.agent.actor)
            critic_ref = ray.put(self.agent.critic)
            recorder_ref = ray.put(self.recorder)
            return [actor_ref, critic_ref, recorder_ref]
        else:
            return []

    def get_recorder(self):
        return self.recorder

    def get_rewards_and_step(self) -> Tensor:
        rewards_steps_list = list()
        evaluate_run_ref = [evaluate.remote(self.env, self.agent.actor, i % len(self.args.pursuer_num) + 10, self.args) for i in range(self.num_cpus_eval)]
        for _ in range(self.eval_times):
            while len(evaluate_run_ref) > 0:
                evaluate_ret_ref, evaluate_run_ref = ray.wait(evaluate_run_ref, num_returns=1, timeout=0.1)
                if len(evaluate_ret_ref) > 0:
                    rewards_steps = ray.get(evaluate_ret_ref)[0]
                    rewards_steps_list.append(rewards_steps)
        rewards_steps_ten = torch.tensor(rewards_steps_list, dtype=torch.float32)
        return rewards_steps_ten  # rewards_steps_ten.shape[1] == 2


"""util"""
# @ray.remote(num_cpus=1, num_gpus=0.001, resources={"node_-1": 0.001})
@ray.remote(num_cpus=1, num_gpus=0.001)
def evaluate(env, actor, p_num, args):  #
    episode_reward = 0
    device = next(actor.parameters()).device
    env.reset()
    
    # The hidden_state is initialized
    hidden_state = torch.zeros(size=(args.num_layers, p_num, args.rnn_hidden_dim), dtype=torch.float32, device=device)
    # The historical embedding is initialized
    history_embedding = [torch.zeros(size=(p_num, args.embedding_dim), dtype=torch.float32, device=device) for _ in range(args.depth)]
    actor_embedding_dataset = EmbeddingDataset(attribute=history_embedding, adjacent=None, depth=args.depth)
    actor_current_embedding = torch.zeros(size=(p_num, args.embedding_dim), dtype=torch.float32, device=device)
    
    o_state = env.boundary_map.obstacle_agent
    o_ten = torch.as_tensor(o_state, dtype=torch.float32).to(device)
    
    # Evaluate Matrix
    epi_obs_p = list()
    epi_obs_e = list()
    epi_target = list()
    epi_r = list()
    epi_path = list()
    epi_p_o_adj = list()
    epi_p_e_adj = list()
    epi_p_p_adj = list()
    epi_extended_obstacles = list()
    win_tag = False
    start_time = time.time()
    idx = 0
    for step in range(env.max_steps):
        print('step: ', idx)
        idx += 1
        p_state = env.get_state(agent_type='defender')  # obs_n.shape=(N,obs_dim)
        e_state = env.get_state(agent_type='attacker')
        # the adjacent matrix of pursuer-pursuer, pursuer-obstacle, pursuer-evader
        p_adj = env.communicate()  # shape of (p_num, p_num)
        o_adj, e_adj = env.sensor()  # shape of (p_num, o_num), (p_num, e_num)
        # evader_step
        path, pred_map = env.attacker_step()
        # make the dataset
        p_ten = torch.as_tensor(p_state, dtype=torch.float32).to(device)
        e_ten = torch.as_tensor(e_state, dtype=torch.float32).to(device)
        p_adj_ten = torch.as_tensor(p_adj, dtype=torch.float32).to(device)
        e_adj_ten = torch.as_tensor(e_adj, dtype=torch.float32).to(device)
        o_adj_ten = torch.as_tensor(o_adj, dtype=torch.float32).to(device)

        attribute_dataset = AttributeDataset(attribute=[p_ten, e_ten, o_ten], adjacent=[p_adj_ten, e_adj_ten, o_adj_ten])
        actor_embedding_dataset.update(embedding=actor_current_embedding, adjacent=p_adj_ten)
        attribute_dataloader = DataLoader(dataset=attribute_dataset, batch_size=1, shuffle=False)
        actor_embedding_dataloader = DataLoader(dataset=actor_embedding_dataset, batch_size=1, shuffle=False)

        a_n, hidden_state, actor_current_embedding = actor.choose_action(attribute_dataloader, actor_embedding_dataloader, hidden_state, deterministic=True)
        actor_current_embedding = actor_current_embedding.squeeze(0)
        # Take a step
        rewards, done, info = env.step(a_n.detach().cpu().numpy())
        episode_reward += sum(rewards)

        # Store Evaluate Matrix
        epi_obs_p.append(env.get_state(agent_type='defender'))
        epi_obs_e.append(env.get_state(agent_type='attacker'))
        epi_target.append(env.target[0])
        epi_r.append(sum(rewards))
        epi_path.append(path)
        epi_p_p_adj.append(p_adj)
        epi_p_e_adj.append(e_adj)
        epi_p_o_adj.append(o_adj)
        epi_extended_obstacles.append(pred_map.ex_moving_obstacles + pred_map.ex_obstacles)

        if done:
            print('DONE!')
            print('time cost: ', time.time() - start_time)
            print(f'reward: {episode_reward}')

            epi_obs_p = np.array(epi_obs_p)
            epi_obs_e = np.array(epi_obs_e)
            # Plotting
            sim_moving(
                step=env.time_step,
                height=map_config.map_size[0],
                width=map_config.map_size[1],
                obstacles=env.occupied_map.obstacles,
                boundary_obstacles=env.boundary_map.obstacles,
                extended_obstacles=epi_extended_obstacles,
                box_width=map_config.resolution,
                n_p=env_config.num_defender,
                n_e=1,
                p_x=epi_obs_p[:, :, 0],
                p_y=epi_obs_p[:, :, 1],
                e_x=epi_obs_e[:, :, 0],
                e_y=epi_obs_e[:, :, 1],
                path=epi_path,
                target=epi_target,
                e_ser=attacker_config.sen_range,
                c_r=defender_config.collision_radius,
                p_p_adj=epi_p_p_adj,
                p_e_adj=epi_p_e_adj,
                p_o_adj=epi_p_o_adj,
                dir='sim_moving' + str(time.time())
            )
            
            break
    return [episode_reward, step]


"""learning curve"""
def draw_learning_curve(recorder: np.ndarray = None,
                        cwd: str = None):
    fig_title = cwd + 'learning_curve'
    save_path = cwd + 'learning_curve.jpg'
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    win_rate = recorder[:, 4]
    collision_rate = recorder[:, 5]
    obj_c = recorder[:, 6]
    obj_a = recorder[:, 7]

    '''plot subplots'''
    import matplotlib as mpl
    mpl.use('Agg')
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)

    '''axs[0]'''
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = 'darkcyan'
    ax01.set_ylabel('Explore AvgReward', color=color01)
    ax01.plot(steps, r_exp, color=color01, alpha=0.5, )
    ax01.tick_params(axis='y', labelcolor=color01)

    color0 = 'lightcoral'
    ax00.set_ylabel('Episode Return', color=color0)
    ax00.plot(steps, r_avg, label='Episode Return', color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()
    '''axs[1]'''
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = 'darkcyan'
    ax11.set_ylabel('objC', color=color11)
    ax11.fill_between(steps, obj_c, facecolor=color11, alpha=0.2, )
    ax11.tick_params(axis='y', labelcolor=color11)

    color10 = 'royalblue'
    ax10.set_xlabel('Total Steps')
    ax10.set_ylabel('objA', color=color10)
    ax10.plot(steps, obj_a, label='objA', color=color10)
    ax10.tick_params(axis='y', labelcolor=color10)
    for plot_i in range(6, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f'{plot_i}', color='grey', alpha=0.5)
    ax10.legend()
    ax10.grid()

    '''plot save'''
    plt.title(fig_title, y=2.3)
    plt.savefig(save_path)
    plt.close('all')  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


class EmbeddingDataset(Dataset):
    def __init__(self, attribute: list, adjacent: torch.Tensor, depth: int):
        self.attribute = attribute
        self.adjacent = adjacent
        self.depth = depth
        
    def __len__(self):
        return len(self.attribute)
    
    def __getitem__(self, index):
        idx = self.depth - (index + 1)
        a1 = self.attribute[idx]
        adj = self.adjacent
        return a1, index, adj
    
    def update(self, embedding, adjacent):
        del self.attribute[0]
        self.attribute.append(embedding)
        self.adjacent = adjacent
        
        
class AttributeDataset(Dataset):
    def __init__(self, attribute: list, adjacent: list):
        self.attribute = attribute
        self.adjacent = adjacent
        
    def __len__(self):
        return len(self.attribute)
    
    def __getitem__(self, idx):
        a1 = self.attribute[0]
        a2 = self.attribute[idx]
        adj = self.adjacent[idx]
        return a1, a2, idx, adj
    

if __name__ == "__main__":
    # Env Config
    parser = argparse.ArgumentParser("Configuration Setting of the MRS Environment")
    parser.add_argument("--env_name", type=str, default='Pursuit-Evasion Game')
    parser.add_argument("--max_steps", type=int, default=250, help='The maximum time steps in a episode')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
    parser.add_argument("--num_target", type=int, default=1, help='The number of target point, 1 in pursuit, n in coverage, 0 in navigation')
    parser.add_argument("--num_defender", type=int, default=10, help='The number of defender (pursuer/server/navigator)')
    parser.add_argument("--num_attacker", type=int, default=1, help='The number of attacker (evader/client/target)')
    parser.add_argument("--defender_class", type=str, default='Pursuer', help='The class of the defender')
    parser.add_argument("--attacker_class", type=str, default='Evader', help='The class of the attacker')
    env_config = parser.parse_args()
    
    # Map Config
    parser = argparse.ArgumentParser("Configuration Setting of the Map")
    parser.add_argument("--resolution", type=int, default=1, help='The resolution of the map')
    parser.add_argument("--num_obstacle_block", type=int, default=5, help='The number of the obstacles')
    parser.add_argument("--center", type=int, default=(30, 30), help='The center of the obstacles')
    parser.add_argument("--variance", type=int, default=10, help='The varience of normal distribution that generate the position of obstacle block')
    parser.add_argument("--map_size", type=tuple, default=(60, 60), help='The size of the map')
    parser.add_argument("--is3D", type=bool, default=False, help='The dimension of freedom, 2 or 3')
    parser.add_argument("--max_num_obstacle", type=int, default=110, help='The max number of boundary obstacle, equivalent to num_obs_block * 22 (boundary of a 6x7 rectangle)')
    map_config = parser.parse_args()
    
    # Pursuer Config
    parser = argparse.ArgumentParser("Configuration Setting of the Defender")
    parser.add_argument("--sen_range", type=int, default=8, help='The sensor range of the agents')
    parser.add_argument("--comm_range", type=int, default=16, help='The communication range of the agents')
    parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
    parser.add_argument("--vmax", type=float, default=2, help='The limitation of the velocity of the defender')
    parser.add_argument("--tau", type=float, default=0.2, help='The time constant of first-order dynamic system')
    parser.add_argument("--DOF", type=int, default=2, help='The dimension of freedom, 2 or 3')
    defender_config = parser.parse_args()
    
    # Evader Config
    parser = argparse.ArgumentParser("Configuration Setting of the Attacker")
    parser.add_argument("--sen_range", type=int, default=8, help='The sensor range of the agents')
    parser.add_argument("--comm_range", type=int, default=16, help='The communication range of the agents')
    parser.add_argument("--collision_radius", type=float, default=0.5, help='The smallest distance at which a collision can occur between two agents')
    parser.add_argument("--step_size", type=float, default=0.1, help='The size of simulation time step')
    parser.add_argument("--vmax", type=float, default=4, help='The limitation of the velocity of the defender')
    parser.add_argument("--tau", type=float, default=0.2, help='The time constant of first-order dynamic system')
    parser.add_argument("--DOF", type=int, default=2, help='The dimension of freedom, 2 or 3')
    parser.add_argument("--x_dim", type=int, default=map_config.map_size[0], help='The x-dimension of map')
    parser.add_argument("--y_dim", type=int, default=map_config.map_size[1], help='The y-dimension of map')
    parser.add_argument("--extend_dis", type=int, default=3, help='The extend distance for astar')
    attacker_config = parser.parse_args()
    
    # Sensor Config
    parser = argparse.ArgumentParser("Configuration Setting of the Sensor")
    parser.add_argument("--num_beams", type=int, default=36, help='The number of beams in LiDAR')
    parser.add_argument("--radius", type=int, default=defender_config.sen_range, help='The radius of beams in LiDAR')
    sensor_config = parser.parse_args()
    
    # Algorithm Config
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
    ## Main parameters
    parser.add_argument("--num_defender", type=int, default=env_config.num_defender, help='The number of defender (pursuer/server/navigator)')
    parser.add_argument("--num_attacker", type=int, default=env_config.num_attacker, help='The number of attacker (evader/client/target)')
    # parser.add_argument("--env_class", default=ParticleEnv)
    parser.add_argument("--state_dim", type=int, default=4, help="The state dimension = pursuer state + evader state")
    parser.add_argument("--action_dim", type=int, default=9, help="The dimension of action space")
    # parser.add_argument("--agent_class", default=MAPPO)
    parser.add_argument("--pretrain_model_cwd", type=str, default="./pretrain/experiment/pretrain_model_15")
    parser.add_argument("--learner_device", type=str, default="cuda")
    parser.add_argument("--worker_device", type=str, default="cpu")
    parser.add_argument("--evaluator_device", type=str, default="cpu")
    ## Learner Hyperparameters
    parser.add_argument("--max_train_steps", type=int, default=int(2e8), help=" Maximum number of training steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.05, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=1, help="GAE parameter")
    parser.add_argument("--entropy_coef", type=float, default=0.05, help="Trick 5: policy entropy")
    parser.add_argument("--save_cwd", type=str, default=f"./model")
    # parser.add_argument("--mini_batch_size", type=int, default=500, help="Minibatch size (the number of episodes)")
    # parser.add_argument("--batch_size", type=int, default=500, help="Batch size (the number of episodes)")
    ## Worker Hyperparameters
    # parser.add_argument("--num_workers", type=int, default=25, help="Number of workers")  
    parser.add_argument("--sample_epi_num", type=int, default=1, help="Number of episodes each worker sampled each round")
    # Evaluator Hyperparameters
    parser.add_argument("--eval_per_step", type=float, default=5000, help="Evaluate the policy every 'eval_per_step'")
    parser.add_argument("--evaluate_times", type=float, default=2, help="Evaluate times")
    parser.add_argument("--save_gap", type=int, default=int(1e5), help="Save frequency")
    # Network Hyperparameters
    parser.add_argument("--num_layers", type=int, default=2, help="The number of the hidden layers of RNN")
    parser.add_argument("--rnn_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of RNN")
    parser.add_argument("--embedding_dim", type=int, default=128, help="The dimension of the middle layer of GNN")
    parser.add_argument("--gnn_output_dim", type=int, default=128, help="The dimension of the output layer of GNN")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of MLP")
    # Tricks
    parser.add_argument("--use_spectral_norm", type=bool, default=True, help="Trick 10:spectral normalization")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_agent_specific", type=float, default=True, help="Whether to use agent specific global state.")
    parser.add_argument("--use_value_clip", type=float, default=True, help="Whether to use value clip.")
    # Attention Hyperparameters
    parser.add_argument("--att_n_heads", type=int, default=8, help="The number of heads of multi-head attention")
    parser.add_argument("--att_dim_k", type=int, default=128, help="The dimension of keys/querys network")
    parser.add_argument("--att_dim_v", type=int, default=128, help="The dimension of values network")
    parser.add_argument("--att_output_dim", type=int, default=128, help="The output dimension of attention")
    # DHGN Hyperparameters
    parser.add_argument("--vertex_level_aggregator", type=str, default='mean', help='.')
    parser.add_argument("--semantic_level_aggregator", type=str, default='mean', help='.')
    parser.add_argument("--fcra_aggregator", type=str, default='mean', help='.')
    parser.add_argument("--num_relation", type=int, default=3, help='.')
    parser.add_argument('--depth', type=int, default=3, help='.')
    
    algo_config = parser.parse_args()
    
    shared_net = DHGN(
        input_dim=algo_config.state_dim,
        embedding_dim=algo_config.embedding_dim,
        is_sn=False,
        algo_config=algo_config,
        device=algo_config.evaluator_device
    )
    actor = SharedActor(
        shared_net=shared_net,
        rnn_input_dim=algo_config.embedding_dim,
        action_dim=algo_config.action_dim,
        num_layers=algo_config.num_layers,
        rnn_hidden_dim=algo_config.rnn_hidden_dim,
        is_sn=False
    )
    
    env = Pursuit_Env(
        map_config=map_config,
        env_config=env_config,
        defender_config=defender_config,
        attacker_config=attacker_config,
        sensor_config=sensor_config
    )
    
    evaluate_ref = evaluate.remote(env, actor, env_config.num_defender, algo_config)
    [episode_reward, step] = ray.get(evaluate_ref)
    print('episode_reward', episode_reward)
    print('step', step)