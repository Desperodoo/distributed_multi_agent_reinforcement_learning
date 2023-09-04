import os
import ray
import time
import random
import torch.nn
import numpy as np
from torch import Tensor
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader


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
@ray.remote(num_cpus=1, num_gpus=0.001, resources={"node_-1": 0.001})
def evaluate(env, actor, p_num, args):  #
    episode_reward = 0
    device = next(actor.parameters()).device
    env.reset(p_num=p_num, e_num=1)
    
    # The hidden_state is initialized
    hidden_state = torch.zeros(size=(args.num_layers, p_num, args.rnn_hidden_dim), dtype=torch.float32, device=device)
    # The historical embedding is initialized
    history_embedding = [torch.zeros(size=(p_num, args.embedding_size), dtype=torch.float32, device=device) for _ in range(args.depth)]
    actor_embedding_dataset = EmbeddingDataset(attribute=history_embedding, adjacent=None)
    actor_current_embedding = torch.zeros(size=(p_num, args.embedding_size), dtype=torch.float32, device=device)
    
    o_state = env.boundary_obstacles
    o_ten = torch.as_tensor(o_state, dtype=torch.float32).to(device)
    for step in range(args.episode_limit):
        p_state = env.get_state(agent_type='defender')  # obs_n.shape=(N,obs_dim)
        e_state = env.get_state(agent_type='attacker')
        # the adjacent matrix of pursuer-pursuer, pursuer-obstacle, pursuer-evader
        p_adj = env.communicate()  # shape of (p_num, p_num)
        o_adj, e_adj = env.sensor()  # shape of (p_num, o_num), (p_num, e_num)
        # evader_step
        _, __ = env.evader_step()
        # make the dataset
        p_ten = torch.as_tensor(p_state, dtype=torch.float32).to(device)
        e_ten = torch.as_tensor(e_state, dtype=torch.float32).to(device)
        p_adj_ten = torch.as_tensor(p_adj, dtype=torch.float32).to(device)
        e_adj_ten = torch.as_tensor(e_adj, dtype=torch.float32).to(device)
        o_adj_ten = torch.as_tensor(o_adj, dtype=torch.float32).to(device)

        attribute_dataset = AttributeDataset(attribute=[p_ten, e_ten, o_ten], adjacent=[p_adj_ten, e_adj_ten, o_adj_ten])
        actor_embedding_dataset.update(attribute=actor_current_embedding, adjacent=p_adj_ten)

        a_n, hidden_state, actor_current_embedding = actor.choose_action(attribute_dataset, actor_embedding_dataset, hidden_state, deterministic=True)
        # Take a step
        r, done, info = env.step(a_n.detach().cpu().numpy())
        episode_reward += sum(r)

        if done:
            break
    # collision = env.collision
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
    def __init__(self, attribute: list, adjacent: torch.Tensor):
        self.attribute = attribute
        self.adjacent = adjacent
        
    def __len__(self):
        return len(self.attribute)
    
    def __getitem__(self, idx):
        a1 = self.attribute[idx]
        adj = self.adjacent
        return a1, idx, adj
    
    def update(self, embedding, adj):
        del self.attribute[0]
        self.attribute.append(embedding)
        self.adjacent = adj
        
        
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
    

class EmbeddingDataset(Dataset):
    def __init__(self, attribute: list, adjacent: torch.Tensor):
        self.attribute = attribute
        self.adjacent = adjacent
        
    def __len__(self):
        return len(self.attribute)
    
    def __getitem__(self, idx):
        a1 = self.attribute[idx]
        adj = self.adjacent
        return a1, idx, adj
    
    def update(self, embedding, adj):
        del self.attribute[0]
        self.attribute.append(embedding)
        self.adjacent = adj