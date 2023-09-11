import os
import ray
from ray.data import read_numpy
import time
import random
# import wandb
import torch
import argparse
import numpy as np

from environment.pursuit_evasion_game.pursuit_env.py import Pursuit_Env
from obstacle_differ_3hop import MAPPO
from obs_differ_3hop.module.runner import Learner, Worker
from obs_differ_3hop.module.evaluator import EvaluatorProc, draw_learning_curve


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


# server 0
# ip: 172.18.166.252
# num_cpus: 80
# num_gpus: 5 * RTX 8000
# server 1
# ip: 172.18.192.190
# num_cpus: 112
# num_gpus: 3 * RTX 3090
# server 2
# ip: 172.18.196.189
# num_cpus: 72
# num_gpus: 4 * RTX 2080
# server 3
# ip: 172.18.196.180
# num_cpus: 32
# num_gpus: 4 * RTX 2080


def train_agent_multiprocessing(args):
    env = args.env_class()
    args.state_dim = env.state_dim  # The state dimension = pursuer state + evader state
    args.episode_limit = env.episode_limit
    nodes_idx = [0, 1]
    num_nodes = len(nodes_idx)
    tasks = len(args.pursuer_num) * len(args.evader_num)
    num_cpus = [80, 112, 72, 32]
    num_workers = [int(num_cpus[node] / tasks) for node in nodes_idx]
    
    mini_batch_size = [round(num_workers[node] * args.sample_epi_num / 2) for node in range(num_nodes)]
    batch_size = [num_workers[node] * args.sample_epi_num for node in range(num_nodes)]
    
    num_cpus_eval = 72
    # map_dir = 'local://~/zhili/map_data/'
    
    # boundary_map_list = read_numpy(map_dir + 'boundary_map_list.npy')
    # boundary_obstacle_list = read_numpy(map_dir + 'boundary_obstacle_list.npy')
    # boundary_obstacle_num_list = read_numpy(map_dir + 'boundary_obstacle_num_list.npy')
    
    # hash_map_list = read_numpy(map_dir + 'hash_map_list.npy')
    
    # obstacle_list = read_numpy(map_dir + 'obstacle_list.npy')
    # obstacle_map_list = read_numpy(map_dir + 'obstacle_map_list.npy')
    # obstacle_num_list = read_numpy(map_dir + 'obstacle_num_list.npy')
    
    map_dir = '/home/lizh/zhili/map_data/'
    
    boundary_map_list = np.load(map_dir + 'boundary_map_list.npy')
    boundary_obstacle_list = np.load(map_dir + 'boundary_obstacle_list.npy')
    boundary_obstacle_num_list = np.load(map_dir + 'boundary_obstacle_num_list.npy')
    
    hash_map_list = np.load(map_dir + 'hash_map_list.npy')
    
    obstacle_list = np.load(map_dir + 'obstacle_list.npy')
    obstacle_map_list = np.load(map_dir + 'obstacle_map_list.npy')
    obstacle_num_list = np.load(map_dir + 'obstacle_num_list.npy')
    
    cwd = args.save_cwd
    learners = list()
    workers = [[] for _ in range(num_nodes)]
    idx = 0
    for node in range(num_nodes): 
        learners.append(Learner.options(resources={f"node_{node}": 0.001}).remote(args, batch_size[node], mini_batch_size[node], node))
        for _ in range(num_workers[node]):
            for p_num in args.pursuer_num:
                workers[node].append(Worker.options(resources={f"node_{node}": 0.001}).remote(idx, p_num, args))
                idx += 1
    print(f'Learners: {len(learners)}')
    print(f'Workers: {len(workers[0])}, {len(workers[1])}')
    
    evaluator = EvaluatorProc.options(resources={f"node_{-1}": 0.001}).remote(args, num_cpus_eval)
    print(f'Evaluator: {num_cpus_eval}')
    
    if_Train = True
    total_steps = 0
    # Attention: Get weights from learner in head node 
    actor_weights, critic_weights = ray.get(learners[0].get_weights.remote())
    for node in range(num_nodes):
        learners[node].set_weights.remote(actor_weights, critic_weights)
    
    exp_r = 0
    eval_run_ref = None
    while if_Train:
        # Learner send actor_weights to Workers and Workers sample        
        front_time = time.time()
        # Learner send actor_weights to Workers and Workers sample                
        worker_run_ref = list()
        for node in range(num_nodes):
            mini_list = list()
            idx = random.randint(0, 2000 - 1)
            boundary_map = boundary_map_list[idx]
            obstacle_map = obstacle_map_list[idx]
            
            obstacles_idx = [obstacle_num_list[:idx].sum(), obstacle_num_list[:idx + 1].sum()]
            boundary_obstacle_idx = [boundary_obstacle_num_list[:idx].sum(), boundary_obstacle_num_list[:idx + 1].sum()]
            
            obstacles = obstacle_list[obstacles_idx[0]:obstacles_idx[1]].tolist()
            boundary_obstacles = boundary_obstacle_list[boundary_obstacle_idx[0]:boundary_obstacle_idx[1]].tolist()
            
            hash_map = hash_map_list[idx]
            
            map_info = [obstacle_map, boundary_map, obstacles, boundary_obstacles, hash_map]
            
            for worker in workers[node]:
                mini_list.append(worker.run.remote(actor_weights, critic_weights, map_info))
            worker_run_ref.append(mini_list)
        # Learners obtain training data from corresponding Workers
        exp_r = 0
        exp_win_rate = 0
        exp_collision_rate = 0
        exp_steps = 0
        learner_run_ref = [learners[node].collect_buffer.remote(worker_run_ref[node]) for node in range(num_nodes)]
        while len(learner_run_ref) > 0:
            learner_ret_ref, learner_run_ref = ray.wait(learner_run_ref, num_returns=1, timeout=0.1)
            if len(learner_ret_ref) > 0:
                r, win_rate, collision_rate, steps = ray.get(learner_ret_ref)[0]
                exp_r += r
                exp_win_rate += win_rate
                exp_collision_rate += collision_rate
                exp_steps += steps
        exp_r /= num_nodes
        exp_collision_rate /= num_nodes
        exp_win_rate /= num_nodes
        total_steps += exp_steps

        for _ in range(args.K_epochs):
            # Learners compute and send gradients to main func
            actor_gradients = []
            critic_gradients = []
            calculator = 0

            learner_run_ref = [learner.compute_and_get_gradients.remote(total_steps) for learner in learners]
            while len(learner_run_ref) > 0:
                learner_ret_ref, learner_run_ref = ray.wait(learner_run_ref, num_returns=1, timeout=0.1)
                if len(learner_ret_ref) > 0:
                    log_tuple, actor_grad, critic_grad = ray.get(learner_ret_ref)[0]
                    actor_gradients.append(actor_grad)
                    critic_gradients.append(critic_grad)
                    
                    if calculator == 0:
                        mean_log_tuple = log_tuple
                        calculator += 1
                    else:
                        mean_log_tuple = [mean_log_tuple[i] + log_tuple[i] for i in range(len(log_tuple))]
            mean_log_tuple = [mean_log_tuple[i] / num_nodes for i in range(len(mean_log_tuple))]        

            # Summed gradients
            summed_actor_gradients = [
                np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*actor_gradients)
            ]
            summed_critic_gradients = [
                np.stack(gradient_zip).sum(axis=0) for gradient_zip in zip(*critic_gradients)
            ]
            # Learners obtain and update global gradients
            for node in range(num_nodes):
                learners[node].set_gradients_and_update.remote(summed_actor_gradients, summed_critic_gradients, total_steps)

        # Get current weights from Learners, send them to Evaluator
        actor_weights, critic_weights = ray.get(learners[0].get_weights.remote())
        print("training time cost: ", time.time() - front_time, "s")
        
        if eval_run_ref is None:  # Evaluate the first time 
            eval_run_ref = [evaluator.run.remote(actor_weights, critic_weights, total_steps, exp_r, exp_win_rate, exp_collision_rate, log_tuple)]
        else:
            return_ref, eval_run_ref = ray.wait(object_refs=eval_run_ref, num_returns=1, timeout=0.1)
            # print('return_ref: ', return_ref)
            # print('eval_run_ref: ', eval_run_ref)
            if len(return_ref):  # if evaluator.run is done
                obj = ray.get(return_ref)[0]
                if_Train, ref_list = obj[0], obj[1]
                if len(ref_list) > 0:
                    actor = ray.get(ref_list[0])
                    critic = ray.get(ref_list[1])
                    recorder = ray.get(ref_list[2])
                    np.save(cwd + '/recorder.npy', recorder)
                    draw_learning_curve(recorder=np.array(recorder), cwd='./model/')
                    
                    torch.save(actor, cwd + '/actor.pth')
                    torch.save(critic, cwd + '/critic.pth')
                    torch.save(actor.shared_net, cwd + '/actor_gnn.pth')
                    torch.save(critic.shared_net, cwd + '/critic_gnn.pth')
                    torch.save(actor.GRU, cwd + '/actor_gru.pth')
                    torch.save(critic.GRU, cwd + '/critic_gru.pth')
                    torch.save(actor.Mean, cwd + '/actor_mean.pth')
                    torch.save(critic.Mean, cwd + '/critic_mean.pth')
                    
                eval_run_ref = [evaluator.run.remote(actor_weights, critic_weights, total_steps, exp_r, exp_win_rate, exp_collision_rate, mean_log_tuple)]
    ref_list = ray.get(learners[1].save.remote())
    actor = ray.get(ref_list[0])
    critic = ray.get(ref_list[1])
    torch.save(actor, cwd + '/actor_final.pth')
    torch.save(critic, cwd + '/critic_final.pth')
    torch.save(actor.shared_net, cwd + '/actor_gnn_final.pth')
    torch.save(critic.shared_net, cwd + '/critic_gnn_final.pth')
    torch.save(actor.GRU, cwd + '/actor_gru_final.pth')
    torch.save(critic.GRU, cwd + '/critic_gru_final.pth')
    torch.save(actor.Mean, cwd + '/actor_mean_final.pth')
    torch.save(critic.Mean, cwd + '/critic_mean_final.pth')
    recorder = ray.get(evaluator.get_recorder.remote())
    np.save(cwd + '/recorder.npy', recorder)
    draw_learning_curve(recorder=np.array(recorder), cwd='./model/')


parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in SMAC environment")
# Main parameters
parser.add_argument("--pursuer_num", default=[i for i in range(10, 16)])
parser.add_argument("--evader_num", default=[1])
parser.add_argument("--env_class", default=ParticleEnv)
# parser.add_argument("--episode_limit", type=int, default=80, help="The max episode length")
parser.add_argument("--p_dim", type=int, default=6)
parser.add_argument("--e_dim", type=int, default=6)
# parser.add_argument("--state_dim", type=int, default=6, help="The state dimension = pursuer state + evader state")
parser.add_argument("--action_dim", type=int, default=9, help="The dimension of action space")
parser.add_argument("--agent_class", default=MAPPO)
parser.add_argument("--pretrain_model_cwd", type=str, default="./pretrain/experiment/pretrain_model_15")
# Learner Hyperparameters
parser.add_argument("--max_train_steps", type=int, default=int(2e8), help=" Maximum number of training steps")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.05, help="GAE parameter")
parser.add_argument("--K_epochs", type=int, default=1, help="GAE parameter")
parser.add_argument("--entropy_coef", type=float, default=0.05, help="Trick 5: policy entropy")
parser.add_argument("--save_cwd", type=str, default=f"./model")
# parser.add_argument("--mini_batch_size", type=int, default=500, help="Minibatch size (the number of episodes)")
# Worker Hyperparameters
# parser.add_argument("--num_workers", type=int, default=25, help="Number of workers")  
parser.add_argument("--sample_epi_num", type=int, default=1, help="Number of episodes each worker sampled each round")
# parser.add_argument("--batch_size", type=int, default=500, help="Batch size (the number of episodes)")
# Evaluator Hyperparameters
parser.add_argument("--eval_per_step", type=float, default=5000, help="Evaluate the policy every 'eval_per_step'")
parser.add_argument("--evaluate_times", type=float, default=2, help="Evaluate times")
parser.add_argument("--save_gap", type=int, default=int(1e5), help="Save frequency")
# Network Hyperparameters
parser.add_argument("--n_hops", type=int, default=1)
parser.add_argument("--num_layers", type=int, default=2, help="The number of the hidden layers of RNN")
parser.add_argument("--rnn_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of RNN")
parser.add_argument("--gnn_middle_dim", type=int, default=128, help="The dimension of the middle layer of GNN")
parser.add_argument("--gnn_output_dim", type=int, default=128, help="The dimension of the output layer of GNN")
parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The dimension of the hidden layer of MLP")
parser.add_argument("--use_relu", type=float, default=True, help="Whether to use relu, if False, we will use tanh")
parser.add_argument("--use_rnn", type=bool, default=True, help="Whether to use RNN")
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
parser.add_argument("--learner_device", type=str, default="cuda")
parser.add_argument("--worker_device", type=str, default="cpu")
parser.add_argument("--evaluator_device", type=str, default="cpu")
# Attention Hyperparameters
parser.add_argument("--att_n_heads", type=int, default=8, help="The number of heads of multi-head attention")
parser.add_argument("--att_dim_k", type=int, default=128, help="The dimension of keys/querys network")
parser.add_argument("--att_dim_v", type=int, default=128, help="The dimension of values network")
parser.add_argument("--att_output_dim", type=int, default=128, help="The output dimension of attention")
# Constraint Hyperparameters
parser.add_argument("--lambda_lr", type=float, default=5e-2, help="The lagrangian multiplier")
config = parser.parse_args()
current_cwd = os.getcwd()
print(current_cwd)
train_agent_multiprocessing(config)
current_cwd = os.getcwd()
print(current_cwd)