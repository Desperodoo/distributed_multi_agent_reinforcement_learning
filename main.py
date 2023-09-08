import os
import ray
import hydra
from ray.data import read_numpy
import time
import random
# import wandb
import torch
import numpy as np
from runner import Learner, Worker
from evaluator import EvaluatorProc, draw_learning_curve


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# server -1
# ip: 172.18.196.193
# num_cpus: 128
# num_gpus: 2 * RTX 4090
# # server 0
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

@hydra.main(config_path='./', config_name='config', version_base=None)
def train_agent_multiprocessing(cfg):
    nodes_idx = [0]
    num_nodes = len(nodes_idx)
    num_cpus = [128, 80, 112, 72, 32]
    # num_workers = [num_cpus[node] - 2 for node in nodes_idx]
    num_workers = [1 for _ in nodes_idx]
    
    mini_batch_size = [round(num_workers[node] * cfg.algo.sample_epi_num / 2) for node in range(num_nodes)]
    batch_size = [num_workers[node] * cfg.algo.sample_epi_num for node in range(num_nodes)]
    
    num_cpus_eval = 1
    
    cwd = cfg.algo.save_cwd
    learners = list()
    workers = [[] for _ in range(num_nodes)]
    idx = 0
    for node in range(num_nodes): 
        # learners.append(Learner.options(resources={f"node_{node}": 0.001}).remote(cfg, batch_size[node], mini_batch_size[node], node))
        learners.append(Learner.remote(cfg, batch_size[node], mini_batch_size[node], node))
        for _ in range(num_workers[node]):
            # workers[node].append(Worker.options(resources={f"node_{node}": 0.001}).remote(idx, p_num, cfg))
            workers[node].append(Worker.remote(idx, cfg))
            idx += 1
    print(f'Learners: {len(learners)}')
    print(f'Workers: ', [len(workers[i]) for i in range(num_nodes)])
    
    # evaluator = EvaluatorProc.options(resources={f"node_{-1}": 0.001}).remote(cfg, num_cpus_eval)
    evaluator = EvaluatorProc.remote(cfg, num_cpus_eval)
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
        front_time = time.time()
        # Learner send actor_weights to Workers and Workers sample                
        worker_run_ref = [[] for _ in range(num_workers)]
        for node in range(num_nodes):            
            for worker in workers[node]:
                worker_run_ref[node].append(worker.run.remote(actor_weights, critic_weights))
        # Learners obtain training data from corresponding Workers
        exp_r = 0
        exp_steps = 0
        learner_run_ref = [learners[node].collect_buffer.remote(worker_run_ref[node]) for node in range(num_nodes)]
        while len(learner_run_ref) > 0:
            learner_ret_ref, learner_run_ref = ray.wait(learner_run_ref, num_returns=1, timeout=0.1)
            if len(learner_ret_ref) > 0:
                r, steps = ray.get(learner_ret_ref)[0]
                exp_r += r
                exp_steps += steps
        exp_r /= num_nodes
        total_steps += exp_steps

        for _ in range(cfg.algo.epochs):
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
            eval_run_ref = [evaluator.run.remote(actor_weights, critic_weights, total_steps, exp_r, log_tuple)]
        else:
            return_ref, eval_run_ref = ray.wait(object_refs=eval_run_ref, num_returns=1, timeout=0.1)
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
                    
                eval_run_ref = [evaluator.run.remote(actor_weights, critic_weights, total_steps, exp_r, mean_log_tuple)]
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

current_cwd = os.getcwd()
print(current_cwd)
train_agent_multiprocessing()
current_cwd = os.getcwd()
print(current_cwd)