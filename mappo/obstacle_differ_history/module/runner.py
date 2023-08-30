import os
import ray
import time
import warnings
# import wandb
import torch
import numpy as np
from mappo.obstacle_differ.module.replay_buffer import BigBuffer
# from multiprocessing import Process, Pipe
# from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp  # torch.multiprocessing extends multiprocessing of Python


@ray.remote(num_cpus=1, num_gpus=1)
class Learner(object):
    def __init__(self, args, batch_size, mini_batch_size, learner_id):
        self.learner_id = learner_id
        self.args = args
        self.total_steps = 0
        self.agent = args.agent_class(args=args, batch_size=batch_size, mini_batch_size=mini_batch_size, agent_type="Learner")
        self.buffer = BigBuffer(args=args)
        self.cwd = args.save_cwd
        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)
        self.learner_device = torch.device(args.learner_device)
        print("Learner is Activated")

    def collect_buffer(self, worker_run_ref):
        self.buffer.reset()
        exp_r = 0
        exp_win_rate = 0
        exp_collision_rate = 0
        exp_steps = 0
        worker_num = len(worker_run_ref)
        # if self.learner_id == 0:
        #     worker_list = [i for i in range(worker_num)]
        # else:
        #     worker_list = [i + 77 for i in range(worker_num)]

        while len(worker_run_ref) > 0:
            worker_return_ref, worker_run_ref = ray.wait(worker_run_ref, num_returns=1, timeout=0.1)
            if len(worker_return_ref) > 0:
                worker_id, p_num, win_rate, collision_rate, reward, buffer_items, steps = ray.get(worker_return_ref)[0]
                exp_r += reward
                exp_win_rate += win_rate
                exp_collision_rate += collision_rate
                
                exp_steps += steps
                self.buffer.add_mini_buffer(p_num, buffer_items)
                # worker_list.remove(worker_id)
                # print(worker_list)
        return exp_r / worker_num, exp_win_rate / worker_num, exp_collision_rate / worker_num, exp_steps

    def compute_and_get_gradients(self, total_steps):
        '''agent update network using training data'''
        # print(f'learner_{self.learner_id} train')
        torch.set_grad_enabled(True)
        # print("training_begin")
        object_c, object_a, actor_grad, critic_grad = self.agent.train(self.buffer, total_steps)
        # print("training_finished")
        torch.set_grad_enabled(False)
        return (object_c, object_a), actor_grad, critic_grad
    
    def save(self):
        '''save'''
        actor_ref = ray.put(self.agent.actor)
        critic_ref = ray.put(self.agent.critic)
        return [actor_ref, critic_ref]
        
    def get_actor(self):
        return self.agent.actor

    def get_weights(self):
        actor_weights = self.agent.actor.get_weights()
        critic_weights = self.agent.critic.get_weights()
        return actor_weights, critic_weights
    
    def set_weights(self, actor_weights, critic_weights):
        self.agent.actor.set_weights(actor_weights)
        self.agent.critic.set_weights(critic_weights)
        
    # def get_gradients(self):
    #     actor_grad = self.agent.actor.get_gradients()
    #     critic_grad = self.agent.critic.get_gradients()
    #     return actor_grad, critic_grad
    
    def set_gradients_and_update(self, actor_grad, critic_grad, total_steps):
        self.agent.ac_optimizer.zero_grad()
        self.agent.actor.set_gradients(actor_grad, self.learner_device)
        self.agent.critic.set_gradients(critic_grad, self.learner_device)
        self.agent.ac_optimizer.step()
        if self.args.use_lr_decay:
            self.agent.lr_decay(total_steps)


@ray.remote(num_cpus=1, num_gpus=0.001)
class Worker(object):
    def __init__(self, worker_id: int, p_num: int, args):
        self.worker_id = worker_id
        self.p_num = p_num
        self.args = args
        self.env = args.env_class()
        self.agent = args.agent_class(args, batch_size=None, mini_batch_size=None, agent_type="Worker")

    def run(self, actor_weights, critic_weights, map_info):
        warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")
        args = self.args
        worker_id = self.worker_id
        p_num = self.p_num
        torch.set_grad_enabled(False)
        '''init environment'''
        # env_class = args.env_class
        # env = env_class()

        '''init agent'''
        # agent.load_pretrain_model(args.pretrain_model_cwd)
        self.agent.actor.set_weights(actor_weights)
        self.agent.critic.set_weights(critic_weights)
        '''init buffer'''
        sample_epi_num = args.sample_epi_num

        '''loop'''
        del args
        # begin_time = time.time()
        '''Worker send the training data to Learner'''
        p_num, win_rate, collision_rate, exp_reward, buffer_items, steps = self.agent.explore_env(self.env, worker_id, p_num, sample_epi_num, map_info)
        # print(f'worker_{worker_id} finished exploring')
        # finish_time = time.time()
        # print('last time: ', finish_time - begin_time, 's')
        return worker_id, p_num, win_rate, collision_rate, exp_reward, buffer_items, steps
