import os
import ray
import hydra
import time 
import warnings
# import wandb
import torch
import numpy as np
from DHGN.replay_buffer import BigBuffer
from DHGN.mappo_parallel import MAPPO


@ray.remote(num_cpus=1, num_gpus=0.001)
class Learner(object):
    def __init__(self, cfg, batch_size, mini_batch_size, learner_id):
        self.learner_id = learner_id
        self.total_steps = 0
        self.agent = hydra.utils.instantiate(cfg.algo.agent_class, cfg, batch_size, mini_batch_size, "Learner")
        self.buffer = BigBuffer()
        self.cwd = cfg.algo.save_cwd
        if not os.path.exists(self.cwd):
            os.makedirs(self.cwd)
        self.learner_device = torch.device(cfg.algo.learner_device)
        self.use_lr_decay = cfg.algo.use_lr_decay
        print("Learner is Activated")

    def collect_buffer(self, worker_run_ref):
        self.buffer.reset()
        exp_r = 0
        exp_steps = 0
        worker_num = len(worker_run_ref)
        while len(worker_run_ref) > 0:
            worker_return_ref, worker_run_ref = ray.wait(worker_run_ref, num_returns=1, timeout=0.1)
            if len(worker_return_ref) > 0:
                reward, buffer_items, steps = ray.get(worker_return_ref)[0]
                exp_r += reward
                exp_steps += steps
                self.buffer.concat_buffer(buffer_items)
        return exp_r / worker_num, exp_steps

    def compute_and_get_gradients(self, total_steps):
        '''agent update network using training data'''
        torch.set_grad_enabled(True)
        object_c, object_a, actor_grad, critic_grad = self.agent.train(self.buffer, total_steps)
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
        if self.use_lr_decay:
            self.agent.lr_decay(total_steps)


@ray.remote(num_cpus=1, num_gpus=0.001)
class Worker(object):
    def __init__(self, worker_id: int, cfg):
        self.worker_id = worker_id
        self.env = hydra.utils.instantiate(cfg.env.env_class, cfg)
        self.agent = hydra.utils.instantiate(cfg.algo.agent_class, cfg, None, None, "Worker")
        self.sample_epi_num = cfg.algo.sample_epi_num

    def run(self, actor_weights, critic_weights):
        warnings.filterwarnings("ignore", message="Values in x were outside bounds during a minimize step, clipping to bounds")
        worker_id = self.worker_id
        torch.set_grad_enabled(False)
        '''init agent'''
        self.agent.actor.set_weights(actor_weights)
        self.agent.critic.set_weights(critic_weights)
        '''Worker send the training data to Learner'''
        exp_reward, buffer_items, steps = self.agent.explore_env(self.env, self.sample_epi_num)
        return exp_reward, buffer_items, steps
