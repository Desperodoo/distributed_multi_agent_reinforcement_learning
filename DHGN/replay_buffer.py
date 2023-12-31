import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, cfg):
        self.episode_limit = cfg.env.max_steps
        self.batch_size = cfg.algo.sample_epi_num
        self.episode_num = 0
        self.max_episode_len = 0
        self.device = torch.device(cfg.algo.worker_device)
        self.max_p_num = cfg.env.num_defender
        self.max_e_num = cfg.env.num_attacker
        self.max_o_num = cfg.map.num_max_obstacle
        # self.p_num = cfg.env.num_defender
        # self.e_num = cfg.env.num_attacker
        # self.o_num = cfg.map.num_max_obstacle
        self.p_dim = cfg.env.state_dim
        self.e_dim = cfg.env.state_dim
        self.o_dim = cfg.env.state_dim
        self.embedding_size = cfg.algo.embedding_dim
        self.depth = cfg.algo.depth
        
    def reset_buffer(self):
        def zero_ten(size):
            return torch.zeros(size=size, dtype=torch.float32, device=self.device)
        self.buffer = dict()
        self.buffer['p_state'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_p_num, self.p_dim))
        self.buffer['e_state'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_e_num, self.e_dim))
        self.buffer['o_state'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_o_num, self.o_dim))
        self.buffer['p_adj'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_p_num, self.max_p_num))
        self.buffer['e_adj'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_p_num, self.max_e_num))
        self.buffer['o_adj'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_p_num, self.max_o_num))
        self.buffer['actor_historical_embedding'] = zero_ten(size=(self.batch_size, self.episode_limit + self.depth, self.max_p_num, self.embedding_size))
        self.buffer['critic_historical_embedding'] = zero_ten(size=(self.batch_size, self.episode_limit + self.depth, self.max_p_num, self.embedding_size))
        self.buffer['v_n'] = zero_ten(size=(self.batch_size, self.episode_limit + 1, self.max_p_num))
        self.buffer['a_n'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_p_num))
        self.buffer['a_logprob_n'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_p_num))
        self.buffer['r'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_p_num))
        self.buffer['active'] = zero_ten(size=(self.batch_size, self.episode_limit, self.max_p_num))

    def store_transition(self, num_episode, episode_step, p_state, e_state, o_state, p_adj, e_adj, o_adj, actor_historical_embedding, critic_historical_embedding, v_n, a_n, a_logprob_n, r, active):
        p_num = len(p_state)
        e_num = len(e_state)
        o_num = len(o_state)

        self.buffer['p_state'][num_episode, episode_step, :p_num] = p_state        
        self.buffer['e_state'][num_episode, episode_step, :e_num] = e_state
        self.buffer['o_state'][num_episode, episode_step, :o_num] = o_state
        self.buffer['p_adj'][num_episode, episode_step, :p_num, :p_num] = p_adj
        self.buffer['e_adj'][num_episode, episode_step, :p_num, :e_num] = e_adj
        self.buffer['o_adj'][num_episode, episode_step, :p_num, :o_num] = o_adj
        self.buffer['actor_historical_embedding'][num_episode, episode_step + self.depth, :p_num] = actor_historical_embedding
        self.buffer['critic_historical_embedding'][num_episode, episode_step + self.depth, :p_num] = critic_historical_embedding

        self.buffer['v_n'][num_episode, episode_step, :p_num] = v_n
        self.buffer['a_n'][num_episode, episode_step, :p_num] = a_n
        self.buffer['a_logprob_n'][num_episode, episode_step, :p_num] = a_logprob_n
        self.buffer['r'][num_episode, episode_step, :p_num] = r
        self.buffer['active'][num_episode, episode_step, :p_num] = active

    def store_last_value(self, num_episode, episode_step, v_n):
        p_num = len(v_n)
        self.buffer['v_n'][num_episode, episode_step, :p_num] = v_n
        # Record max_episode_len


class BigBuffer:
    def __init__(self):
        self.buffer = None

    def get_training_data(self, device):
        for key in self.buffer:
            self.buffer[key] = self.buffer[key].to(device)
        return self.buffer

    def cal_exp_r(self):
        r = 0
        for mini_buffer in self.buffer:
            reward = torch.sum(mini_buffer.buffer['r']) / mini_buffer.batch_size
            r += reward
        r = r / len(self.buffer)
        return r

    def reset(self):
        self.buffer = None
        
    def concat_buffer(self, mini_buffer):
        if self.buffer == None:
            self.buffer = mini_buffer.buffer
        else:
            for key in self.buffer:
                self.buffer[key] = torch.cat([self.buffer[key], mini_buffer.buffer[key]], dim=0)
        
    def print_buffer_info(self):
        for key, value in self.buffer.items():
            batch_size = value.buffer['state'].shape[0]
            print(f'p_num: {key}', f'batch_size: {batch_size}')
            
    