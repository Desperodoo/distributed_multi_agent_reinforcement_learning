import numpy as np
import torch


class MiniBuffer:
    def __init__(self, buffer_id, args):
        self.episode_limit = args.episode_limit
        self.batch_size = args.sample_epi_num
        self.episode_num = 0
        self.max_episode_len = 0
        self.buffer = None
        self.device = torch.device(args.worker_device)
        self.buffer_id = buffer_id
        self.args = args

    def reset_buffer(self, p_num, o_num, p_obs_dim, e_obs_dim, action_dim):
        def zero_ten(size):
            tensor = torch.zeros(size=size, dtype=torch.float32, device=self.device)
            return tensor
        self.buffer = dict()
        self.buffer['state'] = zero_ten(size=(self.batch_size, self.episode_limit, p_num, p_num + o_num, e_obs_dim + p_obs_dim + 1))
        self.buffer['adj'] = zero_ten(size=(self.batch_size, self.episode_limit, p_num, p_num + o_num))
        self.buffer['actor_comm_embedding'] = zero_ten(size=(self.batch_size, self.episode_limit, p_num, self.args.gnn_output_dim))
        self.buffer['critic_comm_embedding'] = zero_ten(size=(self.batch_size, self.episode_limit, p_num, self.args.gnn_output_dim))
        self.buffer['v_n'] = zero_ten(size=(self.batch_size, self.episode_limit + 1, p_num))
        self.buffer['a_n'] = zero_ten(size=(self.batch_size, self.episode_limit, p_num))
        self.buffer['a_logprob_n'] = zero_ten(size=(self.batch_size, self.episode_limit, p_num))
        self.buffer['r'] = zero_ten(size=(self.batch_size, self.episode_limit, p_num))
        self.buffer['active'] = zero_ten(size=(self.batch_size, self.episode_limit, p_num))
        self.buffer['win'] = zero_ten(size=(self.batch_size, 1))

        self.max_episode_len = 0

    def store_transition(self, num_episode, episode_step, state, p_adj, actor_comm_embedding, critic_comm_embedding, v_n, a_n, a_logprob_n, r, active, win):
        self.buffer['state'][num_episode][episode_step] = state
        self.buffer['adj'][num_episode][episode_step] = p_adj
        self.buffer['actor_comm_embedding'][num_episode][episode_step] = actor_comm_embedding
        self.buffer['critic_comm_embedding'][num_episode][episode_step] = critic_comm_embedding

        self.buffer['v_n'][num_episode][episode_step] = v_n
        self.buffer['a_n'][num_episode][episode_step] = a_n
        self.buffer['a_logprob_n'][num_episode][episode_step] = a_logprob_n
        self.buffer['r'][num_episode][episode_step] = r
        self.buffer['active'][num_episode][episode_step] = active
        self.buffer['win'][num_episode] = win

    def store_last_value(self, num_episode, episode_step, v_n):
        self.buffer['v_n'][num_episode][episode_step] = v_n
        # Record max_episode_len
        if episode_step > self.max_episode_len:
            self.max_episode_len = episode_step

    def clear(self, num_episode):
        for key in self.buffer:
            self.buffer[key][num_episode] = torch.zeros_like(input=self.buffer[key][num_episode], dtype=torch.float32, device=self.device)


class BigBuffer:
    def __init__(self, args):
        self.buffer = dict()
        
    def add_mini_buffer(self, p_num, mini_buffer):  # TODO: Match mini_buffer with worker_id
        if f"{p_num}" in self.buffer:
            self.concat_mini_buffer(p_num, mini_buffer)
            # print(self.buffer[f"{p_num}"].buffer['v_n'].shape)
        else:
            self.buffer[f"{p_num}"] = mini_buffer

    def get_training_data(self, p_num, device):
        minibuffer = self.buffer[f"{p_num}"]
        buffer = minibuffer.buffer
        max_episode_len = minibuffer.max_episode_len
        for key in buffer:
            if key == 'v_n':
                buffer[key] = buffer[key][:, :max_episode_len + 1].to(device)
            else:
                buffer[key] = buffer[key][:, :max_episode_len].to(device)
        return buffer, max_episode_len

    def cal_exp_r(self):
        r = 0
        for mini_buffer in self.buffer:
            reward = torch.sum(mini_buffer.buffer['r']) / mini_buffer.batch_size
            r += reward
        r = r / len(self.buffer)
        return r

    def reset(self):
        self.buffer = dict()
        
    def concat_mini_buffer(self, p_num, mini_buffer):
        origin_buffer = self.buffer[f"{p_num}"]
        for key in origin_buffer.buffer:
            origin_buffer.buffer[key] = torch.cat([origin_buffer.buffer[key], mini_buffer.buffer[key]], dim=0)
        
        origin_buffer.max_episode_len = max(origin_buffer.max_episode_len, mini_buffer.max_episode_len)
        
    def print_buffer_info(self):
        for key, value in self.buffer.items():
            batch_size = value.buffer['state'].shape[0]
            print(f'p_num: {key}', f'batch_size: {batch_size}')
            
    