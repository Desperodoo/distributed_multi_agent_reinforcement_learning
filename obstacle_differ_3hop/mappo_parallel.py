import torch
import time
from copy import deepcopy
from tqdm import tqdm
import torch.nn as nn
from torch.nn import Parameter
from torch.distributions import Normal
import torch.nn.functional as f
from torch.distributions import Categorical
from torch.utils.data.sampler import *
import numpy as np
import copy
from torch.nn.utils import spectral_norm
from obstacle_differ_3hop.module.replay_buffer import MiniBuffer
from obstacle_differ_3hop.module.normalization import Normalization


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)
    return layer


def preproc_layer(input_size, output_size, is_sn=False):
    layer = nn.Linear(input_size, output_size)
    orthogonal_init(layer)
    return spectral_norm(layer) if is_sn else layer


class GnnExtractor(nn.Module):
    def __init__(self, input_size, middle_size, output_size, n_hops: int = 1, is_sn: bool = False):
        super().__init__()
        self.n_hop = n_hops
        self.one_hop = nn.Sequential(
            preproc_layer(input_size, middle_size) if is_sn else nn.Linear(input_size, middle_size),
            nn.ReLU(),
            preproc_layer(middle_size, output_size) if is_sn else nn.Linear(middle_size, output_size),
            nn.ReLU()
        )
        self.bottleneck = nn.Sequential(
            preproc_layer(output_size + 2 * output_size, output_size) if is_sn else nn.Linear(output_size + 2 * output_size, output_size),
            nn.ReLU(),
        )
        
    def forward(self, obs: torch.Tensor, last_comm_embedding: torch.Tensor = None, adj: torch.Tensor = None) -> torch.Tensor:
        '''
        :param obs: shape = [*, agent_num, agent_num, orig_feature_dim]
        :param adj: shape = [*, agent_num, 1, agent_num]
        :return residual_extract_feature:  shape = [*, agent_num, 3 * orig_feature_dim]
        '''
        # last_comm_embedding = last_comm_embedding.repeat([])
        adj = f.normalize(adj, p=1, dim=-1)
        adj = adj.unsqueeze(dim=-2)
        h0 = self.one_hop(obs)
                
        h0_agg = torch.matmul(adj, h0)
        h0_agg = h0_agg.squeeze(dim=-2)
        adj = adj.squeeze(dim=-2)
        agent_num = adj.shape[-2]
        if len(adj.shape) == 2:
            comm_agg = torch.matmul(adj[:, :agent_num], last_comm_embedding)
        else:
            comm_agg = torch.matmul(adj[:, :, :, :agent_num], last_comm_embedding)
        feature_agg = torch.concatenate([h0_agg, comm_agg], dim=-1)
        feature = self.bottleneck(feature_agg)
        return feature


class SharedActor(nn.Module):
    def __init__(self, shared_net, rnn_input_dim, output_size, num_layers, hidden_size, is_sn=False):
        super().__init__()
        self.shared_net = shared_net
        self.num_layers = num_layers
        self.rnn_input_size = rnn_input_dim
        self.hidden_size = hidden_size
        self.GRU = nn.GRU(self.rnn_input_size, hidden_size, num_layers)
        self.Mean = preproc_layer(hidden_size, output_size) if is_sn else nn.Linear(hidden_size, output_size)

    def forward(self, state, adj, hidden_state, last_comm_embedding, mode):
        # As for GRU: 
        #             input       : tensor of the shape (Length, Batch, Input_Size)
        #             hidden_state: tensor of the shape (D * num_layers, Batch, Hidden_Size)
        #                           D = 2 if bidirectional == True else 1
        #             output      : tensor of the shape (Length, Batch, D * Output_Size)
        # When choose action: 
        #             input       : tensor of the shape (Num_of_Agent, Middle_Size + Input_Size)
        #                                            => (1, Num_of_Agent, Middle_Size + Input_Size)
        #             hidden_state: tensor of the shape (1 * num_layers, Num_of_Agent, Hidden_Size)
        #             output      : tensor of the shape (1, Num_of_Agent, Hidden_Size)
        #                                            => (Num_of_Agent, Hidden_Size)     
        # When get logprob & entropy: 
        #             input       : tensor of the shape (Batch, Steps, Num_of_Agent, Middle_Size + Input_Size)
        #                                            => (Steps, Batch, Num_of_Agent, Middle_Size + Input_Size)
        #                                            => (Steps, Batch * Num_of_Agent, Middle_Size + Input_Size)
        #             hidden_state: tensor of the shape (1 * num_layers, Batch * Num_of_Agent, Hidden_Size)
        #             output      : tensor of the shape (Steps, Batch * Num_of_Agent, Hidden_Size)
        #                                            => (Steps, Batch, Num_of_Agent, Hidden_Size)
        #                                            => (Batch, Steps, Num_of_Agent, Hidden_Size)            
        comm_embedding = self.shared_net(state, last_comm_embedding, adj)
        # print('comm_embedding.shape', comm_embedding.shape)
        # print('last_comm_embedding.shape', last_comm_embedding.shape)
        assert 2 * comm_embedding.shape[-1] == last_comm_embedding.shape[-1]
        # comm_embedding.shape == [*, agent_num, gnn_output_dim]
        
        if mode == 0:
            feature = comm_embedding.unsqueeze(0)  # (Num_of_Agent, Middle_Size + Input_Size) => (1, Num_of_Agent, Middle_Size + Input_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.squeeze(0)
        else:
            batch = comm_embedding.shape[0]
            steps = comm_embedding.shape[1]
            num_agent = comm_embedding.shape[2]

            feature = comm_embedding.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Middle_Size + Input_Size) => (Steps, Batch, Num_of_Agent, Middle_Size + Input_Size)
            feature = feature.reshape((steps, -1, self.rnn_input_size))  # (Steps, Batch, Num_of_Agent, Middle_Size + Input_Size) => (Steps, Batch * Num_of_Agent, Middle_Size + Input_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.reshape((steps, batch, num_agent, self.hidden_size))  # (Steps, Batch, Num_of_Agent, Hidden_Size) <= (Steps, Batch * Num_of_Agent, Hidden_Size)
            feature = feature.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Hidden_Size) <= (Steps, Batch, Num_of_Agent, Hidden_Size)
        prob = torch.softmax(self.Mean(feature), dim=-1)
        return prob, hidden_state, comm_embedding

    def choose_action(self, state, adj, hidden_state, last_comm_embedding, deterministic=True):
        prob, hidden_state, comm_embedding = self.forward(state, adj, hidden_state, last_comm_embedding, mode=0)
        if deterministic:
            action = prob.argmax(dim=-1, keepdim=False)
            return action, hidden_state, comm_embedding
        else:
            dist = Categorical(probs=prob)
            a_n = dist.sample()
            a_logprob_n = dist.log_prob(a_n)
            return a_n, a_logprob_n, hidden_state, comm_embedding

    def get_logprob_and_entropy(self, state, adj, hidden_state, last_comm_embedding, action):
        prob, _, __ = self.forward(state, adj, hidden_state, last_comm_embedding, mode=1)
        dist = Categorical(prob)
        a_logprob = dist.log_prob(action)
        entropy = dist.entropy()
        return a_logprob, entropy

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients, device):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.tensor(g).to(device)

class SharedCritic(nn.Module):
    def __init__(self, shared_net, rnn_input_dim, output_size, num_layers, hidden_size, is_sn=False):
        super().__init__()
        self.shared_net = shared_net
        self.num_layers = num_layers
        self.rnn_input_size = rnn_input_dim
        self.hidden_size = hidden_size
        self.GRU = nn.GRU(rnn_input_dim, hidden_size, num_layers)
        self.Mean = preproc_layer(hidden_size, output_size, is_sn=is_sn)

    def forward(self, state, adj, hidden_state, last_comm_embedding, mode):
        # As for GRU: 
        #             input       : tensor of the shape (Length, Batch, Input_Size)
        #             hidden_state: tensor of the shape (D * num_layers, Batch, Hidden_Size)
        #                           D = 2 if bidirectional == True else 1
        #             output      : tensor of the shape (Length, Batch, D * Output_Size)
        # When choose action: 
        #             input       : tensor of the shape (Num_of_Agent, Middle_Size + Input_Size)
        #                                            => (1, Num_of_Agent, Middle_Size + Input_Size)
        #             hidden_state: tensor of the shape (1 * num_layers, Num_of_Agent, Hidden_Size)
        #             output      : tensor of the shape (1, Num_of_Agent, Hidden_Size)
        #                                            => (Num_of_Agent, Hidden_Size)     
        # When get logprob & entropy: 
        #             input       : tensor of the shape (Batch, Steps, Num_of_Agent, Middle_Size + Input_Size)
        #                                            => (Steps, Batch, Num_of_Agent, Middle_Size + Input_Size)
        #                                            => (Steps, Batch * Num_of_Agent, Middle_Size + Input_Size)
        #             hidden_state: tensor of the shape (1 * num_layers, Batch * Num_of_Agent, Hidden_Size)
        #             output      : tensor of the shape (Steps, Batch * Num_of_Agent, Hidden_Size)
        #                                            => (Steps, Batch, Num_of_Agent, Hidden_Size)
        #                                            => (Batch, Steps, Num_of_Agent, Hidden_Size)    
        device = next(self.shared_net.parameters()).device
        connected_adj = torch.ones_like(input=adj, device=device)
        # comm_embedding: tensor of the shape ()
        comm_embedding = self.shared_net(state, last_comm_embedding, connected_adj)
        
        assert 2 * comm_embedding.shape[-1] == last_comm_embedding.shape[-1]

        if mode == 0:
            feature = comm_embedding.unsqueeze(0)  # (Num_of_Agent, Hidden_Size) => (1, Num_of_Agent, Hidden_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.squeeze(0)
            val = self.Mean(feature)
            return val, hidden_state, comm_embedding

        else:
            batch = comm_embedding.shape[0]
            steps = comm_embedding.shape[1]
            num_agent = comm_embedding.shape[2]            
            feature = comm_embedding.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Middle_Size) => (Steps, Batch, Num_of_Agent, Middle_Size)
            feature = feature.reshape((steps, -1, self.rnn_input_size))  # (Steps, Batch, Num_of_Agent, Middle_Size) => (Steps, Batch * Num_of_Agent, Middle_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.reshape((steps, batch, num_agent, self.hidden_size))  # (Steps, Batch, Num_of_Agent, Hidden_Size) <= (Steps, Batch * Num_of_Agent, Hidden_Size)
            feature = feature.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Hidden_Size) <= (Steps, Batch, Num_of_Agent, Hidden_Size)
            val = self.Mean(feature)
            return val

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients, device):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.tensor(g).to(device)


class MAPPO:
    def __init__(self, args, batch_size, mini_batch_size, agent_type):
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_value_clip = args.use_value_clip
        # get the input dimension of actor and critic
        self.actor_input_dim = args.state_dim + 1
        self.critic_input_dim = args.state_dim + 1
        self.num_layers = args.num_layers
        self.gnn_output_dim = args.gnn_output_dim
        self.rnn_input_dim = args.gnn_output_dim
        self.rnn_hidden_dim = args.rnn_hidden_dim
        self.n_hops = args.n_hops
        if "Learner" in agent_type:
            self.device = torch.device(args.learner_device)
        elif "Worker" in agent_type:
            self.device = torch.device(args.worker_device)
        else:
            self.device = torch.device(args.evaluator_device)

        if args.use_reward_norm:
            # print("------use reward norm------")
            self.reward_norm = dict()
            for i in args.pursuer_num:
                self.reward_norm[f'{i}'] = Normalization(shape=i)

        # if self.use_agent_specific:
        #     print("------use agent specific global state------")
        #     self.critic_input_dim += args.obs_dim
        
        actor_gnn = GnnExtractor(
            input_size=self.actor_input_dim,
            middle_size=args.gnn_middle_dim, 
            output_size=args.gnn_output_dim, 
            n_hops=args.n_hops, 
            is_sn=args.use_spectral_norm
        )
        
        critic_gnn = GnnExtractor(
            input_size=self.critic_input_dim,
            middle_size=args.gnn_middle_dim, 
            output_size=args.gnn_output_dim, 
            n_hops=args.n_hops, 
            is_sn=args.use_spectral_norm
        )

        self.actor = SharedActor(
            shared_net=actor_gnn,
            rnn_input_dim=self.rnn_input_dim, 
            output_size=args.action_dim, 
            num_layers=args.num_layers, 
            hidden_size=args.rnn_hidden_dim,
            is_sn=args.use_spectral_norm
            )
        
        self.critic = SharedCritic(
            shared_net=critic_gnn,
            rnn_input_dim=self.rnn_input_dim, 
            output_size=1, 
            num_layers=args.num_layers, 
            hidden_size=args.rnn_hidden_dim,
            is_sn=args.use_spectral_norm
            )

        pretrain_actor = torch.load('./model/actor.pth', map_location='cpu')
        pretrain_critic = torch.load('./model/critic.pth', map_location='cpu')

        self.actor.load_state_dict(pretrain_actor.state_dict())
        self.critic.load_state_dict(pretrain_critic.state_dict())
        
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        
        # self.actor.load_state_dict(torch.load(args.pretrain_model_cwd + '/pretrain_model.pth'))
        # self.critic.load_state_dict(torch.load('experiment/pretrain_model_0/actor_199999.pth'))

        self.ac_parameters = list(self.critic.shared_net.parameters()) + list(self.actor.shared_net.parameters()) + list(self.actor.GRU.parameters()) + list(self.critic.GRU.parameters()) + list(self.critic.Mean.parameters()) + list(self.actor.Mean.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        
        self.minibuffer = None
        self.args = args

    def train(self, replay_buffer, total_steps):
        self.actor = self.actor.to(self.device)
        # Optimize policy for K epochs:
        adv_list = list()
        v_target_list = list()
        for i in self.args.pursuer_num:  # 5-15
            batch, max_episode_len = replay_buffer.get_training_data(i, self.device)  # Transform the data into tensor
            # Calculate the advantage using GAE
            adv = []
            gae = 0
            with torch.no_grad():  # adv and v_target have no gradient
                # deltas.shape=(batch_size,max_episode_len,N)
                deltas = batch['r'] + self.gamma * batch['v_n'][:, 1:] - batch['v_n'][:, :-1]
                deltas = deltas * batch['active']
                for t in reversed(range(max_episode_len)):
                    gae = deltas[:, t] + self.gamma * self.lamda * gae
                    adv.insert(0, gae)
                adv = torch.stack(adv, dim=1)  # adv.shape(batch_size,max_episode_len,N)
                v_target = adv + batch['v_n'][:, :-1]  # v_target.shape(batch_size,max_episode_len,N)
                # print(deltas)
                # print(batch['active'])
                # print(adv)
                if self.use_adv_norm:  # Trick 1: advantage normalization
                    mean = adv.mean()
                    std = adv.std()
                    adv = (adv - mean) / (std + 1e-5) * batch['active']
                    # print(adv)
            v_target_list.append(v_target)
            adv_list.append(adv)
            """
                Get actor_inputs and critic_inputs
                actor_inputs.shape=(batch_size, max_episode_len, N, actor_input_dim)
                critic_inputs.shape=(batch_size, max_episode_len, N, critic_input_dim)
            """
        object_critics = 0.0
        object_actors = 0.0
        update_time = 0

        self.ac_optimizer.zero_grad()
        
        for i, num in enumerate(self.args.pursuer_num):  # 5-15
            batch, _ = replay_buffer.get_training_data(num, self.device)  # Transform the data into tensor
            for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                # print('index: ', index)
                """
                    Get probs_now and values_now
                    probs_now.shape=(mini_batch_size, max_episode_len, N, action_dim)
                    values_now.shape=(mini_batch_size, max_episode_len, N)
                """
                actor_hidden_state = torch.zeros(
                    size=(self.num_layers, len(index) * num, self.rnn_hidden_dim),
                    dtype=torch.float32,
                    device=self.device
                )
                critic_hidden_state = torch.zeros(
                    size=(self.num_layers, len(index) * num, self.rnn_hidden_dim),
                    dtype=torch.float32,
                    device=self.device
                )
                a_logprob_n_now, dist_entropy = self.actor.get_logprob_and_entropy(batch['state'][index], batch['adj'][index], actor_hidden_state, batch['actor_comm_embedding'][index], batch['a_n'][index])
                # dist_entropy.shape=(mini_batch_size, max_episode_len, N)
                # a_logprob_n_now.shape=(mini_batch_size, max_episode_len, N)
                # batch['a_n'][index].shape=(mini_batch_size, max_episode_len, N)
                values_now = self.critic(batch['state'][index], batch['adj'][index], critic_hidden_state, batch['critic_comm_embedding'][index], mode=1).squeeze(-1)

                ratios = torch.exp(a_logprob_n_now - batch['a_logprob_n'][index].detach())  # Attention! Attention! 'a_log_prob_n' should be detached.
                # ratios.shape=(mini_batch_size, max_episode_len, N)
                surr1 = ratios * adv_list[i][index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv_list[i][index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

                if self.use_value_clip:
                    values_old = batch["v_n"][index, :-1].detach()
                    values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target_list[i][index]
                    values_error_original = values_now - v_target_list[i][index]
                    critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
                else:
                    critic_loss = (values_now - v_target_list[i][index]) ** 2
                critic_loss = (critic_loss * batch['active'][index]).sum() / batch['active'][index].sum()
                
                ac_loss = actor_loss + critic_loss
                ac_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac_parameters, 10.0)
                
                object_critics += critic_loss.item()
                object_actors += actor_loss.item()
                update_time += 1
                
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        
        actor_gradients = self.actor.get_gradients()
        critic_gradients = self.critic.get_gradients()
        
        return object_critics / update_time, object_actors / update_time, actor_gradients, critic_gradients
    
    def lr_decay(self, total_steps):  # Trick 6: learning rate Decay
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for p in self.ac_optimizer.param_groups:
            p['lr'] = lr_now
    
    def explore_env(self, env, buffer_id, p_num, num_episode, map_info):
        win_rate = 0.0
        collision_rate = 0.0
        exp_reward = 0.0
        sample_steps = 0
        self.minibuffer = MiniBuffer(buffer_id, args=self.args)
        self.minibuffer.reset_buffer(p_num=p_num, o_num=env.max_boundary_obstacle_num, p_obs_dim=4, e_obs_dim=4, action_dim=9)
        for k in range(num_episode):
            win_tag, collision, episode_reward, episode_steps = self.run_episode(env, map_info, p_num=p_num, num_episode=k, idx=buffer_id)  # 5 + i % 11
            win_rate += win_tag
            collision_rate += collision
            exp_reward += episode_reward
            sample_steps += episode_steps
        return p_num, win_rate / num_episode, collision_rate / num_episode, exp_reward / num_episode, self.minibuffer, sample_steps
    
    def run_episode(self, env, map_info, p_num=15, num_episode=0, idx=0):  #
        win_tag = False
        collision = False
        episode_reward = 0

        env.reset(p_num=p_num, e_num=1, worker_id=idx, map_info=map_info)
        o_num = env.boundary_obstacle_num
        max_o_num = env.max_boundary_obstacle_num

        # The hidden_state is initialized according to the shape of state
        actor_hidden_state = torch.zeros(size=(self.num_layers, p_num, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        critic_hidden_state = torch.zeros(size=(self.num_layers, p_num, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        # The last_comm_embedding is initialized
        actor_last_comm_embedding = torch.zeros(size=(p_num, 2 * self.gnn_output_dim), dtype=torch.float32, device=self.device)
        critic_last_comm_embedding = torch.zeros(size=(p_num, 2 * self.gnn_output_dim), dtype=torch.float32, device=self.device)
        for episode_step in range(self.args.episode_limit):
            p_state = env.get_team_state(True, False)  # obs_n.shape=(N,obs_dim)
            e_state = env.get_team_state(False, False)
            # the adjacent matrix of pursuer-pursuer, pursuer-obstacle, pursuer-evader
            p_p_adj = env.communicate()  # shape of (p_num, p_num)
            p_o_adj, p_e_adj = env.sensor(evader_pos=e_state)  # shape of (p_num, o_num), (p_num, e_num)
            active = env.get_active()
            # evader_step
            _, __ = env.evader_step(idx)
            # preprocess the state
            state = list()
            for s in range(p_num):
                p_tmp = np.array(p_state)  # shape of (p_num, obs_dim)
                # calculate relative p_state
                p_s = np.array(p_state)  # shape of (p_num, obs_dim)
                p_s = p_s - p_tmp[s]
                # calculate relative e_state for every pursuer and mask
                e_s = np.array(e_state)  # shape of (1, obs_dim)
                e_s = e_s - p_tmp[s]
                e_s = e_s.repeat(p_num, 0)  # repeat, shape of (p_num, obs_dim)
                p_e_adj = np.array(p_e_adj).reshape((p_num, 1))
                e_s = e_s * p_e_adj  # mask
                # calculate relative o_state
                o_s = np.array(env.boundary_obstacles)
                padding_0 = np.zeros(shape=(o_num, 2))  # phi, v
                o_s = np.concatenate([o_s, padding_0], axis=-1)
                o_s = o_s - p_tmp[s]
                # plan 1
                padding_1 = np.zeros(shape=(o_num, 4))  # e_x, e_y, e_phi, e_v
                o_s = np.concatenate([o_s, padding_1], axis=-1)
                padding_2 = np.zeros(shape=(max_o_num - o_num, 8))
                o_s = np.concatenate([o_s, padding_2], axis=0)
                padding_3 = np.zeros(shape=(max_o_num, 1))
                o_s = np.concatenate([padding_3, o_s], axis=-1)
                
                padding_4 = np.ones(shape=(p_num, 1), dtype=np.float64)
                p_s = np.concatenate([padding_4, p_s, e_s], axis=-1)
                relative_state = np.concatenate([p_s, o_s], axis=0)
                state.append(relative_state.tolist())

            # preprocess the adjacency matrix
            p_p_adj = np.array(p_p_adj)
            p_o_adj = np.array(p_o_adj)
            adj = np.concatenate([p_p_adj, p_o_adj], axis=-1)
            # Get actions and the corresponding log probabilities of N agents            
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            adj = torch.as_tensor(adj, dtype=torch.float32).to(self.device)
            # actor_comm_embedding.shape = [p_num, self.gnn_output_dim]
            a_n, a_logprob_n, actor_hidden_state, actor_comm_embedding = self.actor.choose_action(state, adj, actor_hidden_state, actor_last_comm_embedding, deterministic=False)
            v_n, critic_hidden_state, critic_comm_embedding = self.critic(state, adj, critic_hidden_state, critic_last_comm_embedding, mode=0)  # Get the state values (V(s)) of N agents
            # Take a step    
            r, done, info = env.step(a_n.detach().cpu().numpy())  # Take a step
            win_tag = True if done and not env.e_list['0'].active else False
            episode_reward += sum(r)

            r = self.reward_norm[f'{p_num}'](r)  # TODO: Dynamic shape

            # Store the transition
            r = torch.as_tensor(r, dtype=torch.float32).to(self.device)
            active = torch.as_tensor(active, dtype=torch.float32).to(self.device)
            win = torch.as_tensor(win_tag, dtype=torch.float32).to(self.device)
            self.minibuffer.store_transition(
                num_episode, episode_step, state, adj, actor_last_comm_embedding, critic_last_comm_embedding, v_n.flatten(), a_n.flatten(), a_logprob_n.flatten(), r, active, win
            )
            actor_last_comm_embedding = torch.concatenate((actor_last_comm_embedding[:, self.args.gnn_output_dim:], actor_comm_embedding), dim=-1)
            critic_last_comm_embedding = torch.concatenate((critic_last_comm_embedding[:, self.args.gnn_output_dim:], critic_comm_embedding), dim=-1)
            if done:
                break
        collision = env.collision
        # An episode is over, store obs_n, s and avail_a_n in the last step
        p_state = env.get_team_state(True, False)  # obs_n.shape=(N,obs_dim)
        e_state = env.get_team_state(False, False)
        # the adjacent matrix of pursuer-pursuer, pursuer-obstacle, pursuer-evader
        p_p_adj = env.communicate()  # shape of (p_num, p_num)
        p_o_adj, p_e_adj = env.sensor(evader_pos=e_state)  # shape of (p_num, o_num), (p_num, e_num)
        # preprocess the state
        state = list()
        for s in range(p_num):
            p_tmp = np.array(p_state)  # shape of (p_num, obs_dim)
            # calculate relative p_state
            p_s = np.array(p_state)  # shape of (p_num, obs_dim)
            p_s = p_s - p_tmp[s]
            # calculate relative e_state for every pursuer and mask
            e_s = np.array(e_state)  # shape of (1, obs_dim)
            e_s = e_s - p_tmp[s]
            e_s = e_s.repeat(p_num, 0)  # repeat, shape of (p_num, obs_dim)
            p_e_adj = np.array(p_e_adj).reshape((p_num, 1))
            e_s = e_s * p_e_adj  # mask
            # calculate relative o_state
            o_s = np.array(env.boundary_obstacles)
            padding_0 = np.zeros(shape=(o_num, 2))  # phi, v
            o_s = np.concatenate([o_s, padding_0], axis=-1)
            o_s = o_s - p_tmp[s]
            
            # plan 1
            padding_1 = np.zeros(shape=(o_num, 4))  # e_x, e_y, e_phi, e_v
            o_s = np.concatenate([o_s, padding_1], axis=-1)
            padding_2 = np.zeros(shape=(max_o_num - o_num, 8))
            o_s = np.concatenate([o_s, padding_2], axis=0)
            padding_3 = np.zeros(shape=(max_o_num, 1))
            o_s = np.concatenate([padding_3, o_s], axis=-1)
            
            padding_4 = np.ones(shape=(p_num, 1), dtype=np.float64)
            p_s = np.concatenate([padding_4, p_s, e_s], axis=-1)
            relative_state = np.concatenate([p_s, o_s], axis=0)
            state.append(relative_state.tolist())
            
        # preprocess the adjacency matrix
        p_p_adj = np.array(p_p_adj)
        p_o_adj = np.array(p_o_adj)
        adj = np.concatenate([p_p_adj, p_o_adj], axis=-1)
        # Get actions and the corresponding log probabilities of N agents
        state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        adj = torch.as_tensor(adj, dtype=torch.float32).to(self.device)
        v_n, critic_hidden_state, _ = self.critic(state, adj, critic_hidden_state, critic_last_comm_embedding, mode=0)
        self.minibuffer.store_last_value(num_episode, episode_step + 1, v_n.flatten())

        return win_tag, collision, episode_reward, episode_step + 1

    def save_model(self, cwd):
        torch.save(self.actor.state_dict(), cwd + 'actor.pth')
        torch.save(self.critic.state_dict(), cwd + 'critic.pth')

    # def save_best_model(self, cwd):
    #     torch.save(self.actor.state_dict(), cwd + 'actor_best.pth')
    #     torch.save(self.critic.state_dict(), cwd + 'critic_best.pth')

    # def load_pretrain_model(self, pretrain_model_cwd):
    #     self.actor.load_state_dict(torch.load(pretrain_model_cwd + 'pretrain_actor.pth'))
    #     self.critic.load_state_dict(torch.load(pretrain_model_cwd + 'pretrain_actor.pth'))
    