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
from argparse import Namespace
from torch.nn.utils import spectral_norm
from torch.utils.data import Dataset, DataLoader
from mappo.obstacle_differ_3hop.module.replay_buffer import ReplayBuffer, BigBuffer
from mappo.obstacle_differ_3hop.module.normalization import Normalization


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


class CustomMaxPool(nn.Module):
    def __init__(self):
        super(CustomMaxPool, self).__init__()

    def forward(self, x):
        # Calculate max values along the specified dimension
        _, max_indices = torch.max(x, dim=-1, keepdim=True)
        
        # Mask to retain only the maximum values and zero out others
        mask = torch.zeros_like(x)
        mask.scatter_(1, max_indices, 1)
        
        # Apply the mask and return
        pooled_features = x * mask
        return pooled_features


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


class DHGN(nn.Module):
    def __init__(self, input_size, embedding_size, is_sn: bool, algo_config: Namespace, device):
        super().__init__()
        self.ReLU = nn.ReLU()
        self.MSG_layers = nn.ModuleList()
        self.AGG_layers = nn.ModuleDict()
        self.FCRA_layers = nn.ModuleList()
        self.alpha = nn.ModuleDict()
        self.v_agg = algo_config.vertex_level_aggregator
        self.s_agg = algo_config.semantic_level_aggregator
        self.fcra_agg = algo_config.fcra_aggregator
        self.algo_config = algo_config
        self.device = device
        # feature: (*, num_agent, num_agent, input_size -> embedding_size)
        for r in range(algo_config.num_relation):
            layer = preproc_layer(input_size, embedding_size) if is_sn else nn.Linear(input_size, embedding_size)
            self.MSG_layers.append(layer)

        for k in range(algo_config.depth):
            layer = preproc_layer(input_size, embedding_size) if is_sn else nn.Linear(input_size, embedding_size)
            self.FCRA_layers.append(layer)

        # feature: (*, num_agent, num_agent -> 1, embedding_size -> embedding_size)
        # adjacent matrix: (*, num_agent, 1, num_agent)
        # alpha: (*, )
        if self.v_agg == 'mean':
            for r in range(algo_config.num_relation):
                layer = preproc_layer(embedding_size, embedding_size) if is_sn else nn.Linear(embedding_size, embedding_size)
                self.AGG_layers[f'AGG_vertex_{r}'] = layer
            self.vertex_aggregate = self.mean_operator
        elif self.v_agg == 'pool':
            for r in range(algo_config.num_relation):
                layer = preproc_layer(embedding_size, embedding_size) if is_sn else nn.Linear(embedding_size, embedding_size)
                self.AGG_layers[f'AGG_vertex_{r}'] = layer
            self.max_pool_layer = CustomMaxPool()
            self.vertex_aggregate = self.pool_operator
        elif self.v_agg == 'att':
            for r in range(algo_config.num_relation):
                layer = preproc_layer(embedding_size, embedding_size) if is_sn else nn.Linear(embedding_size, embedding_size)
                self.AGG_layers[f'AGG_vertex_{r}'] = layer
                alpha = Parameter(torch.zeros(size=(embedding_size, 1)), required_grad=True)
                self.alpha[f'AGG_vertex_{r}'] = alpha
            self.LeakyReLU = nn.LeakyReLU()
            self.vertex_aggregate = self.att_operator
            
        if self.s_agg == 'mean':
            layer = preproc_layer(embedding_size, embedding_size) if is_sn else nn.Linear(embedding_size, embedding_size)
            self.AGG_layers['AGG_semantic'] = layer
            self.semantic_aggregate = self.mean_operator
        elif self.s_agg == 'pool':
            layer = preproc_layer(embedding_size, embedding_size) if is_sn else nn.Linear(embedding_size, embedding_size)
            self.AGG_layers['AGG_semantic'] = layer
            self.semantic_aggregate = self.pool_operator
        elif self.s_agg == 'att':
            layer = preproc_layer(embedding_size, embedding_size) if is_sn else nn.Linear(embedding_size, embedding_size)
            self.AGG_layers['AGG_semantic'] = layer
            alpha = Parameter(torch.zeros(size=(embedding_size, 1)), required_grad=True)
            self.alpha['AGG_semantic'] = alpha
            self.semantic_aggregate = self.att_operator
                
        if self.fcra_agg == 'mean':
            for k in range(algo_config.depth):
                layer = preproc_layer(2 * embedding_size, embedding_size) if is_sn else nn.Linear(2 * embedding_size, embedding_size)
                self.AGG_layers[f'AGG_fcra_{k}'] = layer
            self.fcra_aggregate = self.mean_operator
        elif self.fcra_agg == 'pool':
            for k in range(algo_config.depth):
                layer = preproc_layer(2 * embedding_size, embedding_size) if is_sn else nn.Linear(2 * embedding_size, embedding_size)
                self.AGG_layers[f'AGG_fcra_{k}'] = layer
            self.max_pool_layer = CustomMaxPool()
            self.fcra_aggregate = self.pool_operator
        elif self.fcra_agg == 'att':
            for k in range(algo_config.depth):
                layer = preproc_layer(2 * embedding_size, embedding_size) if is_sn else nn.Linear(2 * embedding_size, embedding_size)
                self.AGG_layers[f'AGG_fcra_{k}'] = layer
                alpha = Parameter(torch.zeros(size=(embedding_size, 1)), required_grad=True)
                self.alpha[f'AGG_fcra_{k}'] = alpha
            self.LeakyReLU = nn.LeakyReLU()
            self.fcra_aggregate = self.att_operator

    def fcra(self, h0: torch.Tensor, data_loader: DataLoader) -> torch.Tensor:
        """

        Args:
            h0 (torch.Tensor): current embeddings (*, num_agent, feature_dim)
            data_loader (Dataset): get historical embeddings
            hk (torch.Tensor): historical embeddings (*, num_agent, feature_dim)
            adjacent_mat (torch.Tensor): (*, num_agent, num_agent)
            historical_embedding (torch.Tensor): (*, num_agent, feature_dim)
        Returns:
            observation: (*, num_agent, 1, feature_dim)
        """
        h = h0
        for a, k, adjacent_mat in data_loader:
            if self.fcra_agg == 'att':
                embeddings = self.fcra_aggregate(layer=self.AGG_layers[f'AGG_fcra_{k}'], alpha=self.alpha[f'AGG_fcra_{k}'], message=a, adjacent_mat=adjacent_mat)
            elif self.fcra_agg == 'mean':
                embeddings = self.fcra_aggregate(layer=self.AGG_layers[f'AGG_fcra_{k}'], message=a, adjacent_mat=adjacent_mat)
            elif self.fcra_agg == 'pool':
                num_agent = a.size()[-2]
                a = a.unsqueeze(-2)
                expand_size = [-1] * len(a.size())
                expand_size[-2] = num_agent
                a.unsqueeze(-2).expand(*expand_size).transpose(-2, -3)  # (*, num_agent, feature_dim) -> (*, num_agent, num_agent, feature_dim)
                embeddings = self.fcra_aggregate(layer=self.AGG_layers[f'AGG_fcra_{k}'], message=a, adjacent_mat=adjacent_mat)
                embeddings = embeddings.squeeze(-2)
            FCRA_layer = self.FCRA_layers[k]
            h = self.ReLU(FCRA_layer(torch.concatenate([embeddings, h], dim=-1)))
        return h

    def coordinate(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        t1 = t1.unsqueeze(-2)
        t2 = t2.unsqueeze(-3)
        relative_coordinates = t1 - t2
        return relative_coordinates

    def encoder(self, data_loader: DataLoader) -> torch.Tensor:
        """
        Args:
            data_loader (Dataset): _description_
            attributes: the attributes cooresponding to the relation types
                shape = (*, num_agent, num_agent/obstacle, feature_dim)
            relations: the relation types belong to 0, 1
            adjacent_mat:
                shape = (*, num_agent, num_agent/obstacle)
            
        Returns:
            torch.Tensor: the semantic-level embeddings, also representing the current historical embeddings h0
                shape = (*, num_agnet, feature_dim)
        """
        embeddings_list = list()
        for a1, a2, r, adjacent_mat in data_loader:
            # message
            a = self.coordinate(a1, a2)
            message = self.message(layer=self.MSG_layers[r], attributes=a)
            # aggregate
            if self.v_agg == 'att':
                adjacent_mat = adjacent_mat.unsqueeze(-2)
                embeddings = self.vertex_aggregate(layer=self.AGG_layers['AGG_vertex_{r}'], alpha=self.alpha['AGG_vertex_{r}'], message=message, adjacent_mat=adjacent_mat)
            elif self.v_agg == 'mean':
                adjacent_mat = adjacent_mat.unsqueeze(-2)
                embeddings = self.vertex_aggregate(layer=self.AGG_layers['AGG_vertex_{r}'], message=message, adjacent_mat=adjacent_mat)
            elif self.v_agg == 'pool':
                embeddings = self.vertex_aggregate(layer=self.AGG_layers['AGG_vertex_{r}'], message=message, adjacent_mat=adjacent_mat)
            embeddings_list.append(embeddings)
        vertex_level_embeddings = torch.concatenate(embeddings_list, dim=-2)
        """
            vertex_level_embeddings: (*, num_agent, num_relation, feature_dim)
            adjacent_mat: (*, num_agent, num_relation)
        Returns:
            semantic_level_embeddings: (*, num_agent, 1, feature_dim)
        """
        adjacent_mat = torch.ones(size=vertex_level_embeddings.size()[:-1], device=self.device, requires_grad=False)
        if self.s_agg == 'att':
            adjacent_mat = adjacent_mat.unsqueeze(-2)
            semantic_level_embeddings = self.semantic_aggregate(layer=self.AGG_layers['AGG_semantic'], alpha=self.alpha['AGG_semantic'], message=vertex_level_embeddings, adjacent_mat=None)
        elif self.s_agg == 'mean':
            adjacent_mat = adjacent_mat.unsqueeze(-2)
            semantic_level_embeddings = self.semantic_aggregate(layer=self.AGG_layers['AGG_semantic'], message=vertex_level_embeddings, adjacent_mat=adjacent_mat)
        elif self.s_agg == 'pool':
            semantic_level_embeddings = self.semantic_aggregate(layer=self.AGG_layers['AGG_semantic'], message=vertex_level_embeddings, adjacent_mat=adjacent_mat)
        return semantic_level_embeddings

    def forward(self, attributes: DataLoader, historical_embeddings: DataLoader) -> torch.Tensor:
        """_summary_

        Args:
            attributes (Dataset): 
                (*, num_agent, feature_dim), (*, num_agent/obstacle, feature_dim), r, (*, num_agent, num_agent/obstacle)
            historical_embeddings (Dataset): _description_
                (*, num_agent, feature_dim), r, (*, num_agent, num_agent)
                
        Returns:
            torch.Tensor: encoded_observation (*, num_agent, feature_dim)
        """
        h0 = self.encoder(data_loader=attributes).squeeze(-2)
        observation = self.fcra(h0=h0, data_loader=historical_embeddings)
        observation = observation.squeeze(-2)
        return observation

    def message(self, layer: nn.Linear, attributes: torch.Tensor) -> torch.Tensor:
        """the message operator denoted by MSG() placeholder
        
        Args:
            layer (nn.Linear): the relation-specific linear transform matrix that encodes the attributes into the message
            attributes (torch.Tensor): the attributes that belong to the same relation
            
        Returns:
            torch.Tensor: the message
        """
        message = self.ReLU(layer(attributes))
        return message
        
    def mean_operator(self, layer: nn.Linear, message: torch.Tensor, adjacent_mat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer (nn.Linear): the relation-specific linear transform matrix
            message (torch.Tensor): the message (*, num_agent, num_agent/obstacle, feature_dim)
            adjacent_mat (torch.Tensor): the adjacent matrix (*, num_agent, 1, num_agent/obstacle)

        Returns:
            torch.Tensor: vertex/semantic-level embeddings (*, num_agent, 1, feature_dim)
        """
        adjacent_mat = f.normalize(adjacent_mat, p=1, dim=-1)
        embeddings = self.ReLU(layer(torch.matmul(adjacent_mat, message)))
        return embeddings

    def pool_operator(self, layer: nn.Linear, message: torch.Tensor, adjacent_mat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer (nn.Linear): the relation-specific linear transform matrix
            message (torch.Tensor): the message (*, num_agent, num_agent/obstacle, feature_dim)
            adjacent_mat (torch.Tensor): the adjacent matrix, as a mask (*, num_agent, num_agent/obstacle, 1)

        Returns:
            torch.Tensor: vertex/semantic-level embeddings (*, num_agent, 1, feature_dim)
        """
        adj_1 = adjacent_mat.unsqueeze(-1)
        embeddings = self.max_pool_layer(self.ReLU(layer(adj_1 * message)))
        adj_2 = adjacent_mat.unsqueeze(-2)
        embeddings = torch.matmul(adj_2, embeddings)
        return embeddings
        
    def att_operator(self, layer: nn.Linear, alpha: torch.Tensor, message: torch.Tensor, adjacent_mat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            layer (nn.Linear): the relation-specific linear transform matrix
            alpha (torch.Tensor): the learned weights (*, feature_dim, 1)
            message (torch.Tensor): the message (*, num_agent, num_agent/obstacle, feature_dim)
            adjacent_mat (torch.Tensor): the adjacent matrix, as a mask (*, num_agent, num_agent/obstacle)
            attention_score (torch.Tensor): 
        Returns:
            torch.Tensor: vertex/semantic-level embeddings (*, num_agent, 1, feature_dim)
        """       
        mask = adjacent_mat != 1
        repeat_time = message.size()[:-2]  # (*, )
        alpha = alpha.expand(repeat_time + alpha.size())  
        attention_score = torch.matmul(self.LeakyReLU(layer(message)), alpha)  # (*, num_agent/obstacle, 1)
        attention_score = torch.masked_fill(mask, value=float("-inf"))
        attention_score = f.softmax(attention_score, dim=-1)
        mask = torch.isnan(attention_score)
        attention_score = torch.masked_fill(mask, value=float(0.))
        
        embeddings = self.ReLU(torch.matmul(attention_score, layer(message)))
        return embeddings


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
        embedding = self.shared_net(state, last_comm_embedding, adj)
        # print('comm_embedding.shape', comm_embedding.shape)
        # print('last_comm_embedding.shape', last_comm_embedding.shape)
        # comm_embedding.shape == [*, agent_num, gnn_output_dim]
        
        if mode == 0:
            feature = embedding.unsqueeze(0)  # (Num_of_Agent, Middle_Size + Input_Size) => (1, Num_of_Agent, Middle_Size + Input_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.squeeze(0)
        else:
            batch = embedding.shape[0]
            steps = embedding.shape[1]
            num_agent = embedding.shape[2]

            feature = embedding.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Middle_Size + Input_Size) => (Steps, Batch, Num_of_Agent, Middle_Size + Input_Size)
            feature = feature.reshape((steps, -1, self.rnn_input_size))  # (Steps, Batch, Num_of_Agent, Middle_Size + Input_Size) => (Steps, Batch * Num_of_Agent, Middle_Size + Input_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.reshape((steps, batch, num_agent, self.hidden_size))  # (Steps, Batch, Num_of_Agent, Hidden_Size) <= (Steps, Batch * Num_of_Agent, Hidden_Size)
            feature = feature.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Hidden_Size) <= (Steps, Batch, Num_of_Agent, Hidden_Size)
        prob = torch.softmax(self.Mean(feature), dim=-1)
        return prob, hidden_state, embedding

    def choose_action(self, state, adj, hidden_state, last_comm_embedding, deterministic=True):
        prob, hidden_state, embedding = self.forward(state, adj, hidden_state, last_comm_embedding, mode=0)
        if deterministic:
            action = prob.argmax(dim=-1, keepdim=False)
            return action, hidden_state, embedding
        else:
            dist = Categorical(probs=prob)
            a_n = dist.sample()
            a_logprob_n = dist.log_prob(a_n)
            return a_n, a_logprob_n, hidden_state, embedding

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
        embedding = self.shared_net(state, last_comm_embedding, connected_adj)
        
        if mode == 0:
            feature = embedding.unsqueeze(0)  # (Num_of_Agent, Hidden_Size) => (1, Num_of_Agent, Hidden_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.squeeze(0)
            val = self.Mean(feature)
            return val, hidden_state, embedding

        else:
            batch = embedding.shape[0]
            steps = embedding.shape[1]
            num_agent = embedding.shape[2]            
            feature = embedding.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Middle_Size) => (Steps, Batch, Num_of_Agent, Middle_Size)
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
        self.actor_input_dim = args.state_dim
        self.critic_input_dim = args.state_dim
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
            self.reward_norm = Normalization(shape=args.pursuer_num)

        # if self.use_agent_specific:
        #     print("------use agent specific global state------")
        #     self.critic_input_dim += args.obs_dim
        
        encoder = DHGN(
            input_size=self.input_size,
            embedding_size=args.embedding_size,
            is_sn=args.use_spectral_norm,
            algo_config=args,
            device=self.device
        )
        
        self.actor = SharedActor(
            shared_net=encoder,
            rnn_input_dim=self.rnn_input_dim, 
            output_size=args.action_dim, 
            num_layers=args.num_layers, 
            hidden_size=args.rnn_hidden_dim,
            is_sn=args.use_spectral_norm
            )
        
        self.critic = SharedCritic(
            shared_net=encoder,
            rnn_input_dim=self.rnn_input_dim, 
            output_size=1, 
            num_layers=args.num_layers, 
            hidden_size=args.rnn_hidden_dim,
            is_sn=args.use_spectral_norm
            )

        # pretrain_actor = torch.load('./model/actor.pth', map_location='cpu')
        # pretrain_critic = torch.load('./model/critic.pth', map_location='cpu')

        # self.actor.load_state_dict(pretrain_actor.state_dict())
        # self.critic.load_state_dict(pretrain_critic.state_dict())
        
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        
        # self.actor.load_state_dict(torch.load(args.pretrain_model_cwd + '/pretrain_model.pth'))
        # self.critic.load_state_dict(torch.load('experiment/pretrain_model_0/actor_199999.pth'))

        # self.ac_parameters = list(self.critic.shared_net.parameters()) + list(self.actor.shared_net.parameters()) + list(self.actor.GRU.parameters()) + list(self.critic.GRU.parameters()) + list(self.critic.Mean.parameters()) + list(self.actor.Mean.parameters())
        self.ac_parameters = list(self.actor.shared_net.parameters()) + list(self.actor.GRU.parameters()) + list(self.critic.GRU.parameters()) + list(self.critic.Mean.parameters()) + list(self.actor.Mean.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        
        self.minibuffer = None
        self.args = args

    def train(self, replay_buffer, total_steps):
        self.actor = self.actor.to(self.device)
        # Optimize policy for K epochs:
        batch, max_episode_len = replay_buffer.get_training_data(self.device)  # Transform the data into tensor
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

            if self.use_adv_norm:  # Trick 1: advantage normalization
                mean = adv.mean()
                std = adv.std()
                adv = (adv - mean) / (std + 1e-5) * batch['active']
                
        object_critics = 0.0
        object_actors = 0.0
        update_time = 0
        self.ac_optimizer.zero_grad()
        
        for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
            actor_hidden_state = torch.zeros(
                size=(self.num_layers, len(index) * self.args.num_defender, self.rnn_hidden_dim),
                dtype=torch.float32,
                device=self.device
            )
            critic_hidden_state = torch.zeros(
                size=(self.num_layers, len(index) * self.args.num_defender, self.rnn_hidden_dim),
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
            surr1 = ratios * adv[index]
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
            actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
            actor_loss = (actor_loss * batch['active'][index]).sum() / batch['active'][index].sum()

            if self.use_value_clip:
                values_old = batch["v_n"][index, :-1].detach()
                values_error_clip = torch.clamp(values_now - values_old, -self.epsilon, self.epsilon) + values_old - v_target[index]
                values_error_original = values_now - v_target[index]
                critic_loss = torch.max(values_error_clip ** 2, values_error_original ** 2)
            else:
                critic_loss = (values_now - v_target[index]) ** 2
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
        self.minibuffer = ReplayBuffer(buffer_id, args=self.args)
        self.minibuffer.reset_buffer(p_num=p_num, o_num=env.max_boundary_obstacle_num, p_obs_dim=4, e_obs_dim=4, action_dim=9)
        for k in range(num_episode):
            win_tag, collision, episode_reward, episode_steps = self.run_episode(env, map_info, p_num=p_num, num_episode=k, idx=buffer_id)  # 5 + i % 11
            win_rate += win_tag
            collision_rate += collision
            exp_reward += episode_reward
            sample_steps += episode_steps
        return p_num, win_rate / num_episode, collision_rate / num_episode, exp_reward / num_episode, self.minibuffer, sample_steps
    
    def run_episode(self, env, map_info, p_num=15, num_episode=0, idx=0):  #
        episode_reward = 0
        env.reset(p_num=p_num, e_num=1, worker_id=idx, map_info=map_info)

        # The hidden_state is initialized
        actor_hidden_state = torch.zeros(size=(self.num_layers, p_num, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        critic_hidden_state = torch.zeros(size=(self.num_layers, p_num, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        # The historical embedding is initialized
        history_embedding = [torch.zeros(size=(p_num, self.embedding_size), dtype=torch.float32, device=self.device) for _ in range(self.depth)]
        actor_embedding_dataset = EmbeddingDataset(attribute=history_embedding, adjacent=None)
        critic_embedding_dataset = EmbeddingDataset(attribute=history_embedding, adjacent=None)
        actor_current_embedding = torch.zeros(size=(p_num, self.embedding_size), dtype=torch.float32, device=self.device)
        critic_current_embedding = torch.zeros(size=(p_num, self.embedding_size), dtype=torch.float32, device=self.device)
        
        o_state = env.boundary_obstacles
        o_ten = torch.as_tensor(o_state, dtype=torch.float32).to(self.device)
        for episode_step in range(self.args.episode_limit):
            p_state = env.get_state(agent_type='defender')  # obs_n.shape=(N,obs_dim)
            e_state = env.get_state(agent_type='attacker')
            
            p_adj = env.communicate()  # shape of (p_num, p_num)
            o_adj, e_adj = env.sensor()  # shape of (p_num, o_num), (p_num, e_num)
            # evader_step
            _, __ = env.evader_step(idx)
            # make the dataset
            p_ten = torch.as_tensor(p_state, dtype=torch.float32).to(self.device)
            e_ten = torch.as_tensor(e_state, dtype=torch.float32).to(self.device)
            p_adj_ten = torch.as_tensor(p_adj, dtype=torch.float32).to(self.device)
            e_adj_ten = torch.as_tensor(e_adj, dtype=torch.float32).to(self.device)
            o_adj_ten = torch.as_tensor(o_adj, dtype=torch.float32).to(self.device)

            attribute_dataset = AttributeDataset(attribute=[p_ten, e_ten, o_ten], adjacent=[p_adj_ten, e_adj_ten, o_adj_ten])
            actor_embedding_dataset.update(attribute=actor_current_embedding, adjacent=p_adj_ten)
            critic_embedding_dataset.update(attribute=critic_current_embedding, adjacent=p_adj_ten)

            a_n, a_logprob_n, actor_hidden_state, actor_current_embedding = self.actor.choose_action(attribute_dataset, actor_embedding_dataset, actor_hidden_state, deterministic=False)
            v_n, critic_hidden_state, critic_current_embedding = self.critic(attribute_dataset, critic_embedding_dataset, critic_hidden_state, mode=0)  # Get the state values (V(s)) of N agents
            # Take a step    
            r, done, info = env.step(a_n.detach().cpu().numpy())  # Take a step
            episode_reward += sum(r)
            r = self.reward_norm[f'{p_num}'](r)  # TODO: Dynamic shape
            # Store the transition
            r = torch.as_tensor(r, dtype=torch.float32).to(self.device)
            active = torch.as_tensor(active, dtype=torch.float32).to(self.device)
            self.minibuffer.store_transition(
                num_episode, episode_step, p_ten, e_ten, o_ten, p_adj, e_adj, o_adj, actor_current_embedding, critic_current_embedding, v_n.flatten(), a_n.flatten(), a_logprob_n.flatten(), r, active
            )
            if done:
                break
            
        # collision = env.collision
        # An episode is over, store obs_n, s and avail_a_n in the last step
        p_state = env.get_state(agent_type='defender')  # obs_n.shape=(N,obs_dim)
        e_state = env.get_state(agent_type='attacker')
        # the adjacent matrix of pursuer-pursuer, pursuer-obstacle, pursuer-evader
        p_adj = env.communicate()  # shape of (p_num, p_num)
        o_adj, e_adj = env.sensor(evader_pos=e_state)  # shape of (p_num, o_num), (p_num, e_num)            
        # make the dataset
        p_ten = torch.as_tensor(p_state, dtype=torch.float32).to(self.device)
        e_ten = torch.as_tensor(e_state, dtype=torch.float32).to(self.device)
        p_adj_ten = torch.as_tensor(p_adj, dtype=torch.float32).to(self.device)
        e_adj_ten = torch.as_tensor(e_adj, dtype=torch.float32).to(self.device)
        o_adj_ten = torch.as_tensor(o_adj, dtype=torch.float32).to(self.device)

        attribute_dataset = AttributeDataset(attribute=[p_ten, e_ten, o_ten], adjacent=[p_adj_ten, e_adj_ten, o_adj_ten])
        critic_embedding_dataset.update(attribute=critic_current_embedding, adjacent=p_adj_ten)
        v_n, critic_hidden_state, critic_current_embedding = self.critic(attribute_dataset, critic_embedding_dataset, critic_hidden_state, mode=0)  # Get the state values (V(s)) of N agents
        self.minibuffer.store_last_value(num_episode, episode_step + 1, v_n.flatten())

        return episode_reward, episode_step + 1

    def save_model(self, cwd):
        torch.save(self.actor.state_dict(), cwd + 'actor.pth')
        torch.save(self.critic.state_dict(), cwd + 'critic.pth')
