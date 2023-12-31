import time
import torch, hydra
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
from DHGN.normalization import Normalization
from DHGN.replay_buffer import ReplayBuffer

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
    def __init__(self, attribute: 'list[torch.Tensor]', adjacent: 'list[torch.Tensor]', is_critic: bool):
        self.attribute = attribute
        self.adjacent = adjacent
        self.is_critic = is_critic
        
    def __len__(self):
        return len(self.attribute)
    
    def __getitem__(self, idx):
        a1 = self.attribute[0]
        a2 = self.attribute[1]
        a3 = self.attribute[2]
        if self.is_critic:
            adj = torch.ones_like(input=self.adjacent[idx], dtype=self.adjacent[idx].dtype, device=self.adjacent[idx].device)
        else:
            adj = self.adjacent[idx]
        return a1, a2, a3, idx, adj
    

class EmbeddingDataset(Dataset):
    def __init__(self, attribute: list, adjacent: torch.Tensor, is_critic: bool, depth: int):
        self.attribute = attribute
        self.adjacent = adjacent
        self.is_critic = is_critic
        self.depth = depth
    def __len__(self):
        return len(self.attribute)
    
    def __getitem__(self, index):
        idx = self.depth - (index + 1)
        a1 = self.attribute[idx]
        if self.is_critic:
            adj = torch.ones_like(input=self.adjacent, dtype=self.adjacent.dtype, device=self.adjacent.device)
        else:
            adj = self.adjacent
        return a1, index, adj
    
    def update(self, embedding, adjacent):
        del self.attribute[0]
        self.attribute.append(embedding)
        self.adjacent = adjacent


class EmbeddingDataset2(Dataset):
    def __init__(self, attribute: torch.Tensor, adjacent: torch.Tensor, is_critic: bool, depth: int):
        self.attribute = attribute
        self.adjacent = adjacent
        self.is_critic = is_critic
        self.num_episode = adjacent.size()[1]
        self.depth = depth
        
    def __len__(self):
        return self.depth
    
    def __getitem__(self, index):
        idx = self.depth - (index + 1)
        a1 = self.attribute[:, idx:idx + self.num_episode, :, :]
        if self.is_critic:
            adj = torch.ones_like(input=self.adjacent, dtype=self.adjacent.dtype, device=self.adjacent.device)
        else:
            adj = self.adjacent
        return a1, index, adj
        

class DHGN(nn.Module):
    def __init__(self, input_dim, embedding_dim, is_sn: bool, algo_config: Namespace, device):
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
        
        self.semantic_layer = preproc_layer(3 * embedding_dim + input_dim, embedding_dim) if is_sn else nn.Linear(3 * embedding_dim + input_dim, embedding_dim)
        
        # feature: (*, num_agent, num_agent, input_size -> embedding_size)
        for r in range(algo_config.num_relation):
            if r == 0:
                layer = preproc_layer(2 * input_dim, embedding_dim) if is_sn else nn.Linear(2 * input_dim, embedding_dim)
            else:
                layer = preproc_layer(input_dim, embedding_dim) if is_sn else nn.Linear(input_dim, embedding_dim)
            self.MSG_layers.append(layer)

        for k in range(algo_config.depth):
            layer = preproc_layer(2 * embedding_dim, embedding_dim) if is_sn else nn.Linear(2 * embedding_dim, embedding_dim)
            self.FCRA_layers.append(layer)

        # feature: (*, num_agent, num_agent -> 1, embedding_size -> embedding_size)
        # adjacent matrix: (*, num_agent, 1, num_agent)
        # alpha: (*, )
        if self.v_agg == 'mean':
            # for r in range(algo_config.num_relation):
            #     layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
            #     self.AGG_layers[f'AGG_vertex_{r}'] = layer
            layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
            self.AGG_layers[f'AGG_vertex_{0}'] = layer
            self.vertex_aggregate = self.mean_operator
        elif self.v_agg == 'pool':
            for r in range(algo_config.num_relation):
                layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
                self.AGG_layers[f'AGG_vertex_{r}'] = layer
            self.max_pool_layer = CustomMaxPool()
            self.vertex_aggregate = self.pool_operator
        elif self.v_agg == 'att':
            for r in range(algo_config.num_relation):
                layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
                self.AGG_layers[f'AGG_vertex_{r}'] = layer
                alpha = Parameter(torch.zeros(size=(embedding_dim, 1)), required_grad=True)
                self.alpha[f'AGG_vertex_{r}'] = alpha
            self.LeakyReLU = nn.LeakyReLU()
            self.vertex_aggregate = self.att_operator
            
        # if self.s_agg == 'mean':
        #     layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
        #     self.AGG_layers['AGG_semantic'] = layer
        #     self.semantic_aggregate = self.mean_operator
        # elif self.s_agg == 'pool':
        #     layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
        #     self.AGG_layers['AGG_semantic'] = layer
        #     self.semantic_aggregate = self.pool_operator
        # elif self.s_agg == 'att':
        #     layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
        #     self.AGG_layers['AGG_semantic'] = layer
        #     alpha = Parameter(torch.zeros(size=(embedding_dim, 1)), required_grad=True)
        #     self.alpha['AGG_semantic'] = alpha
        #     self.semantic_aggregate = self.att_operator
                
        if self.fcra_agg == 'mean':
            for k in range(algo_config.depth):
                layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
                self.AGG_layers[f'AGG_fcra_{k}'] = layer
            self.fcra_aggregate = self.mean_operator
        elif self.fcra_agg == 'pool':
            for k in range(algo_config.depth):
                layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
                self.AGG_layers[f'AGG_fcra_{k}'] = layer
            self.max_pool_layer = CustomMaxPool()
            self.fcra_aggregate = self.pool_operator
        elif self.fcra_agg == 'att':
            for k in range(algo_config.depth):
                layer = preproc_layer(embedding_dim, embedding_dim) if is_sn else nn.Linear(embedding_dim, embedding_dim)
                self.AGG_layers[f'AGG_fcra_{k}'] = layer
                alpha = Parameter(torch.zeros(size=(embedding_dim, 1)), required_grad=True)
                self.alpha[f'AGG_fcra_{k}'] = alpha
            self.LeakyReLU = nn.LeakyReLU()
            self.fcra_aggregate = self.att_operator

    def fcra(self, h0: torch.Tensor, data_loader: DataLoader) -> torch.Tensor:
        """

        cfg:
            h0 (torch.Tensor): current embeddings (*, num_agent, feature_dim)
            data_loader (Dataset): get historical embeddings
            hk (torch.Tensor): historical embeddings (*, num_agent, feature_dim)
            adjacent_mat (torch.Tensor): (*, num_agent, num_agent)
            historical_embedding (torch.Tensor): (*, num_agent, feature_dim)
        Returns:
            observation: (*, num_agent, 1, feature_dim)
        """
        h = h0
        for (a, idx, adjacent_mat) in data_loader:
            k = idx[0]
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
        cfg:
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
        for (a1, a2, a3, idx, adjacent_mat) in data_loader:
            r = idx[0]
            # message
            if r == 0:
                t_1 = self.coordinate(a1, a1)
                t_2 = self.coordinate(a1, a2)
                expand_size = [-1] * len(t_2.size())
                expand_size[-2] = t_2.size()[-3]
                t_2 = t_2.expand(*expand_size)
                a = torch.concatenate((t_1, t_2), dim=-1)
            elif r == 1:
                a = self.coordinate(a1, a2)
            else:
                a = self.coordinate(a1, a3)

            message = self.message(layer=self.MSG_layers[r], attributes=a)
            # aggregate
            if self.v_agg == 'att':
                adjacent_mat = adjacent_mat.unsqueeze(-2)
                embeddings = self.vertex_aggregate(layer=self.AGG_layers[f'AGG_vertex_{r}'], alpha=self.alpha[f'AGG_vertex_{r}'], message=message, adjacent_mat=adjacent_mat)
            elif self.v_agg == 'mean':
                adjacent_mat = adjacent_mat.unsqueeze(-2)
                embeddings = self.vertex_aggregate(layer=self.AGG_layers[f'AGG_vertex_{0}'], message=message, adjacent_mat=adjacent_mat)
            elif self.v_agg == 'pool':
                embeddings = self.vertex_aggregate(layer=self.AGG_layers[f'AGG_vertex_{r}'], message=message, adjacent_mat=adjacent_mat)
            embeddings_list.append(embeddings)
            
        # vertex_level_embeddings = torch.concatenate(embeddings_list, dim=-2)
        vertex_level_embeddings = torch.concatenate(embeddings_list, dim=-1)
        a1 = a1.unsqueeze(-2)
        vertex_level_embeddings = torch.concatenate([a1, vertex_level_embeddings], axis=-1)    
        """
            vertex_level_embeddings: (*, num_agent, num_relation, feature_dim)
            vertex_level_embeddings: (*, num_agent, num_relation * feature_dim)
            adjacent_mat: (*, num_agent, num_relation)
        Returns:
            semantic_level_embeddings: (*, num_agent, 1, feature_dim)
        """
        # adjacent_mat = torch.ones(size=vertex_level_embeddings.size()[:-1], device=self.device, requires_grad=False)
        # if self.s_agg == 'att':
        #     adjacent_mat = adjacent_mat.unsqueeze(-2)
        #     semantic_level_embeddings = self.semantic_aggregate(layer=self.AGG_layers['AGG_semantic'], alpha=self.alpha['AGG_semantic'], message=vertex_level_embeddings, adjacent_mat=None)
        # elif self.s_agg == 'mean':
        #     adjacent_mat = adjacent_mat.unsqueeze(-2)
        #     semantic_level_embeddings = self.semantic_aggregate(layer=self.AGG_layers['AGG_semantic'], message=vertex_level_embeddings, adjacent_mat=adjacent_mat)
        # elif self.s_agg == 'pool':
        #     semantic_level_embeddings = self.semantic_aggregate(layer=self.AGG_layers['AGG_semantic'], message=vertex_level_embeddings, adjacent_mat=adjacent_mat)
        semantic_level_embeddings = self.semantic_layer(vertex_level_embeddings)
        return semantic_level_embeddings

    def forward(self, attributes: DataLoader, historical_embeddings: DataLoader) -> torch.Tensor:
        """_summary_

        cfg:
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
        
        cfg:
            layer (nn.Linear): the relation-specific linear transform matrix that encodes the attributes into the message
            attributes (torch.Tensor): the attributes that belong to the same relation
            
        Returns:
            torch.Tensor: the message
        """
        message = self.ReLU(layer(attributes))
        return message
        
    def mean_operator(self, layer: nn.Linear, message: torch.Tensor, adjacent_mat: torch.Tensor) -> torch.Tensor:
        """
        cfg:
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
        cfg:
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
        cfg:
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
    def __init__(self, shared_net, rnn_input_dim, action_dim, num_layers, rnn_hidden_dim, is_sn=False):
        super().__init__()
        self.shared_net = shared_net
        self.num_layers = num_layers
        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.GRU = nn.GRU(rnn_input_dim, rnn_hidden_dim, num_layers)
        self.Mean = preproc_layer(rnn_hidden_dim, action_dim) if is_sn else nn.Linear(rnn_hidden_dim, action_dim)

    def forward(self, attributes, historical_embeddings, hidden_state, mode):
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
        embedding = self.shared_net(attributes, historical_embeddings)
                
        if mode == 0:
            # feature = embedding.unsqueeze(0)  # (Num_of_Agent, Middle_Size + Input_Size) => (1, Num_of_Agent, Middle_Size + Input_Size)
            feature, hidden_state = self.GRU(embedding, hidden_state)
            feature = feature.squeeze(0)
        else:
            embedding = embedding.squeeze(0)
            batch = embedding.shape[0]
            steps = embedding.shape[1]
            num_agent = embedding.shape[2]

            feature = embedding.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Middle_Size + Input_Size) => (Steps, Batch, Num_of_Agent, Middle_Size + Input_Size)
            feature = feature.reshape((steps, -1, self.rnn_input_dim))  # (Steps, Batch, Num_of_Agent, Middle_Size + Input_Size) => (Steps, Batch * Num_of_Agent, Middle_Size + Input_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.reshape((steps, batch, num_agent, self.rnn_hidden_dim))  # (Steps, Batch, Num_of_Agent, Hidden_Size) <= (Steps, Batch * Num_of_Agent, Hidden_Size)
            feature = feature.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Hidden_Size) <= (Steps, Batch, Num_of_Agent, Hidden_Size)
        prob = torch.softmax(self.Mean(feature), dim=-1)
        return prob, hidden_state, embedding

    def choose_action(self, attributes, historical_embeddings, hidden_state, deterministic=True):
        prob, hidden_state, embedding = self.forward(attributes, historical_embeddings, hidden_state, mode=0)
        if deterministic:
            action = prob.argmax(dim=-1, keepdim=False)
            return action, hidden_state, embedding
        else:
            dist = Categorical(probs=prob)
            a_n = dist.sample()
            a_logprob_n = dist.log_prob(a_n)
            return a_n, a_logprob_n, hidden_state, embedding

    def get_logprob_and_entropy(self, attributes, historical_embeddings, hidden_state, action):
        prob, _, __ = self.forward(attributes, historical_embeddings, hidden_state, mode=1)
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
    def __init__(self, shared_net, rnn_input_dim, value_dim, num_layers, rnn_hidden_dim, is_sn=False):
        super().__init__()
        self.shared_net = shared_net
        self.num_layers = num_layers
        self.rnn_input_dim = rnn_input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.GRU = nn.GRU(rnn_input_dim, rnn_hidden_dim, num_layers)
        self.Mean = preproc_layer(rnn_hidden_dim, value_dim, is_sn=is_sn)

    def forward(self, attributes, historical_embeddings, hidden_state, mode):
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
        embedding = self.shared_net(attributes, historical_embeddings)
        
        if mode == 0:
            # feature = embedding.unsqueeze(0)  # (Num_of_Agent, Hidden_Size) => (1, Num_of_Agent, Hidden_Size)
            feature, hidden_state = self.GRU(embedding, hidden_state)
            feature = feature.squeeze(0)
            val = self.Mean(feature)
            return val, hidden_state, embedding

        else:
            embedding = embedding.squeeze(0)
            batch = embedding.shape[0]
            steps = embedding.shape[1]
            num_agent = embedding.shape[2]            
            feature = embedding.permute(1, 0, 2, 3)  # (Batch, Steps, Num_of_Agent, Middle_Size) => (Steps, Batch, Num_of_Agent, Middle_Size)
            feature = feature.reshape((steps, -1, self.rnn_input_dim))  # (Steps, Batch, Num_of_Agent, Middle_Size) => (Steps, Batch * Num_of_Agent, Middle_Size)
            feature, hidden_state = self.GRU(feature, hidden_state)
            feature = feature.reshape((steps, batch, num_agent, self.rnn_hidden_dim))  # (Steps, Batch, Num_of_Agent, Hidden_Size) <= (Steps, Batch * Num_of_Agent, Hidden_Size)
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
    def __init__(self, cfg, batch_size, mini_batch_size, agent_type):
        # MAPPO Configurations
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.max_train_steps = cfg.algo.max_train_steps
        self.lr = cfg.algo.lr
        self.gamma = cfg.algo.gamma
        self.lamda = cfg.algo.lamda
        self.epsilon = cfg.algo.epsilon
        self.K_epochs = cfg.algo.epochs
        self.entropy_coef = cfg.algo.entropy_coef
        self.use_grad_clip = cfg.algo.use_grad_clip
        self.use_lr_decay = cfg.algo.use_lr_decay
        self.use_adv_norm = cfg.algo.use_adv_norm
        self.use_value_clip = cfg.algo.use_value_clip
        # Policy Configurations
        self.action_dim = cfg.env.action_dim
        self.input_dim = cfg.env.state_dim
        self.num_layers = cfg.algo.num_layers
        self.embedding_dim = cfg.algo.embedding_dim
        self.rnn_input_dim = cfg.algo.embedding_dim
        self.rnn_hidden_dim = cfg.algo.rnn_hidden_dim
        self.sn = cfg.algo.use_spectral_norm
        if "Learner" in agent_type:
            self.device = torch.device(cfg.algo.learner_device)
        elif "Worker" in agent_type:
            self.device = torch.device(cfg.algo.worker_device)
        else:
            self.device = torch.device(cfg.algo.evaluator_device)

        if cfg.algo.use_reward_norm:
            self.reward_norm = Normalization(shape=cfg.env.num_defender)
        
        encoder = DHGN(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            is_sn=self.sn,
            algo_config=cfg.algo,
            device=self.device
        )
        
        # critic_encoder = DHGN(
        #     input_dim=self.input_dim,
        #     embedding_dim=self.embedding_dim,
        #     is_sn=self.sn,
        #     algo_config=cfg.algo,
        #     device=self.device
        # )
        
        self.depth = cfg.algo.depth
        
        self.actor = SharedActor(
            shared_net=encoder,
            rnn_input_dim=self.rnn_input_dim, 
            action_dim=self.action_dim, 
            num_layers=self.num_layers, 
            rnn_hidden_dim=self.rnn_hidden_dim,
            is_sn=self.sn
            )
        
        self.critic = SharedCritic(
            shared_net=encoder,
            rnn_input_dim=self.rnn_input_dim, 
            value_dim=1, 
            num_layers=self.num_layers, 
            rnn_hidden_dim=self.rnn_hidden_dim,
            is_sn=self.sn
            )

        # pretrain_actor = torch.load('./model/actor.pth', map_location='cpu')
        # pretrain_critic = torch.load('./model/critic.pth', map_location='cpu')

        # self.actor.load_state_dict(pretrain_actor.state_dict())
        # self.critic.load_state_dict(pretrain_critic.state_dict())
        
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)
        
        # self.actor.load_state_dict(torch.load(cfg.pretrain_model_cwd + '/pretrain_model.pth'))
        # self.critic.load_state_dict(torch.load('experiment/pretrain_model_0/actor_199999.pth'))

        # self.ac_parameters = list(self.critic.shared_net.parameters()) + list(self.actor.shared_net.parameters()) + list(self.actor.GRU.parameters()) + list(self.critic.GRU.parameters()) + list(self.critic.Mean.parameters()) + list(self.actor.Mean.parameters())
        self.ac_parameters = list(self.actor.shared_net.parameters()) + list(self.actor.GRU.parameters()) + list(self.critic.GRU.parameters()) + list(self.critic.Mean.parameters()) + list(self.actor.Mean.parameters())
        self.ac_optimizer = torch.optim.Adam(self.ac_parameters, lr=self.lr, eps=1e-5)
        
        self.minibuffer = None
        self.total_step = 0
        self.cfg = cfg

    def train(self, replay_buffer, total_steps):
        self.actor = self.actor.to(self.device)
        # Optimize policy for K epochs:
        batch = replay_buffer.get_training_data(self.device)  # Transform the data into tensor
        # Calculate the advantage using GAE
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # deltas.shape=(batch_size,max_episode_len,N)
            deltas = batch['r'] + self.gamma * batch['v_n'][:, 1:] - batch['v_n'][:, :-1]
            deltas = deltas * batch['active']
            for t in reversed(range(self.cfg.env.max_steps)):
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
                size=(self.num_layers, len(index) * self.cfg.env.num_defender, self.rnn_hidden_dim),
                dtype=torch.float32,
                device=self.device
            )
            critic_hidden_state = torch.zeros(
                size=(self.num_layers, len(index) * self.cfg.env.num_defender, self.rnn_hidden_dim),
                dtype=torch.float32,
                device=self.device
            )
            actor_attribute_dataset = AttributeDataset(attribute=[batch['p_state'][index], batch['e_state'][index], batch['o_state'][index]], adjacent=[batch['p_adj'][index], batch['e_adj'][index], batch['o_adj'][index]], is_critic=False)
            critic_attribute_dataset = AttributeDataset(attribute=[batch['p_state'][index], batch['e_state'][index], batch['o_state'][index]], adjacent=[batch['p_adj'][index], batch['e_adj'][index], batch['o_adj'][index]], is_critic=True)
            actor_embedding_dataset = EmbeddingDataset2(attribute=batch['actor_historical_embedding'][index], adjacent=batch['p_adj'][index], is_critic=False, depth=self.cfg.algo.depth)
            critic_embedding_dataset = EmbeddingDataset2(attribute=batch['critic_historical_embedding'][index], adjacent=batch['p_adj'][index], is_critic=True, depth=self.cfg.algo.depth)

            actor_attribute_dataloader = DataLoader(dataset=actor_attribute_dataset, batch_size=1, shuffle=False)
            critic_attribute_dataloader = DataLoader(dataset=critic_attribute_dataset, batch_size=1, shuffle=False)
            actor_embedding_dataloader = DataLoader(dataset=actor_embedding_dataset, batch_size=1, shuffle=False)
            critic_embedding_dataloader = DataLoader(dataset=critic_embedding_dataset, batch_size=1, shuffle=False)
            
            a_logprob_n_now, dist_entropy = self.actor.get_logprob_and_entropy(actor_attribute_dataloader, actor_embedding_dataloader, actor_hidden_state, batch['a_n'][index])
            # dist_entropy.shape=(mini_batch_size, max_episode_len, N)
            # a_logprob_n_now.shape=(mini_batch_size, max_episode_len, N)
            # batch['a_n'][index].shape=(mini_batch_size, max_episode_len, N)
            values_now = self.critic(critic_attribute_dataloader, critic_embedding_dataloader, critic_hidden_state, mode=1).squeeze(-1)

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
                torch.nn.utils.clip_grad_norm_(self.ac_parameters, 5.0)
            
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
        self.total_step = total_steps

    def explore_env(self, env, num_episode):
        exp_reward = 0.0
        sample_steps = 0
        self.minibuffer = ReplayBuffer(cfg=self.cfg)
        self.minibuffer.reset_buffer()
        for k in range(num_episode):
            episode_reward, episode_steps = self.run_episode(env, num_episode=k)  # 5 + i % 11
            exp_reward += episode_reward
            sample_steps += episode_steps
        return exp_reward / num_episode, self.minibuffer, sample_steps
    
    def run_episode(self, env, num_episode=0):  #
        episode_reward = 0
        env.reset()
        p_num = env.num_defender
        # The hidden_state is initialized
        actor_hidden_state = torch.zeros(size=(self.num_layers, p_num, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        critic_hidden_state = torch.zeros(size=(self.num_layers, p_num, self.rnn_hidden_dim), dtype=torch.float32, device=self.device)
        # The historical embedding is initialized
        history_embedding = [torch.zeros(size=(p_num, self.embedding_dim), dtype=torch.float32, device=self.device) for _ in range(self.depth)]
        actor_embedding_dataset = EmbeddingDataset(attribute=history_embedding, adjacent=None, is_critic=False, depth=self.depth)
        critic_embedding_dataset = EmbeddingDataset(attribute=history_embedding, adjacent=None, is_critic=True, depth=self.depth)
        actor_current_embedding = torch.zeros(size=(p_num, self.embedding_dim), dtype=torch.float32, device=self.device)
        critic_current_embedding = torch.zeros(size=(p_num, self.embedding_dim), dtype=torch.float32, device=self.device)
        
        o_state = env.boundary_map.obstacle_agent
        o_ten = torch.as_tensor(o_state, dtype=torch.float32).to(self.device)
        for episode_step in range(env.max_steps):
            p_state = env.get_state(agent_type='defender')  # obs_n.shape=(N,obs_dim)
            e_state = env.get_state(agent_type='attacker')
            
            p_adj = env.communicate()  # shape of (p_num, p_num)
            o_adj, e_adj = env.sensor()  # shape of (p_num, o_num), (p_num, e_num)
            # evader_step
            _ = env.attacker_step()
            # make the dataset
            p_ten = torch.as_tensor(p_state, dtype=torch.float32).to(self.device)
            e_ten = torch.as_tensor(e_state, dtype=torch.float32).to(self.device)
            p_adj_ten = torch.as_tensor(p_adj, dtype=torch.float32).to(self.device)
            e_adj_ten = torch.as_tensor(e_adj, dtype=torch.float32).to(self.device)
            o_adj_ten = torch.as_tensor(o_adj, dtype=torch.float32).to(self.device)

            actor_attribute_dataset = AttributeDataset(attribute=[p_ten, e_ten, o_ten], adjacent=[p_adj_ten, e_adj_ten, o_adj_ten], is_critic=False)
            critic_attribute_dataset = AttributeDataset(attribute=[p_ten, e_ten, o_ten], adjacent=[p_adj_ten, e_adj_ten, o_adj_ten], is_critic=True)
            actor_embedding_dataset.update(embedding=actor_current_embedding, adjacent=p_adj_ten)
            critic_embedding_dataset.update(embedding=critic_current_embedding, adjacent=p_adj_ten)

            actor_attribute_dataloader = DataLoader(dataset=actor_attribute_dataset, batch_size=1, shuffle=False)
            critic_attribute_dataloader = DataLoader(dataset=critic_attribute_dataset, batch_size=1, shuffle=False)
            actor_embedding_dataloader = DataLoader(dataset=actor_embedding_dataset, batch_size=1, shuffle=False)
            critic_embedding_dataloader = DataLoader(dataset=critic_embedding_dataset, batch_size=1, shuffle=False)

            a_n, a_logprob_n, actor_hidden_state, actor_current_embedding = self.actor.choose_action(actor_attribute_dataloader, actor_embedding_dataloader, actor_hidden_state, deterministic=False)
            v_n, critic_hidden_state, critic_current_embedding = self.critic(critic_attribute_dataloader, critic_embedding_dataloader, critic_hidden_state, mode=0)  # Get the state values (V(s)) of N agents
            actor_current_embedding = actor_current_embedding.squeeze(0)
            critic_current_embedding = critic_current_embedding.squeeze(0)
            # Take a step    
            actions = a_n.detach().cpu().numpy()
            # curricula = self.total_step / self.max_train_steps
            # if np.random.rand() > curricula:
            #     action_list = env.demon()
            #     actions = [round((actions[i] + action_list[i]) / 2) for i in range(len(action_list))]
            r, done, info = env.step(actions)  # Take a step
            episode_reward += sum(r)
            r = self.reward_norm(r)  # TODO: Dynamic shape
            # Store the transition
            r = torch.as_tensor(r, dtype=torch.float32).to(self.device)
            active = torch.ones(size=(p_num, ), dtype=torch.float32).to(self.device)
            self.minibuffer.store_transition(
                num_episode, episode_step, p_ten, e_ten, o_ten, p_adj_ten, e_adj_ten, o_adj_ten, actor_current_embedding, critic_current_embedding, v_n.flatten(), a_n.flatten(), a_logprob_n.flatten(), r, active
            )
            if done:
                break
            
        # collision = env.collision
        # An episode is over, store obs_n, s and avail_a_n in the last step
        p_state = env.get_state(agent_type='defender')  # obs_n.shape=(N,obs_dim)
        e_state = env.get_state(agent_type='attacker')
        # the adjacent matrix of pursuer-pursuer, pursuer-obstacle, pursuer-evader
        p_adj = env.communicate()  # shape of (p_num, p_num)
        o_adj, e_adj = env.sensor()  # shape of (p_num, o_num), (p_num, e_num)            
        # make the dataset
        p_ten = torch.as_tensor(p_state, dtype=torch.float32).to(self.device)
        e_ten = torch.as_tensor(e_state, dtype=torch.float32).to(self.device)
        p_adj_ten = torch.as_tensor(p_adj, dtype=torch.float32).to(self.device)
        e_adj_ten = torch.as_tensor(e_adj, dtype=torch.float32).to(self.device)
        o_adj_ten = torch.as_tensor(o_adj, dtype=torch.float32).to(self.device)

        critic_attribute_dataset = AttributeDataset(attribute=[p_ten, e_ten, o_ten], adjacent=[p_adj_ten, e_adj_ten, o_adj_ten], is_critic=True)
        critic_embedding_dataset.update(embedding=critic_current_embedding, adjacent=p_adj_ten)
        critic_attribute_dataloader = DataLoader(dataset=critic_attribute_dataset, batch_size=1, shuffle=False)
        critic_embedding_dataloader = DataLoader(dataset=critic_embedding_dataset, batch_size=1, shuffle=False)

        v_n, critic_hidden_state, critic_current_embedding = self.critic(critic_attribute_dataloader, critic_embedding_dataloader, critic_hidden_state, mode=0)  # Get the state values (V(s)) of N agents
        self.minibuffer.store_last_value(num_episode, episode_step + 1, v_n.flatten())

        return episode_reward, episode_step + 1

    def save_model(self, cwd):
        torch.save(self.actor.state_dict(), cwd + 'actor.pth')
        torch.save(self.critic.state_dict(), cwd + 'critic.pth')
