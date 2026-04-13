
from common_utils import nonzero_averaging
from model.attention_layer import *
from model.sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class MTSRNetwork(nn.Module):
    def __init__(self, config):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(MTSRNetwork, self).__init__()

        self.fea_mou_input_dim = config.fea_mou_input_dim
        self.fea_wor_input_dim = config.fea_wor_input_dim
        self.output_dim_per_layer = config.layer_fea_output_dim
        self.num_heads_OAB = config.num_heads_OAB
        self.num_heads_MAB = config.num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(config.num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = config.dropout_prob

        self.cross_attention = CrossAttention(128, 4)
        self.cross_attention2 = CrossAttention(128, 4)

        self.selfAttention = SelfAttention(128, 4)
        self.selfAttention2 = SelfAttention(128, 4)

        num_heads_OAB_per_layer = [1] + self.num_heads_OAB
        num_heads_MAB_per_layer = [1] + self.num_heads_MAB

        mid_dim = self.output_dim_per_layer[:-1]

        j_input_dim_per_layer = [self.fea_mou_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_wor_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.worker_attention_blocks = torch.nn.ModuleList()

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.worker_attention_blocks.append(
                MultiHeadworkerAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

    def forward(self, fea_mou, task_mask, candidate, fea_wor, worker_mask, comp_idx):

        sz_b, M, _, J = comp_idx.size()

        fea_mou = self.selfAttention(fea_mou)
        fea_wor = self.selfAttention1(fea_wor)

        fea_mou_new = self.cross_attention(fea_mou, fea_wor) + fea_mou
        fea_wor_new = self.cross_attention2(fea_wor, fea_mou) + fea_wor

        fea_mou_global = nonzero_averaging(fea_mou_new)
        fea_wor_global = nonzero_averaging(fea_wor_new)

        return fea_mou_new, fea_wor_new, fea_mou_global, fea_wor_global


class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.addition = nn.Linear(in_features=8, out_features=128)
        self.addition2 = nn.Linear(in_features=8, out_features=128)
        self.descend = nn.Linear(in_features=128, out_features=8)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.layer_norm = nn.LayerNorm(8)

    def forward(self, value, context, mask=None):

        value = self.sigmoid(self.addition(value))
        context = self.sigmoid(self.addition2(context))

        value = value.permute(1, 0, 2)
        context = context.permute(1, 0, 2)

        attn_output, _ = self.attention(value, context, context, key_padding_mask=mask)

        attn_output = self.sigmoid(self.descend(attn_output))

        return attn_output.permute(1, 0, 2)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)

    def forward(self, value, mask=None):
        value = value.permute(0, 1, 2)
        attn_output, _ = self.attention(value, value, value, key_padding_mask=mask)
        return attn_output.permute(1, 0, 2)


class MemoryNetwork_Fin(nn.Module):
    def __init__(self, hidden_dim=128):
        super(MemoryNetwork_Fin, self).__init__()
        self.hidden_dim = hidden_dim


        self.fc1 = nn.Linear(2, hidden_dim)
        self.local_ratio = nn.Linear(hidden_dim, 1)

        self.fc2 = nn.Linear(2, hidden_dim)
        self.global_ratio = nn.Linear(hidden_dim, 1)
        self.cache = None

        self.fc3 = nn.Linear(hidden_dim, 1)


    def forward(self, worker_running_time, worker_resting_time, worker_memory_time, done_flag):
        b, J, M, T = worker_running_time.shape
        input_tensor = torch.cat((worker_running_time.unsqueeze(4), worker_resting_time.unsqueeze(4)), dim=-1)
        input_tensor = input_tensor.view(b, J * M, T, 2)
        embedding = F.relu(self.fc1(input_tensor))


        local_memory = torch.zeros(b, J * M, 128).to("cuda")
        for i in range(T):
            local_memory += torch.sigmoid(self.local_ratio(embedding[:, :, i, :])) * embedding[:, :, i, :]


        worker_memory_time = worker_memory_time.view(b, J * M, 2)

        h_memory = F.relu(self.fc2(worker_memory_time))

        worker_memory_time_mask = (worker_memory_time > 0).int()

        global_ratio = torch.sigmoid(self.global_ratio(h_memory)) * torch.unsqueeze(worker_memory_time_mask[:, :, 0], 2)

        if done_flag:
            self.cache = None

        if self.cache is None:
            self.cache = torch.zeros_like(h_memory)

        self.cache = global_ratio * h_memory + self.cache

        x = local_memory + self.cache

        fatigue_prediction = self.fc3(x).view(b, J, M)

        return fatigue_prediction


class MTSR(nn.Module):
    def __init__(self, config):

        super(MTSR, self).__init__()
        device = torch.device(config.device)
        self.pair_input_dim = 8

        self.embedding_output_dim = config.layer_fea_output_dim[-1]

        self.feature_exact = MTSRNetwork(config).to(
            device)

        self.feature_worker_fatigue_exact = MemoryNetwork_Fin().to(device)

        self.actor = Actor(config.num_mlp_layers_actor, (4 * self.embedding_output_dim + self.pair_input_dim) * 2,
                           config.hidden_dim_actor, 1).to(device)

        self.critic = Critic(config.num_mlp_layers_critic, 2 * self.embedding_output_dim, config.hidden_dim_critic,
                             1).to(device)

        self.activative = torch.tanh
        self.fatigue_weight = nn.Parameter(torch.ones(1))


    def forward(self, fea_mou, op_mask, candidate, fea_wor, worker_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                worker_fatigue_time_tensor, worker_memory_time_tensor, done_flag):

        worker_fatigue_time_tensor = worker_fatigue_time_tensor.to(torch.float32)

        worker_memory_time_tensor = worker_memory_time_tensor.to(torch.float32)


        worker_fatigue_time_tensor = self.feature_worker_fatigue_exact(worker_fatigue_time_tensor[:, :, :, :, 0],
                                                                         worker_fatigue_time_tensor[:, :, :, :, 1], worker_memory_time_tensor, done_flag)

        fea_mou, fea_wor, fea_mou_global, fea_wor_global = self.feature_exact(fea_mou, op_mask, candidate, fea_wor, worker_mask,
                                                                      comp_idx)

        sz_b, M, _, J = comp_idx.size()
        d = fea_mou.size(-1)

        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        fea_mou_JC = torch.gather(fea_mou, 1, candidate_idx)

        fea_mou_JC_serialized = fea_mou_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        fea_wor_serialized = fea_wor.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_mou_global.unsqueeze(1).expand_as(fea_mou_JC_serialized)
        Fea_Gm_input = fea_wor_global.unsqueeze(1).expand_as(fea_mou_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)

        candidate_feature = torch.cat((fea_mou_JC_serialized, fea_wor_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)

        worker_fatigue_time_tensor = worker_fatigue_time_tensor.reshape(sz_b, M * J).unsqueeze(2).expand_as(
            candidate_feature)

        candidate_feature = torch.cat((worker_fatigue_time_tensor, candidate_feature), dim=-1)

        candidate_scores = self.actor(candidate_feature)

        candidate_scores = candidate_scores.squeeze(-1)

        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)

        global_feature = torch.cat((fea_mou_global, fea_wor_global), dim=-1)
        v = self.critic(global_feature)
        return pi, v
