import torch, math, itertools, os
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from itertools import permutations, product

from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from itertools import chain
import config

total_num = (len(config.d.entity_idxs) - len(config.d.entity_types_idxs))


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class CMultiE2T(torch.nn.Module):
    def __init__(self, num_entity, num_entity_type, embedding_entity_size, embedding_type_size, num_filters,
                 fcn_droupt=0.2):
        super(CMultiE2T, self).__init__()
        self.num_filters = num_filters
        self.num_entity = num_entity
        self.num_entity_type = num_entity_type
        self.embedding_entity_size = embedding_entity_size
        self.embedding_type_size = embedding_type_size
        self.fcn_droupt = fcn_droupt

        self.emb_entities = torch.nn.Embedding(self.num_entity, self.embedding_entity_size, padding_idx=0)
        self.emb_types = torch.nn.Embedding(self.num_entity_type, self.embedding_type_size, padding_idx=0)

        self.emb_entitiy_transfer = torch.nn.Parameter(torch.Tensor(self.num_entity, self.embedding_entity_size))
        self.emb_type_transfer = torch.nn.Parameter(torch.Tensor(self.num_entity_type, self.embedding_type_size))
        self.trans_matrix = torch.nn.Parameter(torch.Tensor(self.embedding_entity_size, self.embedding_type_size))

        self.f_FCN_net = torch.nn.Linear(num_filters * (self.embedding_type_size), 1)
        xavier_normal_(self.f_FCN_net.weight.data)
        zeros_(self.f_FCN_net.bias.data)

        self.conv1 = torch.nn.Conv2d(1, num_filters, (2, 1))
        zeros_(self.conv1.bias.data)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)
        self.dropout = torch.nn.Dropout(fcn_droupt)

        self.loss = torch.nn.Softplus()

    def init(self):
        torch.nn.init.xavier_uniform_(self.emb_types.weight)
        torch.nn.init.xavier_uniform_(self.emb_entities.weight)

        torch.nn.init.xavier_uniform_(self.emb_entitiy_transfer.data)
        torch.nn.init.xavier_uniform_(self.emb_type_transfer.data)
        # stdv_entity = 1. / math.sqrt(self.emb_entity_size)
        # self.emb_entitiy_transfer.data.uniform_(-stdv_entity, stdv_entity)
        # stdv_type = 1. / math.sqrt(self.emb_type_size)
        # self.emb_type_transfer.data.uniform_(-stdv_type, stdv_type)

        stdv_transfer = 1. / math.sqrt(self.embedding_type_size)
        self.trans_matrix.data.uniform_(-stdv_transfer, stdv_transfer)

    def forward(self, x_batch):
        entity_id2x = torch.tensor(x_batch[:, 0], dtype=torch.long).flatten().cuda()
        type_id2x = torch.tensor(x_batch[:, 1] - total_num, dtype=torch.long).flatten().cuda()

        entity_embedding_vec = self.emb_entities(entity_id2x).view(len(x_batch), -1)
        type_embedding_vec = self.emb_types(type_id2x).view(len(x_batch), 1, -1)

        type_embedding_transfer_vec = self.emb_type_transfer[type_id2x].view(len(x_batch), -1)
        entity_embedding_transfer_vec = self.emb_entitiy_transfer[entity_id2x].view(len(x_batch), -1)

        entity_to_type_embedding_vec = (torch.mm(entity_embedding_vec, self.trans_matrix) + torch.sum(
            entity_embedding_vec * entity_embedding_transfer_vec, -1, True) * type_embedding_transfer_vec).unsqueeze(1)

        et_concat_vec = torch.cat((entity_to_type_embedding_vec, type_embedding_vec), dim=1).unsqueeze(1)

        # et_concat_vec = self.batchNorm0(et_concat_vec)
        et_concat_conv_vec = self.conv1(et_concat_vec)
        et_concat_conv_vec = self.batchNorm1(et_concat_conv_vec)
        et_concat_conv_vec = F.relu(et_concat_conv_vec)

        et_concat_fully_vec = et_concat_conv_vec.view(et_concat_conv_vec.shape[0], -1)

        et_concat_fully_vec = self.dropout(et_concat_fully_vec)

        evaluation_score = self.f_FCN_net(et_concat_fully_vec)

        l2_reg = torch.mean(entity_embedding_vec ** 2) + torch.mean(type_embedding_vec ** 2) + torch.mean(
            entity_to_type_embedding_vec ** 2)  # + torch.mean(type_embedding_transfer_vec**2)
        for W in self.conv1.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.f_FCN_net.parameters():
            l2_reg = l2_reg + W.norm(2)
        return evaluation_score, l2_reg

    def predict(self, x_batch):
        entity_id2x = torch.tensor(x_batch[:, 0], dtype=torch.long).flatten().cuda()
        type_id2x = torch.tensor(x_batch[:, 1] - total_num, dtype=torch.long).flatten().cuda()

        entity_embedding_vec = self.emb_entities(entity_id2x).view(len(x_batch), -1)
        type_embedding_vec = self.emb_types(type_id2x).view(len(x_batch), 1, -1)

        type_embedding_transfer_vec = self.emb_type_transfer[type_id2x].view(len(x_batch), -1)
        entity_embedding_transfer_vec = self.emb_entitiy_transfer[entity_id2x].view(len(x_batch), -1)

        entity_to_type_embedding_vec = (torch.mm(entity_embedding_vec, self.trans_matrix) + torch.sum(
            entity_embedding_vec * entity_embedding_transfer_vec, -1, True) * type_embedding_transfer_vec).unsqueeze(1)

        et_concat_vec = torch.cat((entity_to_type_embedding_vec, type_embedding_vec), dim=1).unsqueeze(1)

        # et_concat_vec = self.batchNorm0(et_concat_vec)
        et_concat_conv_vec = self.conv1(et_concat_vec)
        et_concat_conv_vec = self.batchNorm1(et_concat_conv_vec)
        et_concat_conv_vec = F.relu(et_concat_conv_vec)

        et_concat_fully_vec = et_concat_conv_vec.view(et_concat_conv_vec.shape[0], -1)

        et_concat_fully_vec = self.dropout(et_concat_fully_vec)

        evaluation_score = self.f_FCN_net(et_concat_fully_vec)
        return evaluation_score


class TransE_MIX(torch.nn.Module):
    def __init__(self, num_entity, num_relation, num_entity_type, embedding_entity_size, embedding_relation_size,
                 margin, norm='L2'):
        super(TransE_MIX, self).__init__()
        self.num_entity = num_entity
        self.num_entity_type = num_entity_type
        self.num_relation = num_relation
        self.relation_embedding_size = embedding_relation_size
        self.entity_embedding_size = embedding_entity_size
        self.margin = margin
        self.norm = norm

        self.emb_relations = torch.nn.Embedding(self.num_relation, self.relation_embedding_size, padding_idx=0)
        self.emb_total_emtities = torch.nn.Embedding(num_entity + num_entity_type, self.relation_embedding_size,
                                                     padding_idx=0)

        # self.bn0 = torch.nn.BatchNorm1d(embedding_relation_size)
        # self.bn1 = torch.nn.BatchNorm1d(embedding_relation_size)
        self.loss = torch.nn.MarginRankingLoss(self.margin, False).cuda()

    def create_total_embedding_vec(self, e, trans_matrix, t):
        tmp = torch.mm(e, trans_matrix)
        tmp = torch.cat((tmp, t), dim=0)
        self.emb_total_emtities.weight = torch.nn.Parameter(tmp)

    def init(self):
        torch.nn.init.xavier_uniform_(self.emb_relations.weight)

    def set_grad(self):
        self.emb_total_emtities.requires_grad_(requires_grad=False)

    def clac_distance(self, h, r, t):
        if self.norm == 'L2':
            return torch.norm(h + r - t, p=2, dim=1)  # 2为l2范数  1为l1范数
        else:
            return torch.norm(h + r - t, p=1, dim=1)

    def forward(self, x_batch, train_num):
        rel_id2x_positive = torch.tensor(x_batch[0:train_num, 1], dtype=torch.long).flatten().cuda()
        rel_embedding_positive_vec = self.emb_relations(rel_id2x_positive).view(int(len(x_batch) / 2),
                                                                                self.relation_embedding_size)
        rel_id2x_negative = torch.tensor(x_batch[train_num:, 1], dtype=torch.long).flatten().cuda()
        rel_embedding_negative_vec = self.emb_relations(rel_id2x_negative).view(int(len(x_batch) / 2),
                                                                                self.relation_embedding_size)
        # rel_embedding_positive_vec = self.bn0(rel_embedding_positive_vec)
        # rel_embedding_negative_vec = self.bn0(rel_embedding_negative_vec)

        ent_h_positive_id2x = torch.tensor(x_batch[0:train_num, 0], dtype=torch.long).flatten().cuda()
        ent_t_positive_id2x = torch.tensor(x_batch[0:train_num, 2], dtype=torch.long).flatten().cuda()
        ent_h_negative_id2x = torch.tensor(x_batch[train_num:, 0], dtype=torch.long).flatten().cuda()
        ent_t_negative_id2x = torch.tensor(x_batch[train_num:, 2], dtype=torch.long).flatten().cuda()

        ent_h_positive_embedding_vec = self.emb_total_emtities(ent_h_positive_id2x).view(
            int(len(x_batch) / 2),
            self.relation_embedding_size)
        ent_t_positive_embedding_vec = self.emb_total_emtities(ent_t_positive_id2x).view(
            int(len(x_batch) / 2),
            self.relation_embedding_size)
        ent_h_negative_embedding_vec = self.emb_total_emtities(ent_h_negative_id2x).view(
            int(len(x_batch) / 2),
            self.relation_embedding_size)
        ent_t_negative_embedding_vec = self.emb_total_emtities(ent_t_negative_id2x).view(
            int(len(x_batch) / 2),
            self.relation_embedding_size)

        # ent_h_positive_embedding_vec = self.bn1(ent_h_positive_embedding_vec)
        # ent_t_positive_embedding_vec = self.bn1(ent_t_positive_embedding_vec)
        # ent_h_negative_embedding_vec = self.bn1(ent_h_negative_embedding_vec)
        # ent_t_negative_embedding_vec = self.bn1(ent_t_negative_embedding_vec)

        p_score = self.clac_distance(ent_h_positive_embedding_vec, rel_embedding_positive_vec,
                                     ent_t_positive_embedding_vec)
        n_score = self.clac_distance(ent_h_negative_embedding_vec, rel_embedding_negative_vec,
                                     ent_t_negative_embedding_vec)
        return p_score, n_score

    def predict(self, data):
        e, r, t = data
        emb_e_vec = self.emb_total_emtities(torch.tensor(e, dtype=torch.long).cuda())
        emb_r_vec = self.emb_relations(torch.tensor(r, dtype=torch.long).cuda())

        range_of_type = np.arange(len(config.d.entity_types_idxs)) + total_num
        emb_t_vec = self.emb_total_emtities(torch.tensor(range_of_type, dtype=torch.long).cuda())

        # emb_e_vec = self.bn1(emb_e_vec)
        # emb_r_vec = self.bn0(emb_r_vec)
        # emb_t_vec = self.bn1(emb_t_vec)

        score = emb_e_vec + emb_r_vec - emb_t_vec

        if self.norm == 'L2':
            score = torch.norm(score, p=2, dim=1)
        else:
            score = torch.norm(score, p=1, dim=1)

        return score.unsqueeze(1)
