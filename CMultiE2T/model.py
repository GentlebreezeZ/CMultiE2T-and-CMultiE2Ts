import torch, math, itertools, os
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from itertools import permutations, product

from torch.nn.init import xavier_normal_, xavier_uniform_, uniform_, zeros_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

from itertools import chain
import config


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class Mymodel(torch.nn.Module):
    def __init__(self, num_entity, num_entity_type, entity_embedding_size, entity_type_embedding_size, droupt,
                 num_filters=200):
        super(Mymodel, self).__init__()
        self.num_filters = num_filters
        self.emb_entity_size = entity_embedding_size
        self.emb_type_size = entity_type_embedding_size

        # self.batchNorm0 = torch.nn.BatchNorm2d(1, momentum=0.1)
        self.emb_type = torch.nn.Embedding(num_entity_type, entity_type_embedding_size, padding_idx=0)
        self.emb_entities = torch.nn.Embedding(num_entity, entity_embedding_size, padding_idx=0)

        self.emb_entitiy_transfer = torch.nn.Parameter(torch.Tensor(num_entity, entity_embedding_size))
        self.emb_type_transfer = torch.nn.Parameter(torch.Tensor(num_entity_type, entity_type_embedding_size))
        self.trans_matrix = torch.nn.Parameter(torch.Tensor(entity_embedding_size, entity_type_embedding_size))

        self.f_FCN_net = torch.nn.Linear(num_filters * (entity_type_embedding_size), 1)
        xavier_normal_(self.f_FCN_net.weight.data)
        zeros_(self.f_FCN_net.bias.data)

        self.conv1 = torch.nn.Conv2d(1, num_filters, (2, 1))
        zeros_(self.conv1.bias.data)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_filters, momentum=0.1)
        truncated_normal_(self.conv1.weight, mean=0.0, std=0.1)
        # xavier_normal_(self.conv1.weight.data)

        self.dropout = torch.nn.Dropout(droupt)

        self.loss = torch.nn.Softplus()

    def init(self):
        torch.nn.init.xavier_uniform_(self.emb_type.weight)
        torch.nn.init.xavier_uniform_(self.emb_entities.weight)

        torch.nn.init.xavier_uniform_(self.emb_entitiy_transfer.data)
        torch.nn.init.xavier_uniform_(self.emb_type_transfer.data)
        # stdv_entity = 1. / math.sqrt(self.emb_entity_size)
        # self.emb_entitiy_transfer.data.uniform_(-stdv_entity, stdv_entity)
        # stdv_type = 1. / math.sqrt(self.emb_type_size)
        # self.emb_type_transfer.data.uniform_(-stdv_type, stdv_type)

        stdv_transfer = 1. / math.sqrt(self.emb_type_size)
        self.trans_matrix.data.uniform_(-stdv_transfer, stdv_transfer)

    def forward(self, x_batch):
        entity_id2x = torch.tensor(x_batch[:, 0], dtype=torch.long).flatten().cuda()
        type_id2x = torch.tensor(x_batch[:, 1], dtype=torch.long).flatten().cuda()

        entity_embedding_vec = self.emb_entities(entity_id2x).view(len(x_batch), -1)
        type_embedding_vec = self.emb_type(type_id2x).view(len(x_batch), 1, -1)

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
            entity_to_type_embedding_vec ** 2) #+ torch.mean(type_embedding_transfer_vec**2)
        for W in self.conv1.parameters():
            l2_reg = l2_reg + W.norm(2)
        for W in self.f_FCN_net.parameters():
            l2_reg = l2_reg + W.norm(2)
        return evaluation_score, l2_reg

    def pred_evalation(self, x_batch):
        entity_id2x = torch.tensor(x_batch[:, 0], dtype=torch.long).flatten().cuda()
        type_id2x = torch.tensor(x_batch[:, 1], dtype=torch.long).flatten().cuda()

        entity_embedding_vec = self.emb_entities(entity_id2x).view(len(x_batch), -1)
        type_embedding_vec = self.emb_type(type_id2x).view(len(x_batch), 1, -1)

        type_embedding_transfer_vec = self.emb_type_transfer[type_id2x].view(len(x_batch), -1)
        entity_embedding_transfer_vec = self.emb_entitiy_transfer[entity_id2x].view(len(x_batch), -1)

        entity_to_type_embedding_vec = (torch.mm(entity_embedding_vec, self.trans_matrix) + torch.sum(
            entity_embedding_vec * entity_embedding_transfer_vec, -1,
            True) * type_embedding_transfer_vec).unsqueeze(1)

        et_concat_vec = torch.cat((entity_to_type_embedding_vec, type_embedding_vec), dim=1).unsqueeze(1)

        # et_concat_vec = self.batchNorm0(et_concat_vec)
        et_concat_conv_vec = self.conv1(et_concat_vec)
        et_concat_conv_vec = self.batchNorm1(et_concat_conv_vec)
        et_concat_conv_vec = F.relu(et_concat_conv_vec)

        et_concat_fully_vec = et_concat_conv_vec.view(et_concat_conv_vec.shape[0], -1)

        et_concat_fully_vec = self.dropout(et_concat_fully_vec)

        evaluation_score = self.f_FCN_net(et_concat_fully_vec)
        return evaluation_score
