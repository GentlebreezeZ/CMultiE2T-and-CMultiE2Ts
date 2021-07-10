from copy import deepcopy
from load_data import Data
import torch
import numpy as np
from collections import defaultdict
from model import CMultiE2T
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import time
from collections import defaultdict
from torch.optim.lr_scheduler import ExponentialLR
from datetime import datetime
import random

import config

total_num = len(config.d.entity_idxs) - len(config.d.entity_types_idxs)


def chunks(L, n):
    """ Yield successive n-sized chunks from L."""
    for i in range(0, len(L), n):
        yield L[i:i + n]


class Experiment_CMULTIE2T:
    def __init__(self, param, batch_size=128, learning_rate=0.001, entity_embedding_dim=200,
                 entity_type_embedding_dim=100,
                 epochs=50000, num_filters=200, lmbda=0.2, logger=None, fcn_droupt=0.1):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_entity_dim = entity_embedding_dim
        self.embedding_entity_type_dim = entity_type_embedding_dim
        self.epochs = epochs
        self.num_filters = num_filters
        self.num_entity = len(config.d.entity_idxs)
        self.num_entity_type = len(config.d.entity_types_idxs)
        self.lmbda = lmbda
        self.param = param
        self.logger = logger
        self.fcn_droupt = fcn_droupt

    def set_grad(self):
        self.param.emb_types.requires_grad_(requires_grad=True)
        self.param.trans_matrix.requires_grad = True
        self.param.emb_entities.requires_grad_(requires_grad=True)
        self.param.emb_entitiy_transfer.requires_grad = True
        self.param.emb_type_transfer.requires_grad = True

        self.param.emb_total_emtities.requires_grad_(requires_grad=False)
        self.param.emb_relations.requires_grad_(requires_grad=False)

    def get_batch(self, train_data, train_lable, epoch):
        idxs = np.random.randint(0, len(train_data), self.batch_size)
        new_ets_indexes = np.empty((self.batch_size + self.batch_size, 2)).astype(np.int32)
        new_ets_values = np.empty((self.batch_size + self.batch_size, 1)).astype(np.float32)

        new_ets_indexes[:self.batch_size, :] = train_data[idxs, :]

        negative_ets = [config.one_negative_sampling(tuple(new_ets_indexes[i])) for i in
                        range(self.batch_size)]

        new_ets_indexes[self.batch_size:2 * self.batch_size, :] = np.array(negative_ets)
        new_ets_values[:self.batch_size] = train_lable[idxs, :]
        new_ets_values[self.batch_size:2 * self.batch_size] = np.array(
            [[-1.0] for i in range(self.batch_size)])
        return new_ets_indexes, new_ets_values

    # def init(self):

    def train(self):
        self.train_data = {key: [1] for key in config.d.train_type_idxs}
        self.train_i_indexes = np.array(list(self.train_data.keys()))
        self.train_i_lables = np.array(list(self.train_data.values()))

        if config.args.load == 'True':
            model = torch.load(config.args.outdir + 'ce2t_model.pth')
        else:
            # def __init__(self,entity_type_embedding_size,param, num_filters,fcn_droupt = 0.2):
            model = CMultiE2T(entity_type_embedding_size=self.embedding_entity_type_dim, num_filters=self.num_filters,
                              param=self.param, fcn_droupt=self.fcn_droupt)
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=float(self.learning_rate))
        batchs_num = int(len(config.d.train_type_idxs) / self.batch_size) + 1
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, y_batch = self.get_batch(self.train_i_indexes, self.train_i_lables, batch)
                pred, l2_reg = model.forward(x_batch)
                pred = pred * torch.FloatTensor(y_batch).cuda() * (-1)
                loss = model.loss(pred).mean() + self.lmbda * l2_reg  # Softplus
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            if epoch % 1850 == 0:
                # torch.save(self.param, config.args.outdir + '2000param.pth')
                torch.save(model, config.args.outdir + '1850ce2t_model.pth')
                # reset opt
                opt = torch.optim.Adam(model.parameters(), lr=float(0.000001))
            # if epoch == 2100:
            #     # torch.save(self.param, config.args.outdir + '2300param.pth')
            #     torch.save(model, config.args.outdir + '2300ce2t_model.pth')
            #     # reset opt
            #     #opt = torch.optim.Adam(model.parameters(), lr=float(self.learning_rate / (15 * 15)))
            self.logger.info('-------cmultie2t-------epoch: {} train_loss: {}'.format(epoch, train_loss))
        self.logger.info('-------cmultie2t training is cmopleted-------')
        # save
        torch.save(model, config.args.outdir + 'cmultie2t_model.pth')
        return model

    def train_second(self, model, epochs, lr):
        model.cuda()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        batchs_num = int(len(config.d.train_type_idxs) / self.batch_size) + 1
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, y_batch = self.get_batch(self.train_i_indexes, self.train_i_lables, batch)
                pred, l2_reg = model.forward(x_batch)
                pred = pred * torch.FloatTensor(y_batch).cuda() * (-1)
                loss = model.loss(pred).mean() + self.lmbda * l2_reg  # Softplus
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            self.logger.info('-------cmultie2t-------epoch: {} train_loss: {}'.format(epoch, train_loss))
        self.logger.info('-------cmultie2t training is cmopleted-------')
        return model

    def evaluate_rank_and_hits(self, model, data):
        with torch.no_grad():
            model.cuda()
            model.eval()
            test_data = list(data)
            range_of_type = np.arange(len(config.d.entity_types_idxs)) + total_num

            over_num_type = 0
            hit1_type = 0
            hit3_type = 0
            hit10_type = 0
            mrr_type = 0.0
            index = 0
            for fact in test_data:
                tiled_fact = None
                correct_index = fact[1]
                index += 1
                tiled_fact = np.array(fact * len(config.d.entity_types_idxs)).reshape(len(config.d.entity_types_idxs),
                                                                                      -1)
                tiled_fact[:, 1] = range_of_type

                over_num_type += 1

                tiled_fact = list(chunks(tiled_fact, 128))
                pred = model.predict(tiled_fact[0])
                for batch_it in range(1, len(tiled_fact)):
                    pred_tmp = model.predict(tiled_fact[batch_it])
                    pred = torch.cat((pred, pred_tmp))

                sorted_pred = torch.argsort(pred, dim=0, descending=True)

                position_of_correct_fact_in_sorted_pred = 0
                for tmpxx in sorted_pred:
                    if tmpxx == correct_index:
                        break
                    tmp_list = deepcopy(fact)
                    tmp_list = list(tmp_list)
                    tmp_list[1] = tmpxx.item()
                    tmp_et = tuple(tmp_list)
                    if tmp_et in config.d.over_data:
                        continue
                    else:
                        position_of_correct_fact_in_sorted_pred += 1
                if position_of_correct_fact_in_sorted_pred == 0:
                    hit1_type += 1
                    hit3_type += 1
                    hit10_type += 1
                elif position_of_correct_fact_in_sorted_pred <= 2:
                    hit3_type += 1
                    hit10_type += 1
                elif position_of_correct_fact_in_sorted_pred <= 9:
                    hit10_type += 1
                mrr_type += float(1 / (position_of_correct_fact_in_sorted_pred + 1))
                print('index:', index, 'position: ', position_of_correct_fact_in_sorted_pred)
            self.logger.info(
                'mrr: {},hit1:{},hit3:{},hit10:{}'.format(float(mrr_type / over_num_type),
                                                          float(hit1_type / over_num_type),
                                                          float(hit3_type / over_num_type),
                                                          float(hit10_type / over_num_type)))
            return over_num_type, hit1_type, hit3_type, hit10_type, mrr_type

    def evaluate_rank_score(self, model, data):
        model.eval()
        range_of_type = np.arange(len(config.d.entity_types_idxs)) + total_num
        tiled_fact = None
        # correct_index = data[1]
        tiled_fact = np.array(data * len(config.d.entity_types_idxs)).reshape(len(config.d.entity_types_idxs),
                                                                              -1)
        tiled_fact[:, 1] = range_of_type
        tiled_fact = list(chunks(tiled_fact, 128))
        pred = model.predict(tiled_fact[0])
        for batch_it in range(1, len(tiled_fact)):
            pred_tmp = model.predict(tiled_fact[batch_it])
            pred = torch.cat((pred, pred_tmp))

        return -pred
