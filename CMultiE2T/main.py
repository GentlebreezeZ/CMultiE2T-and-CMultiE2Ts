from copy import deepcopy
import torch
import numpy as np
from model import Mymodel
from datetime import datetime
import random

import config

from logger_init import get_logger

logger = get_logger('train', True, file_log=True)
logger.info('START TIME : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def chunks(L, n):
    """ Yield successive n-sized chunks from L."""
    for i in range(0, len(L), n):
        yield L[i:i + n]


class Experiment:
    def __init__(self, batch_size=128, learning_rate=0.001, entity_embedding_dim=200, entity_type_embedding_dim=100,
                 epochs=50000, num_filters=200, lmbda=0.2, droupt=0.2):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_entity_dim = entity_embedding_dim
        self.embedding_entity_type_dim = entity_type_embedding_dim
        self.epochs = epochs
        self.num_filters = num_filters
        self.num_entity = len(config.d.entity_idxs)
        self.num_entity_type = len(config.d.entity_types_idxs)
        self.lmbda = lmbda
        self.droupt = droupt
        logger.info('batch_size: {} '.format(batch_size))
        logger.info('learning_rate: {} '.format(learning_rate))
        logger.info('num_filters: {} '.format(num_filters))
        logger.info('droupt: {} '.format(droupt))
        logger.info('lmbda: {} '.format(lmbda))

    def get_batch(self, train_data, train_lable, epoch):
        # r = min((epoch + 1) * self.batch_size, len(train_data))
        # last_idx = r - epoch * self.batch_size
        #
        # new_ets_indexes = np.empty((last_idx + last_idx, 2)).astype(np.int32)
        # new_ets_values = np.empty((last_idx + last_idx, 1)).astype(np.float32)
        #
        # new_ets_indexes[:r - epoch * self.batch_size, :] = train_data[epoch * self.batch_size: r, :]
        #
        # negative_ets = [config.one_negative_sampling(tuple(new_ets_indexes[i])) for i in
        #                 range(r - epoch * self.batch_size)]
        #
        # new_ets_indexes[last_idx:2 * last_idx, :] = np.array(negative_ets)
        # new_ets_values[:r - epoch * self.batch_size] = train_lable[:r - epoch * self.batch_size, :]
        # new_ets_values[last_idx:2 * last_idx] = np.array([[-1.0] for i in range(r - epoch * self.batch_size)])
        # return new_ets_indexes, new_ets_values
        idxs = np.random.randint(0, len(train_data), config.args.batchsize)
        # r = min((epoch + 1) * config.args.batchsize, len(train_data))
        # last_idx = r - epoch * config.args.batchsize

        new_ets_indexes = np.empty((config.args.batchsize + config.args.batchsize, 2)).astype(np.int32)
        new_ets_values = np.empty((config.args.batchsize + config.args.batchsize, 1)).astype(np.float32)

        new_ets_indexes[:config.args.batchsize, :] = train_data[idxs, :]

        negative_ets = [config.one_negative_sampling(tuple(new_ets_indexes[i])) for i in
                        range(config.args.batchsize)]

        new_ets_indexes[config.args.batchsize:2 * config.args.batchsize, :] = np.array(negative_ets)
        new_ets_values[:config.args.batchsize] = train_lable[idxs, :]
        new_ets_values[config.args.batchsize:2 * config.args.batchsize] = np.array([[-1.0] for i in range(config.args.batchsize)])
        return new_ets_indexes, new_ets_values

    def train_and_eval(self):
        self.train_data = {key: [1] for key in config.d.train_idxs}
        self.train_i_indexes = np.array(list(self.train_data.keys()))
        self.train_i_lables = np.array(list(self.train_data.values()))
        if config.args.load == 'True':
            model = torch.load(config.args.outdir + '802model.pth')
        else:
            model = Mymodel(self.num_entity, self.num_entity_type, self.embedding_entity_dim,
                            self.embedding_entity_type_dim, self.droupt,self.num_filters)
            model.init()
        model.cuda()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        opt = torch.optim.Adam(model.parameters(), lr=float(self.learning_rate))

        batchs_num = int(len(config.d.train_idxs) / self.batch_size) + 1
        for epoch in range(self.epochs):
            model.train()
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, y_batch = self.get_batch(self.train_i_indexes, self.train_i_lables, batch)
                pred, l2_reg = model.forward(x_batch)
                pred = pred * torch.FloatTensor(y_batch).cuda() * (-1)
                loss = model.loss(pred).mean() + config.args.lmbda * l2_reg  # Softplus
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                # print('epoch: ', epoch, 'train_loss: ', train_loss)
            logger.info('epoch: {} train_loss: {}'.format(epoch, train_loss))
            model.eval()
            with torch.no_grad():
                if epoch % 401 == 0 and epoch > 0:
                    torch.save(model, config.args.outdir + str(epoch) + 'model.pth')
                if epoch == 1550:
                    torch.save(model, config.args.outdir + str(epoch) + 'model.pth')
                    self.evaluate_rank_and_hits(model, config.d.test_idxs)
                    opt = torch.optim.Adam(model.parameters(), lr=float(0.000001))
                if epoch == 1850:
                    torch.save(model, config.args.outdir + str(epoch) + 'model.pth')
                    self.evaluate_rank_and_hits(model, config.d.test_idxs)
                    opt = torch.optim.Adam(model.parameters(), lr=float(0.0000001))
                if epoch >= 1900 and epoch % 15 ==0:
                    torch.save(model, config.args.outdir + str(epoch) + 'model.pth')
                    self.evaluate_rank_and_hits(model, config.d.test_idxs)
                # if epoch > 50:
                #     torch.save(model, config.args.outdir + str(epoch) + 'model.pth')
                #     self.evaluate_rank_and_hits(model, config.d.test_idxs)

    def evaluate_rank_and_hits(self, model, data):
        test_data = list(data)
        range_of_type = np.arange(len(config.d.entity_types_idxs))

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
            tiled_fact = np.array(fact * len(config.d.entity_types_idxs)).reshape(len(config.d.entity_types_idxs), -1)
            tiled_fact[:, 1] = range_of_type

            over_num_type += 1

            if index % 2000 == 0:
                print('剩余测试数据的数量为: ', len(test_data) - index)

            tiled_fact = list(chunks(tiled_fact, 128))
            pred = model.pred_evalation(tiled_fact[0])
            for batch_it in range(1, len(tiled_fact)):
                pred_tmp = model.pred_evalation(tiled_fact[batch_it])
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
            # print('index:', index, 'position: ', position_of_correct_fact_in_sorted_pred)
        logger.info(
            'mrr: {},hit1:{},hit3:{},hit10:{}'.format(float(mrr_type / over_num_type), float(hit1_type / over_num_type),
                                                      float(hit3_type / over_num_type),
                                                      float(hit10_type / over_num_type)))
        return over_num_type, hit1_type, hit3_type, hit10_type, mrr_type

    def create_positive_negative_samples(self, data):
        data = np.array(list(data))

        new_ets_lables = np.empty((1, len(data))).astype(np.float32)
        new_ets_lables[:, :int(len(data) / 2)] = np.array([1.0 for i in range(int(len(data) / 2))])
        new_ets_lables[:, int(len(data) / 2):] = np.array([-1.0 for i in range(int(len(data) / 2))])
        new_ets_lables = new_ets_lables[0]
        negative_ets = [config.one_negative_sampling(tuple(data[i + int(len(data) / 2)])) for i in
                        range(int(len(data) / 2))]

        new_ets_indexes = np.empty((len(data), 2)).astype(np.int32)
        new_ets_indexes[:int(len(data) / 2), :] = data[0:int(len(data) / 2), :]
        new_ets_indexes[int(len(data) / 2):, :] = np.array(negative_ets)
        return new_ets_indexes, new_ets_lables

    def calc_valid_param(self, model, data):
        test_data = list(data)
        test_data = np.array(test_data)
        tiled_fact = list(chunks(test_data, 128))
        pred = model.pred_evalation(tiled_fact[0])
        for batch_it in range(1, len(tiled_fact)):
            pred_tmp = model.pred_evalation(tiled_fact[batch_it])
            pred = torch.cat((pred, pred_tmp))
        min_temp = torch.min(pred, dim=0)

        return min_temp.values.data.item()

    def classification_et_instance(self, model, lables, values, min_param):
        true_classification_num = 0

        test_data = list(lables)
        test_data = np.array(test_data)
        tiled_fact = list(chunks(test_data, 128))
        pred = model.pred_evalation(tiled_fact[0])
        for batch_it in range(1, len(tiled_fact)):
            pred_tmp = model.pred_evalation(tiled_fact[batch_it])
            pred = torch.cat((pred, pred_tmp))
        pred = pred.numpy()

        values_list = []
        for j in range(len(lables)):
            if pred[j] >= min_param:
                values_list.append(1.0)
            else:
                values_list.append(-1.0)

        for i in range(len(values)):
            if values_list[i] == values[i]:
                true_classification_num += 1

        print(float(true_classification_num / len(lables)))


if __name__ == '__main__':
    experiment = Experiment(batch_size=config.args.batchsize, learning_rate=config.args.learningrate,
                            entity_embedding_dim=config.args.embsize_entity,
                            entity_type_embedding_dim=config.args.embsize_entity_type, epochs=config.args.epochs,
                            num_filters=config.args.num_filters, lmbda=config.args.lmbda, droupt=config.args.droupt)
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    experiment.train_and_eval()
