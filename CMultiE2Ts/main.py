from copy import deepcopy
import torch
import numpy as np
from datetime import datetime
import config
from logger_init import get_logger

from model import CMultiE2T, TransE_MIX

logger = get_logger('train', True, file_log=True)
logger.info('START TIME : {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
total_num = len(config.d.entity_idxs) - len(config.d.entity_types_idxs)


def chunks(L, n):
    """ Yield successive n-sized chunks from L."""
    for i in range(0, len(L), n):
        yield L[i:i + n]


class Experiment():
    def __init__(self, num_entity, num_relation, num_entity_type, embedding_entity_size, embedding_relation_size,
                 embedding_type_size, batch_transe, batch_cmultie2t, transe_epoch, cmultie2t_epoch, transe_lr,
                 cmultie2t_lr, transe_margin, cmultie2t_numfilters, cmultie2t_lmbda, cmultie2t_droupt, norm,
                 is_load='False', lmbda=None):
        self.num_entity = num_entity
        self.num_relation = num_relation
        self.num_entity_type = num_entity_type
        self.embedding_entity_size = embedding_entity_size
        self.embedding_relation_size = embedding_relation_size
        self.embedding_type_size = embedding_type_size
        self.batch_transe = batch_transe
        self.batch_cmultie2t = batch_cmultie2t
        self.transe_epoch = transe_epoch
        self.cmultie2t_epoch = cmultie2t_epoch
        self.transe_lr = transe_lr
        self.cmultie2t_lr = cmultie2t_lr
        self.transe_margin = transe_margin
        self.cmultie2t_numfilters = cmultie2t_numfilters
        self.cmultie2t_lmbda = cmultie2t_lmbda
        self.norm = norm
        self.is_load = is_load
        self.cmultie2t_droupt = cmultie2t_droupt
        self.pred_lmbda = lmbda

        logger.info('------embedding_entity_size------: {}'.format(embedding_entity_size))
        logger.info('------embedding_relation_size------: {}'.format(embedding_relation_size))
        logger.info('------embedding_type_size------: {}'.format(embedding_type_size))
        logger.info('------batch_transe------: {}'.format(batch_transe))
        logger.info('------batch_cmultie2t------: {}'.format(batch_cmultie2t))
        logger.info('------transe_epoch------: {}'.format(transe_epoch))
        logger.info('------cmultie2t_epoch------: {}'.format(cmultie2t_epoch))
        logger.info('------transe_lr------: {}'.format(transe_lr))
        logger.info('------cmultie2t_lr------: {}'.format(cmultie2t_lr))
        logger.info('------transe_margin------: {}'.format(transe_margin))
        logger.info('------cmultie2t_numfilters------: {}'.format(cmultie2t_numfilters))
        logger.info('------cmultie2t_lmbda------: {}'.format(cmultie2t_lmbda))
        logger.info('------norm------: {}'.format(norm))
        logger.info('------cmultie2t_droupt------: {}'.format(cmultie2t_droupt))
        logger.info('------pred_lmbda------: {}'.format(lmbda))

    def get_batch(self, train_data, train_lable, epoch):
        idxs = np.random.randint(0, len(train_data), self.batch_cmultie2t)
        new_ets_indexes = np.empty((self.batch_cmultie2t + self.batch_cmultie2t, 2)).astype(np.int32)
        new_ets_values = np.empty((self.batch_cmultie2t + self.batch_cmultie2t, 1)).astype(np.float32)

        new_ets_indexes[:self.batch_cmultie2t, :] = train_data[idxs, :]

        negative_ets = [config.one_negative_sampling(tuple(new_ets_indexes[i])) for i in
                        range(self.batch_cmultie2t)]

        new_ets_indexes[self.batch_cmultie2t:2 * self.batch_cmultie2t, :] = np.array(negative_ets)
        new_ets_values[:self.batch_cmultie2t] = train_lable[idxs, :]
        new_ets_values[self.batch_cmultie2t:2 * self.batch_cmultie2t] = np.array(
            [[-1.0] for i in range(self.batch_cmultie2t)])
        return new_ets_indexes, new_ets_values

    def get_batch1(self, train_i_indexes, epoch):
        r = min((epoch + 1) * self.batch_transe, len(config.d.train_triplet_data_idxs))
        last_idx = r - epoch * self.batch_transe
        new_facts_indexes = np.empty((last_idx + last_idx, 3)).astype(np.int32)
        new_facts_indexes[:r - epoch * self.batch_transe, :] = train_i_indexes[epoch * self.batch_transe: r, :]
        negative_triplets = [config.one_negative_sampling_triplet(tuple(new_facts_indexes[t])) for t in
                             range(r - epoch * self.batch_transe)]
        new_facts_indexes[last_idx:2 * last_idx, :] = np.array(negative_triplets)
        return new_facts_indexes, last_idx
        # idxs = np.random.randint(0, len(config.d.train_triplet_data_idxs), self.batch_transe)
        # new_facts_indexes = np.empty((self.batch_transe + self.batch_transe, 3)).astype(np.int32)
        # new_facts_indexes[:self.batch_transe, :] = train_i_indexes[idxs, :]
        # negative_triplets = [config.one_negative_sampling_triplet(tuple(new_facts_indexes[t])) for t in
        #                      range(self.batch_transe)]
        # new_facts_indexes[self.batch_transe:2 * self.batch_transe, :] = np.array(negative_triplets)
        # return new_facts_indexes

    def train_and_eval(self):
        self.train_data_cmultie2t = {key: [1] for key in config.d.train_type_idxs}
        self.train_i_indexes_cmultie2t = np.array(list(self.train_data_cmultie2t.keys()))
        self.train_i_lables_cmultie2t = np.array(list(self.train_data_cmultie2t.values()))

        self.train_i_indexes_transe = np.array(list(config.d.train_triplet_data_idxs))

        # if config.args.load == 'True':
        #     self.cmultie2t_model = torch.load(config.args.outdir + '1cmultie2t.pth')
        #     self.transe_mix_model = torch.load(config.args.outdir + '1transe.pth')
        #     self.cmultie2t_model.cuda()
        #     self.transe_mix_model.cuda()
        #     self.evaluate_rank_and_hits(config.d.test_et_data_idxs, cmultie2t_model=self.cmultie2t_model,
        #                                 transe_model=self.transe_mix_model)
        # else:
        #     # num_entity, num_entity_type, embedding_entity_size, embedding_type_size, num_filters,fcn_droupt
        #     self.cmultie2t_model = CMultiE2T(num_entity=self.num_entity, num_entity_type=self.num_entity_type,
        #                                      embedding_entity_size=self.embedding_entity_size,
        #                                      embedding_type_size=self.embedding_type_size,
        #                                      num_filters=self.cmultie2t_numfilters,
        #                                      fcn_droupt=self.cmultie2t_droupt)
        #     # num_entity, num_relation, num_entity_type, embedding_entity_size, embedding_relation_size,margin, param, norm='L2'
        #     self.transe_mix_model = TransE_MIX(num_entity=self.num_entity, num_relation=self.num_relation,
        #                                        num_entity_type=self.num_entity_type,
        #                                        embedding_entity_size=self.embedding_entity_size,
        #                                        embedding_relation_size=self.embedding_relation_size,
        #                                        margin=self.transe_margin, norm=self.norm)

        self.cmultie2t_model = CMultiE2T(num_entity=self.num_entity, num_entity_type=self.num_entity_type,
                                         embedding_entity_size=self.embedding_entity_size,
                                         embedding_type_size=self.embedding_type_size,
                                         num_filters=self.cmultie2t_numfilters,
                                         fcn_droupt=self.cmultie2t_droupt)
        self.cmultie2t_model.init()
        self.cmultie2t_model.cuda()

        self.transe_mix_model = TransE_MIX(num_entity=self.num_entity, num_relation=self.num_relation,
                                           num_entity_type=self.num_entity_type,
                                           embedding_entity_size=self.embedding_entity_size,
                                           embedding_relation_size=self.embedding_relation_size,
                                           margin=self.transe_margin, norm=self.norm)
        self.transe_mix_model.init()
        self.transe_mix_model.cuda()

        print('-------------------------------start training cmultie2t-------------------------------')
        self.train_cmultie2t(model=self.cmultie2t_model, epochs=self.cmultie2t_epoch, lr=self.cmultie2t_lr,
                             train_data=self.train_i_indexes_cmultie2t, train_lables=self.train_i_lables_cmultie2t)

        print('-------------------------------start training transe-mix-------------------------------')
        self.transe_mix_model.create_total_embedding_vec(e=self.cmultie2t_model.emb_entities.weight,
                                                         trans_matrix=self.cmultie2t_model.trans_matrix,
                                                         t=self.cmultie2t_model.emb_types.weight)
        self.train_transe_mix(model=self.transe_mix_model, lr=self.transe_lr, epochs=self.transe_epoch,
                              train_data=self.train_i_indexes_transe)
        print('-------------------------------start eval-------------------------------')
        self.evaluate_rank_and_hits(config.d.test_et_data_idxs, cmultie2t_model=self.cmultie2t_model,
                                    transe_model=self.transe_mix_model)
        index = 0
        while True:
            print('-------------------------------start training cmultie2t second-------------------------------')
            self.train_cmultie2t_second(model=self.cmultie2t_model, epochs=15, lr=0.0000001,
                                        train_data=self.train_i_indexes_cmultie2t,
                                        train_lables=self.train_i_lables_cmultie2t)
            print('-------------------------------start training transe-mix second-------------------------------')
            self.train_transe_mix_second(model=self.transe_mix_model, lr=0.0000001, epochs=15,
                                         train_data=self.train_i_indexes_transe)
            # evalation
            index += 1
            print('-------------------------------start eval-------------------------------')
            self.evaluate_rank_and_hits(config.d.test_et_data_idxs, cmultie2t_model=self.cmultie2t_model,
                                        transe_model=self.transe_mix_model)

    def train_cmultie2t(self, model, epochs, lr, train_data, train_lables):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        batchs_num = int(len(config.d.train_type_idxs) / self.batch_cmultie2t) + 1
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, y_batch = self.get_batch(train_data, train_lables, batch)
                pred, l2_reg = model.forward(x_batch)
                pred = pred * torch.FloatTensor(y_batch).cuda() * (-1)
                loss = model.loss(pred).mean() + self.cmultie2t_lmbda * l2_reg  # Softplus
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            if epoch == 1550:
                logger.info('-------cmultie2t lr has changed-------{}------>{}'.format(self.cmultie2t_lr, 0.000001))
                opt = torch.optim.Adam(model.parameters(), lr=float(0.000001))
            if epoch == 1850:
                logger.info('-------cmultie2t lr has changed-------{}------>{}'.format(self.cmultie2t_lr, 0.000001))
                opt = torch.optim.Adam(model.parameters(), lr=float(0.0000001))
            logger.info('-------cmultie2t-------epoch: {} train_loss: {}'.format(epoch, train_loss))
        logger.info('-------cmultie2t training is cmopleted-------')

    def train_cmultie2t_second(self, model, epochs, lr, train_data, train_lables):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        batchs_num = int(len(config.d.train_type_idxs) / self.batch_cmultie2t) + 1
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, y_batch = self.get_batch(train_data, train_lables, batch)
                pred, l2_reg = model.forward(x_batch)
                pred = pred * torch.FloatTensor(y_batch).cuda() * (-1)
                loss = model.loss(pred).mean() + self.cmultie2t_lmbda * l2_reg  # Softplus
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            logger.info('-------cmultie2t_second-------epoch: {} train_loss: {}'.format(epoch, train_loss))
        logger.info('-------cmultie2t_second training is cmopleted-------')

    def train_transe_mix(self, model, lr, epochs, train_data):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())

        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        batchs_num = int(len(config.d.train_triplet_data_idxs) / self.batch_transe) + 1
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, train_num = self.get_batch1(train_data, batch)
                p_scores, n_scores = model.forward(x_batch, train_num)
                y = torch.Tensor([-1]).cuda()
                loss = model.loss(p_scores, n_scores, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            if epoch == 500:
                logger.info('-------transe lr has changed-------{}------>{}'.format(self.cmultie2t_lr, 0.00002))
                opt = torch.optim.Adam(model.parameters(), lr=float(0.00002))
            if epoch == 1000:
                logger.info('-------transe lr has changed-------{}------>{}'.format(0.00002, 0.000005))
                opt = torch.optim.Adam(model.parameters(), lr=float(0.000005))
            if epoch == 1500:
                logger.info('-------transe lr has changed-------{}------>{}'.format(0.000005, 0.000001))
                opt = torch.optim.Adam(model.parameters(), lr=float(0.000001))
            logger.info('-------transe-------epoch: {} train_loss: {}'.format(epoch, train_loss))
        logger.info('-------transe training is cmopleted-------')

    def train_transe_mix_test_lr(self, model, lr, epochs, train_data):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        lr_list = [0.000001, 0.000002, 0.000003, 0.000004, 0.000005, 0.00001, 0.00002, 0.00003, 0.00005, 0.0001, 0.0002,
                   0.0003, 0.0005, 0.001, 0.002, 0.003,
                   0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        with open('loss.txt', 'w') as ffile:
            batchs_num = int(len(config.d.train_triplet_data_idxs) / self.batch_transe) + 1
            for epoch in range(epochs):
                model.train()
                train_loss = 0
                for batch in range(batchs_num):
                    x_batch, train_num = self.get_batch1(train_data, batch)
                    p_scores, n_scores = model.forward(x_batch, train_num)
                    y = torch.Tensor([-1]).cuda()
                    loss = model.loss(p_scores, n_scores, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    train_loss += loss.item()
                    s_f = str(loss.item()) + ('\t') + str(lr_list[batch]) + str('\n')
                    ffile.write(s_f)
                    logger.info('-------transe-------epoch: {} train_loss: {}'.format(epoch, train_loss))
                if epoch == 600:
                    opt = torch.optim.Adam(model.parameters(), lr=float(0.00001))
                if epoch == 1200:
                    opt = torch.optim.Adam(model.parameters(), lr=float(0.000002))
                if epoch == 1800:
                    opt = torch.optim.Adam(model.parameters(), lr=float(0.0000005))
                opt = torch.optim.Adam(model.parameters(), lr=float(lr_list[batch + 1]))
            logger.info('-------transe training is cmopleted-------')

    def train_transe_mix_second(self, model, lr, epochs, train_data):
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        batchs_num = int(len(config.d.train_triplet_data_idxs) / self.batch_transe) + 1
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, train_num = self.get_batch1(train_data, batch)
                p_scores, n_scores = model.forward(x_batch, train_num)
                y = torch.Tensor([-1]).cuda()
                loss = model.loss(p_scores, n_scores, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            logger.info('-------transe_second-------epoch: {} train_loss: {}'.format(epoch, train_loss))
        logger.info('-------transe_second training is cmopleted-------')

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

    def evaluate_rank_and_hits(self, data, cmultie2t_model, transe_model):
        with torch.no_grad():
            over_num_type = 0
            hit1_type = 0
            hit3_type = 0
            hit10_type = 0
            mrr_type = 0.0

            hit1_type_second = 0
            hit3_type_second = 0
            hit10_type_second = 0
            mrr_type_second = 0.0

            cmultie2t_hit1_type = 0
            cmultie2t_hit3_type = 0
            cmultie2t_hit10_type = 0
            cmultie2t_mrr_type = 0.0

            transe_hit1_type = 0
            transe_hit3_type = 0
            transe_hit10_type = 0
            transe_mrr_type = 0.0

            index = 0
            test_data = list(data)
            for fact in test_data:
                e, r, t = fact
                fact_et = (e, t)
                index += 1
                over_num_type += 1

                if index % 50 == 0:
                    print('The number of remaining test data is:', len(test_data) - index)

                correct_index = fact_et[1]
                score1 = self.evaluate_rank_score(cmultie2t_model, fact_et)
                score_cmultie2t = -score1

                transe_model.eval()
                score_tri = transe_model.predict(fact)
                total_score = (self.pred_lmbda * score1 + (1 - self.pred_lmbda) * score_tri).view(-1, 1)

                sorted_pred_transe = torch.argsort(score_tri, dim=0, descending=False)
                position_of_correct_fact_in_sorted_pred_transe = 0
                for tmpxx_transe in sorted_pred_transe:
                    tmpxx_transe += total_num
                    if tmpxx_transe == correct_index:
                        break
                    tmp_list = deepcopy(fact)
                    tmp_list = list(tmp_list)
                    tmp_list[2] = tmpxx_transe.item()
                    tmp_ert = tuple(tmp_list)

                    tmp_et = (e, tmpxx_transe.item())

                    if tmp_et in config.d.over_data or tmp_ert in config.d.total_tri_data_idxs:
                        continue
                    else:
                        position_of_correct_fact_in_sorted_pred_transe += 1
                ###################################################################################################################################
                sorted_pred_cmultie2t = torch.argsort(score_cmultie2t, dim=0, descending=True)
                position_of_correct_fact_in_sorted_pred_cmultie2t = 0
                for tmpxx_cmultie2t in sorted_pred_cmultie2t:
                    tmpxx_cmultie2t += total_num
                    if tmpxx_cmultie2t == correct_index:
                        break
                    tmp_list_cmultie2t = deepcopy(fact_et)
                    tmp_list_cmultie2t = list(tmp_list_cmultie2t)
                    tmp_list_cmultie2t[1] = tmpxx_cmultie2t.item()
                    tmp_et_cmultie2t = tuple(tmp_list_cmultie2t)
                    if tmp_et_cmultie2t in config.d.over_data:
                        continue
                    else:
                        position_of_correct_fact_in_sorted_pred_cmultie2t += 1
                ###################################################################################################################################
                sorted_pred = torch.argsort(total_score, dim=0, descending=False)
                position_of_correct_fact_in_sorted_pred = 0
                for tmpxx in sorted_pred:
                    tmpxx += total_num
                    if tmpxx == correct_index:
                        break
                    tmp_list = deepcopy(fact)
                    tmp_list = list(tmp_list)
                    tmp_list[2] = tmpxx.item()
                    tmp_ert = tuple(tmp_list)
                    tmp_et = (e, tmpxx.item())
                    if tmp_et in config.d.over_data or tmp_ert in config.d.total_tri_data_idxs:
                        continue
                    else:
                        position_of_correct_fact_in_sorted_pred += 1
                ###################################################################################################################################
                position_of_correct_fact_in_sorted_pred_second = position_of_correct_fact_in_sorted_pred
                if position_of_correct_fact_in_sorted_pred_cmultie2t <= position_of_correct_fact_in_sorted_pred_transe:
                    position_of_correct_fact_in_sorted_pred = position_of_correct_fact_in_sorted_pred_cmultie2t
                if position_of_correct_fact_in_sorted_pred_transe == 0:
                    transe_hit1_type += 1
                    transe_hit3_type += 1
                    transe_hit10_type += 1
                elif position_of_correct_fact_in_sorted_pred_transe <= 2:
                    transe_hit3_type += 1
                    transe_hit10_type += 1
                elif position_of_correct_fact_in_sorted_pred_transe <= 9:
                    transe_hit10_type += 1
                transe_mrr_type += float(1 / (position_of_correct_fact_in_sorted_pred_transe + 1))

                if position_of_correct_fact_in_sorted_pred_cmultie2t == 0:
                    cmultie2t_hit1_type += 1
                    cmultie2t_hit3_type += 1
                    cmultie2t_hit10_type += 1
                elif position_of_correct_fact_in_sorted_pred_cmultie2t <= 2:
                    cmultie2t_hit3_type += 1
                    cmultie2t_hit10_type += 1
                elif position_of_correct_fact_in_sorted_pred_cmultie2t <= 9:
                    cmultie2t_hit10_type += 1
                cmultie2t_mrr_type += float(1 / (position_of_correct_fact_in_sorted_pred_cmultie2t + 1))

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

                if position_of_correct_fact_in_sorted_pred_second == 0:
                    hit1_type_second += 1
                    hit3_type_second += 1
                    hit10_type_second += 1
                elif position_of_correct_fact_in_sorted_pred_second <= 2:
                    hit3_type_second += 1
                    hit10_type_second += 1
                elif position_of_correct_fact_in_sorted_pred_second <= 9:
                    hit10_type_second += 1
                mrr_type_second += float(1 / (position_of_correct_fact_in_sorted_pred_second + 1))

                # print('----transe----index:', index, 'position: ', position_of_correct_fact_in_sorted_pred_transe)
                # print('----cmultie2t-----index:', index, 'position: ',
                #       position_of_correct_fact_in_sorted_pred_cmultie2t)
                # print('----total----index:', index, 'position: ', position_of_correct_fact_in_sorted_pred)
                # print('----total----second-----index:', index, 'position: ', position_of_correct_fact_in_sorted_pred_second)

            logger.info(
                '------transe------mrr: {}'.format(float(transe_mrr_type / over_num_type)))
            logger.info(
                '------transe------hit1: {}'.format(float(transe_hit1_type / over_num_type)))
            logger.info(
                '------transe------hit3: {}'.format(float(transe_hit3_type / over_num_type)))
            logger.info(
                '------transe------hit10: {}'.format(float(transe_hit10_type / over_num_type)))

            logger.info(
                '------cmultie2t------mrr: {}'.format(float(cmultie2t_mrr_type / over_num_type)))
            logger.info(
                '------cmultie2t------hit1: {}'.format(float(cmultie2t_hit1_type / over_num_type)))
            logger.info(
                '------cmultie2t------hit3: {}'.format(float(cmultie2t_hit3_type / over_num_type)))
            logger.info(
                '------cmultie2t------hit10: {}'.format(float(cmultie2t_hit10_type / over_num_type)))

            logger.info(
                '------total------mrr: {}'.format(float(mrr_type / over_num_type)))
            logger.info(
                '------total------hit1: {}'.format(float(hit1_type / over_num_type)))
            logger.info(
                '------total------hit3: {}'.format(float(hit3_type / over_num_type)))
            logger.info(
                '------total------hit10: {}'.format(float(hit10_type / over_num_type)))

            logger.info(
                '------total------second-------mrr: {}'.format(float(mrr_type_second / over_num_type)))
            logger.info(
                '------total------second-------hit1: {}'.format(float(hit1_type_second / over_num_type)))
            logger.info(
                '------total------second-------hit3: {}'.format(float(hit3_type_second / over_num_type)))
            logger.info(
                '------total------second-------hit10: {}'.format(float(hit10_type_second / over_num_type)))
            return over_num_type, hit1_type, hit3_type, hit10_type, mrr_type, cmultie2t_mrr_type, cmultie2t_hit1_type, cmultie2t_hit3_type, cmultie2t_hit10_type


if __name__ == '__main__':
    args = config.args
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    experiment = Experiment(num_entity=len(config.d.entity_idxs) - len(config.d.entity_types_idxs),
                            num_relation=len(config.d.relation_idxs),
                            num_entity_type=len(config.d.entity_types_idxs), embedding_entity_size=args.embsize_entity,
                            embedding_relation_size=args.embsize_relation, embedding_type_size=args.embsize_entity_type,
                            batch_transe=args.transe_batchsize, batch_cmultie2t=args.cmultie2t_batchsize,
                            transe_epoch=args.transe_epochs, cmultie2t_epoch=args.cmultie2t_epochs,
                            transe_lr=args.transe_learningrate, cmultie2t_lr=args.cmultie2t_learningrate,
                            transe_margin=args.transe_margin, cmultie2t_numfilters=args.cmultie2t_num_filters,
                            cmultie2t_lmbda=args.cmultie2t_lmbda, cmultie2t_droupt=args.cmultie2t_droupt,
                            norm=args.norm, is_load=args.load, lmbda=args.pred_lmbda)
    experiment.train_and_eval()
