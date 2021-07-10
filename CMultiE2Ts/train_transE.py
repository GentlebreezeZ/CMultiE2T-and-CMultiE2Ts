import numpy as np
from model import TransE_MIX
import torch

import config

total_num = len(config.d.entity_idxs) - len(config.d.entity_types_idxs)


class Experiment_TransE:
    def __init__(self, param, batch_size=128, learning_rate=0.001, embedding_dim=200, embedding_dim_relation=200,
                 epochs=50000, logger=None, norm='L2'):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.embedding_dim = embedding_dim
        self.embedding_dim_relation = embedding_dim_relation
        self.epochs = epochs
        self.num_entity = len(config.d.entity_idxs)
        self.num_relation = len(config.d.relation_idxs)
        self.share_parm = param
        self.logger = logger
        self.norm = norm

    def set_grad(self):
        self.share_parm.emb_types.requires_grad_(requires_grad=False)
        self.share_parm.emb_entities.requires_grad_(requires_grad=False)
        self.share_parm.emb_entitiy_transfer.requires_grad = False
        self.share_parm.emb_type_transfer.requires_grad = False
        self.share_parm.trans_matrix.requires_grad = False

        self.share_parm.emb_total_emtities.requires_grad_(requires_grad=False)
        self.share_parm.emb_relations.requires_grad_(requires_grad=True)

    def get_batch1(self, train_i_indexes, epoch):
        r = min((epoch + 1) * self.batch_size, len(config.d.train_triplet_data_idxs))
        last_idx = r - epoch * self.batch_size
        new_facts_indexes = np.empty((last_idx + last_idx, 3)).astype(np.int32)
        new_facts_indexes[:r - epoch * self.batch_size, :] = train_i_indexes[epoch * self.batch_size: r, :]
        negative_triplets = [config.one_negative_sampling_triplet(tuple(new_facts_indexes[t])) for t in
                             range(r - epoch * self.batch_size)]
        new_facts_indexes[last_idx:2 * last_idx, :] = np.array(negative_triplets)
        return new_facts_indexes, last_idx

    def train(self):
        self.train_i_indexes = np.array(list(config.d.train_triplet_data_idxs))

        if config.args.load == 'True':
            model = torch.load(config.args.outdir + 'transe_model.pth')
        else:
            # def __init__(self, embedding_size_entity, embedding_size_relation, margin, param, norm='L2')
            model = TransE_MIX(embedding_size_entity=self.embedding_dim,
                               embedding_size_relation=self.embedding_dim_relation, margin=config.args.transe_margin,
                               param=self.share_parm)
        model.cuda()

        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())

        opt = torch.optim.Adam(model.parameters(), lr=float(self.learning_rate))

        batchs_num = int(len(config.d.train_triplet_data_idxs) / self.batch_size) + 1
        for epoch in range(self.epochs):
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, train_num = self.get_batch1(self.train_i_indexes, batch)
                p_scores, n_scores = model.forward(x_batch, train_num)
                y = torch.Tensor([-1]).cuda()
                loss = model.loss(p_scores, n_scores, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            if epoch == 500:
                # torch.save(self.param, config.args.outdir + '2000param.pth')
                torch.save(model, config.args.outdir + '500transe_model.pth')
                # reset opt
                opt = torch.optim.Adam(model.parameters(), lr=float(0.0005))
            if epoch == 1000:
                # torch.save(self.param, config.args.outdir + '2000param.pth')
                torch.save(model, config.args.outdir + '1000transe_model.pth')
                # reset opt
                opt = torch.optim.Adam(model.parameters(), lr=float(0.00025))
            if epoch == 1500:
                # torch.save(self.param, config.args.outdir + '2000param.pth')
                torch.save(model, config.args.outdir + '1500transe_model.pth')
                # reset opt
                opt = torch.optim.Adam(model.parameters(), lr=float(0.00001))
            self.logger.info('-------transe-------epoch: {} train_loss: {}'.format(epoch, train_loss))
        self.logger.info('-------transe training is cmopleted-------')
        torch.save(model, config.args.outdir + 'transe_model.pth')
        return model



    def train_second(self,model,epochs,lr):
        model.cuda()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("param:", name, param.size())
        opt = torch.optim.Adam(model.parameters(), lr=float(lr))
        batchs_num = int(len(config.d.train_triplet_data_idxs) / self.batch_size) + 1
        for epoch in range(epochs):
            train_loss = 0
            for batch in range(batchs_num):
                x_batch, train_num = self.get_batch1(self.train_i_indexes, batch)
                p_scores, n_scores = model.forward(x_batch, train_num)
                y = torch.Tensor([-1]).cuda()
                loss = model.loss(p_scores, n_scores, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
            self.logger.info('-------transe-------epoch: {} train_loss: {}'.format(epoch, train_loss))
        self.logger.info('-------transe training is cmopleted-------')
        return model



# if __name__ == '__main__':
#     experiment_transe = Experiment_TransE(batch_size=config.args.batchsize, learning_rate=config.args.learningrate,
#                             embedding_dim=config.args.embsize, epochs=config.args.epochs)
#     seed = 1234
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     experiment_transe.train()
