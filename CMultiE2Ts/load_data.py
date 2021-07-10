import random
from collections import defaultdict

train_epochs = 1024


class Data:

    def __init__(self, data_dir="data/FB15k/"):
        self.train_type_data = self.load_type_data(data_dir, "Entity_Type_train")
        self.valid_type_data = self.load_type_data(data_dir, "Entity_Type_valid")
        self.test_type_data = self.load_type_data(data_dir, "Entity_Type_test")

        self.train_tri_type_data = self.create_type_triplet(self.train_type_data)
        self.valid_tri_type_data = self.create_type_triplet(self.valid_type_data)
        self.test_tri_type_data = self.create_type_triplet(self.test_type_data)

        self.type_data = self.train_type_data + self.valid_type_data + self.test_type_data

        self.types = self.get_types(self.type_data)
        self.entities = self.get_type_entities(self.type_data)

        self.entity_idxs, self.relation_idxs, self.entity_types_idxs = self.encode_entity_to_id(data_dir)

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        total = len(self.entity_idxs)
        self.entity_types_idxs = {self.types[i]: i for i in range(len(self.types))}

        self.entity_idxs = self.create_total_entities()
        self.entity_types_idxs = {self.types[i]: i + total for i in range(len(self.types))}

        self.types_entity_idxs = {v: k for k, v in self.entity_types_idxs.items()}
        self.idxs_entity = {v: k for k, v in self.entity_idxs.items()}

        self.relation_idxs = self.encode_relation_to_id(data_dir)
        self.idxs_relation = {v: k for k, v in self.relation_idxs.items()}

        self.train_type_idxs = self.get_type_data_idxs(self.train_type_data)
        random.shuffle(self.train_type_idxs)
        self.valid_type_idxs = self.get_type_data_idxs(self.valid_type_data)
        self.test_type_idxs = self.get_type_data_idxs(self.test_type_data)

        self.over_data = self.train_type_idxs + self.valid_type_idxs + self.test_type_idxs

        self.entity_to_type_dict = self.get_type_data_idxs_dict(self.train_type_idxs)
        self.type_to_entity_dict = self.get_type_to_entity(self.train_type_idxs)

        self.entity_negative_type_dict = self.get_entity_neg_candidate_set()
        ##self.type_negative_entity_dict = self.get_type_neg_candidata_set()

        #################################################knowledge triplet###########################################################
        self.train_triplet_data = self.load_triplet_data(data_dir, "train", reverse=False)
        self.valid_triplet_data = self.load_triplet_data(data_dir, "valid", reverse=False)
        self.test_triplet_data = self.load_triplet_data(data_dir, "test", reverse=False)

        self.total_train_triplet_data = self.train_triplet_data + self.train_tri_type_data
        self.total_valid_triplet_data = self.valid_triplet_data + self.valid_tri_type_data
        self.total_test_triplet_data = self.test_triplet_data + self.test_tri_type_data

        self.triplet_data = self.train_triplet_data + self.valid_triplet_data + self.test_triplet_data

        self.train_triplet_data_idxs = self.get_data_idxs(self.total_train_triplet_data)
        random.shuffle(self.train_triplet_data_idxs)
        self.valid_triplet_data_idxs = self.get_data_idxs(self.total_valid_triplet_data)
        self.test_triplet_data_idxs = self.get_data_idxs(self.total_test_triplet_data)

        self.test_et_data_idxs = self.get_data_idxs(self.test_tri_type_data)

        self.total_tri_data_idxs = self.train_triplet_data_idxs + self.valid_triplet_data_idxs + self.test_triplet_data_idxs

        self.head_and_tail_entity_negative_samples = self.get_head_and_tail_entity_negative_candidate()

    def load_type_data(self, data_dir, data_type):
        data = []
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding='utf-8') as f:
            for line in f.readlines():
                e, et = line.strip().split("\t")
                data.append([e, et])
        return data

    def create_total_entities(self):
        total_num_entity = len(self.entity_idxs)
        total_entity_dict = {}

        for k, v in self.entity_idxs.items():
            total_entity_dict[k] = v
        for k, v in self.entity_types_idxs.items():
            total_entity_dict[k] = v + total_num_entity
        return total_entity_dict

    def encode_relation_to_id(self, data_dir):
        relation2id = {}
        with open(data_dir + 'relation2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                rel, rel2id = line.strip().split("\t")
                relation2id[rel] = int(rel2id)
        return relation2id

    def encode_entity_to_id(self, data_dir):
        entity2id = {}
        relation2id = {}
        type2id = {}
        with open(data_dir + 'entity2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ent, ent2id = line.strip().split("\t")
                entity2id[ent] = int(ent2id)
        total_entity_num = len(entity2id)
        with open(data_dir + 'relation2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                rel, rel2id = line.strip().split("\t")
                relation2id[rel] = int(rel2id)
        with open(data_dir + 'type2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                t, t2id = line.strip().split("\t")
                type2id[t] = int(t2id) + total_entity_num
        with open(data_dir + 'type2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                t, t2id = line.strip().split("\t")
                entity2id[t] = int(t2id) + total_entity_num
        return entity2id, relation2id, type2id

    def get_types(self, data):
        types = sorted(list(set([d[1] for d in data])))
        return types

    def get_type_entities(self, data):
        entities = sorted(list(set([d[0] for d in data])))
        return entities

    def get_triplet_entities(self, data):
        entities = sorted(list(set([d[0] for d in data] + [d[2] for d in data])))
        return entities

    def get_type_data_idxs(self, data):
        entity_type_data_idxs = [(self.entity_idxs[data[i][0]], self.entity_types_idxs[data[i][1]]) for i in
                                 range(len(data))]
        return entity_type_data_idxs

    def get_type_data_idxs_dict(self, data):
        entity_type = {}
        for temp in data:
            entity_type.setdefault(temp[0], set()).add(temp[1])
        return entity_type

    def get_type_to_entity(self, data):
        type2entity = {}
        for temp in data:
            type2entity.setdefault(temp[1], set()).add(temp[0])
        return type2entity

    def get_total_entity_type(self):
        tt = defaultdict(list)
        for e, t in self.over_data:
            tt[e].append(t)
        tt = dict(tt)
        return tt

    def get_entity_neg_candidate_set(self):
        type_num = len(self.entity_types_idxs)
        total_num = len(self.entity_idxs) - len(self.entity_types_idxs)
        type_set = set([(i + total_num) for i in range(type_num)])

        entity_negative_type_dict = {}
        print('load entity type instance negative samples!')
        for k, v in self.entity_to_type_dict.items():
            tmp = list(type_set - v)
            random.shuffle(tmp)
            entity_negative_type_dict[k] = tmp

        return entity_negative_type_dict

    def get_type_neg_candidata_set(self):
        entity_num = len(self.entity_idxs)
        entity_set = set([i for i in range(entity_num)])

        type_negative_entity_dict = {}
        print('load type negative samples!')
        for k, v in self.type_to_entity_dict.items():
            tmp = list(entity_set - v)
            random.shuffle(tmp)
            type_negative_entity_dict[k] = tmp[0:400]

        return type_negative_entity_dict

    #########################################################################################################################
    def load_triplet_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding='utf-8') as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1] + "_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    def get_head_and_tail_entity_negative_candidate(self):
        print('load entity negative samples!')
        head_and_tail_entity = dict()
        for train_data in self.train_triplet_data_idxs:
            h, r, t = train_data
            head_and_tail_entity.setdefault(h, set()).add(t)
            head_and_tail_entity.setdefault(t, set()).add(h)
        entity_total_set = set([i for i in range(len(self.entity_idxs))])
        res = dict()
        for k, v in head_and_tail_entity.items():
            tmp = list(entity_total_set - v)
            random.shuffle(tmp)
            res[k] = tmp[0:train_epochs]
        return res

    def create_type_triplet(self, et_list):
        res = []
        for tmp in et_list:
            e, t = tuple(tmp)
            tri = [e, '/rdf/type', t]
            res.append(tri)
        return res
