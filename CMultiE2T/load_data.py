import random


class Data:
    def __init__(self, data_dir="data/FB15k/"):
        ###############################################entity type#######################################################
        self.train_type_data = self.load_type_data(data_dir, "Entity_Type_train")
        self.valid_type_data = self.load_type_data(data_dir, "Entity_Type_valid")
        self.test_type_data = self.load_type_data(data_dir, "Entity_Type_test")

        self.type_data = self.train_type_data + self.valid_type_data + self.test_type_data

        self.types = self.get_types(self.type_data)
        self.entities = self.get_entities(self.type_data)

        self.entity_types_idxs = {self.types[i]: i for i in range(len(self.types))}
        self.types_entity_idxs = {v: k for k, v in self.entity_types_idxs.items()}

        self.entity_idxs = self.encode_entity_to_id()
        self.idxs_entity = {v: k for k, v in self.entity_idxs.items()}

        self.train_idxs = self.get_type_data_idxs(self.train_type_data)
        random.shuffle(self.train_idxs)
        self.valid_idxs = self.get_type_data_idxs(self.valid_type_data)
        self.test_idxs = self.get_type_data_idxs(self.test_type_data)

        self.over_data = self.train_idxs + self.valid_idxs + self.test_idxs

        self.test_valid_idxs = self.test_idxs + self.valid_idxs
        random.shuffle(self.test_valid_idxs)


        self.entity_to_type_dict = self.get_type_data_idxs_dict(self.train_idxs)
        self.type_to_entity_dict = self.get_type_to_entity(self.train_idxs)

        self.entity_negative_type_dict = self.get_entity_neg_candidate_set()
        #self.type_negative_entity_dict = self.get_type_neg_candidata_set()

        #self.total_negative_type_dict = self.get_total_type_neg_candidate()












    def encode_entity_to_id(self):
        entity2id = {}
        with open('data/FB15k/entity2id.txt', 'r', encoding='utf-8') as f:
            for line in f.readlines():
                ent, ent2id = line.strip().split("\t")
                entity2id[ent] = int(ent2id)
        return entity2id

    def load_type_data(self, data_dir, data_type):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding='utf-8') as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
        return data

    def get_types(self, data):
        types = sorted(list(set([d[1] for d in data])))
        return types

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data])))
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

    def get_entity_neg_candidate_set(self):
        type_num = len(self.entity_types_idxs)
        type_set = set([i for i in range(type_num)])
        entity_negative_type_dict = {}

        for k, v in self.entity_to_type_dict.items():
            tmp = list(type_set - v)
            random.shuffle(tmp)
            entity_negative_type_dict[k] = tmp

        return entity_negative_type_dict




    def get_type_neg_candidata_set(self):
        entity_num = len(self.entity_idxs)
        entity_set = set([i for i in range(entity_num)])

        type_negative_entity_dict = {}
        print('load negative samples!')
        for k, v in self.type_to_entity_dict.items():
            tmp = list(entity_set - v)
            random.shuffle(tmp)
            type_negative_entity_dict[k] = tmp

        return type_negative_entity_dict
