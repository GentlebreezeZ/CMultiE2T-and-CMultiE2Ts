import argparse
import pickle

from load_data import Data
import random

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=5000, help='Number of epochs (default: 200)')
parser.add_argument('--batchsize', type=int, default=128, help='Batch size (default: 128)')
parser.add_argument('--num_filters', type=int, default=400, help='number of filters CNN')
parser.add_argument('--embsize_entity', default=200, help='Entity Embedding size (default: 200)')
parser.add_argument('--embsize_entity_type', default=100, help='Entity Type Embedding size (default: 100)')
parser.add_argument('--learningrate', default=0.00005, help='Learning rate (default: 0.00005)')
parser.add_argument("--lmbda", default=0.095, type=float, help="L2 regularization item 0.0001")
parser.add_argument("--droupt", default=0.4, type=float, help="droupt regularization item 0.2")
parser.add_argument('--dataset', type=str, default="YAGO", nargs="?",
                    help='Which dataset to use: FB15k, YAGO')
parser.add_argument('--indir', type=str, default='data/YAGO/', help='Input dir of train, test and valid data')
parser.add_argument('--outdir', type=str, default='output/YAGO/', help='Output dir of model')
parser.add_argument('--negative_sample', type=str, default='True', help='load negative triplets')
parser.add_argument('--load_data', type=str, default='True', help='load data from txt')
parser.add_argument('--load', default='false',help='If true, it loads a saved model in outdir and train or evaluate it (default: False)')
args = parser.parse_args()

d = Data(data_dir=args.indir)

def one_negative_sampling(entity_types, train_set=d.train_idxs, entity_total=len(d.entity_idxs),
                          type_total=len(d.entity_types_idxs)):
    e, t = entity_types
    total_num = len(d.entity_negative_type_dict[e])
    key = random.randint(0, total_num-1)
    neg_t = d.entity_negative_type_dict[e][key]
    negative_entity_type = (e, neg_t)
    return negative_entity_type

def one_negative_sampling1(entity_types, train_set=d.train_idxs, entity_total=len(d.entity_idxs),
                          type_total=len(d.entity_types_idxs)):
    e, t = entity_types
    is_entity = random.randint(0, 1)
    if is_entity:
        total_num = len(d.type_negative_entity_dict[t])
        key = random.randint(0, total_num-1)
        neg_e = d.type_negative_entity_dict[t][key]
        negative_entity_type = (neg_e,t)
        # if negative_entity_type in d.train_idxs:
        #     print('cccccccccccc')
    else:
        total_num = len(d.entity_negative_type_dict[e])
        key = random.randint(0, total_num-1)
        neg_t = d.entity_negative_type_dict[e][key]
        negative_entity_type = (e, neg_t)
        # if negative_entity_type in d.train_idxs:
        #     print('cccccccccccc')
    return negative_entity_type






#
# if args.load_data == 'True':
#     d = Data(data_dir=args.indir)
#     data_info = {}
#     data_info['train_data'] = d.train_idxs
#     data_info['valid_data'] = d.valid_idxs
#     data_info['test_data'] = d.test_idxs
#
#     data_info['entity2id'] = d.entity_idxs
#     data_info['id2entity'] = d.idxs_entity
#
#     data_info['type2id'] = d.entity_types_idxs
#     data_info['id2type'] = d.types_entity_idxs


# def get_negative_et(golden_et):
#     negative_triplets = {}
#     for t in golden_et:
#         negative_et_list = []
#         for i in range(args.epochs):
#             negative_triple = one_negative_sampling(t, d.train_idxs, len(d.entity_idxs), len(d.entity_types_idxs))
#             negative_et_list.append(negative_triple)
#         negative_triplets[t] = negative_et_list
#     print('create negative triplets finish!')
#     return negative_triplets


# negative_ets = get_negative_et(d.train_idxs)
# data_info['negative_ets'] = negative_ets
#
# with open(args.indir + "/dictionaries" + ".bin", 'wb') as f:
#     pickle.dump(data_info, f)
# print('prepare data finish!')

# with open(args.indir + "/dictionaries" + ".bin", 'rb') as fin:
#     data_info = pickle.load(fin)
# train_data = data_info['train_data']
# valid_data =data_info['valid_data']
# test_data =data_info['test_data']
#
# entity2id =data_info['entity2id']
# id2entity =data_info['id2entity']
#
# type2id =data_info['type2id']
# id2type =data_info['id2type']
#
# negative_ets = data_info['negative_ets']
