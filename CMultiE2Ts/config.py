import argparse

from load_data import Data
import random

parser = argparse.ArgumentParser()
parser.add_argument('--transe_epochs', default=2050, help='Number of epochs (default: 0)')
parser.add_argument('--cmultie2t_epochs', default=2050, help='Number of epochs (default: 0)')
parser.add_argument('--transe_batchsize', type=int, default=256, help='Batch size (default: 0)')
parser.add_argument('--cmultie2t_batchsize', type=int, default=90, help='Batch size (default: 0)')
parser.add_argument('--cmultie2t_num_filters', type=int, default=500, help='number of filters CNN (default: 0)')
parser.add_argument('--embsize_entity', default=300, help='Entity Embedding size (default: 300)')
parser.add_argument('--embsize_relation', default=200, help='relation Embedding size (default: 200)')
parser.add_argument('--embsize_entity_type', default=200, help='Entity Type Embedding size (default: 200)')
parser.add_argument('--transe_learningrate', default=0.0001, help='Learning rate (default: 0.00005)')
parser.add_argument('--cmultie2t_learningrate', default=0.00005, help='Learning rate (default: 0.00005)')
parser.add_argument("--cmultie2t_lmbda", default=0.095, type=float, help="L2 regularization item")
parser.add_argument("--cmultie2t_droupt", default=0.4, type=float, help="cmultie2t fully connect droupt")
parser.add_argument('--transe_margin', type=float, help='margin', default=5.0)
parser.add_argument('--norm', type=str, help='margin', default='L2')
parser.add_argument('--dataset', type=str, default="FB15K", nargs="?",
                    help='Which dataset to use: FB15k, YAGO')
parser.add_argument('--indir', type=str, default='data/FB15K/', help='Input dir of train, test and valid data')
parser.add_argument('--outdir', type=str, default='output/FB15K/', help='Output dir of model')
parser.add_argument('--load', default='False',
                    help='If true, it loads a saved model in dir outdir and evaluate or train it (default: False)')
parser.add_argument("--pred_lmbda", default=0.5, type=float, help="L2 regularization item")


args = parser.parse_args()

d = Data(data_dir=args.indir)


def one_negative_sampling(entity_types):
    e, t = entity_types
    total_num = len(d.entity_negative_type_dict[e])
    key = random.randint(0, total_num - 1)
    neg_t = d.entity_negative_type_dict[e][key]
    negative_entity_type = (e, neg_t)
    return negative_entity_type

def one_negative_sampling_triplet(triplet):
    h, r, t = triplet
    # is_head = random.randint(0, 1)
    # if is_head:
    #     total_num = len(d.head_and_tail_entity_negative_samples[t])
    #     key = random.randint(0, total_num - 1)
    #     neg_h = d.head_and_tail_entity_negative_samples[t][key]
    #     negative_entity_type = (neg_h, r,t)
    # else:
    #     total_num = len(d.head_and_tail_entity_negative_samples[h])
    #     key = random.randint(0, total_num - 1)
    #     neg_t = d.head_and_tail_entity_negative_samples[h][key]
    #     negative_entity_type = (h, r, neg_t)
    if r == 1345 and args.dataset == "FB15K":
        total_num = len(d.entity_negative_type_dict[h])
        key = random.randint(0, total_num - 1)
        neg_t = d.entity_negative_type_dict[h][key]
        negative_entity_type = (h, r, neg_t)
    else:
        total_num = len(d.head_and_tail_entity_negative_samples[h])
        key = random.randint(0, total_num - 1)
        neg_t = d.head_and_tail_entity_negative_samples[h][key]
        negative_entity_type = (h, r, neg_t)

    return negative_entity_type















