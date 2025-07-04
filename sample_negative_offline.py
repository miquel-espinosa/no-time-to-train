import os
import yaml
import pickle
import warnings

warnings.filterwarnings("ignore")

from dev_hongyi.pl_wrapper.sam2matcher_pl import get_dataset

import argparse
parser = argparse.ArgumentParser("Sample negatives")
parser.add_argument("--config", default=None, required=True, type=str)
parser.add_argument("--out_support_res", default=None, required=True, type=str)
parser.add_argument("--out_neg_pkl", default=None, required=True, type=str)
parser.add_argument("--out_neg_json", default=None, required=True, type=str)
args = parser.parse_args()


if __name__ == "__main__":
    with open(args.out_support_res, 'rb') as f:
        results = pickle.load(f)

    with open(args.config, 'r') as f:
        cfgs = yaml.safe_load(f)

    dataset_cfg = cfgs["model"]["init_args"]["dataset_cfgs"]["support"]
    support_set = get_dataset(dataset_cfg, "test_support")

    n_sample = cfgs["model"]["init_args"]["model_cfg"]["memory_bank_cfg"]["length_negative"]
    score_thr = cfgs["model"]["init_args"]["model_cfg"]["sam2_infer_cfgs"]["negative_score_thr"]

    out_pkl = args.out_neg_pkl
    out_json = args.out_neg_json
    support_set.sample_negative(results, out_pkl, out_json, n_sample, score_thr)
