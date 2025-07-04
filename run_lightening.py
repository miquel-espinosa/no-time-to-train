import copy
import json
import os
import pickle
import shutil
import tempfile
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.distributed as dist
import mmengine

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dev_hongyi.pl_wrapper.sam2ref_pl import RefSam2LightningModel
from dev_hongyi.pl_wrapper.sam2matcher_pl import Sam2MatcherLightningModel


def collect_results_cpu(result_part, size=None, tmpdir=None):
    
    # Check if distributed training is initialized
    if not dist.is_initialized():
        return result_part
    
    # Reference: MMDetection
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmengine.mkdir_or_exist('/tmp/.mydist_test')
            tmpdir = tempfile.mkdtemp(dir='/tmp/.mydist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmengine.dump(result_part, os.path.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = os.path.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmengine.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        if size is not None:
            ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


class SAM2RefLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--out_path", default=None, type=str)
        parser.add_argument("--out_support_res", default=None, required=False, type=str)
        parser.add_argument("--out_neg_pkl", default=None, required=False, type=str)
        parser.add_argument("--out_neg_json", default=None, required=False, type=str)
        parser.add_argument("--export_result", default=None, type=str)


    def before_test(self):
        memory_bank_cfg = self.model.model_cfg["memory_bank_cfg"]

        if self.model.test_mode == "fill_memory":
            self.model.dataset_cfgs["fill_memory"]["memory_length"] = memory_bank_cfg["length"]
        elif self.model.test_mode == "fill_memory_neg":
            self.model.dataset_cfgs["fill_memory"]["memory_length"] = memory_bank_cfg["length_negative"]
            self.model.dataset_cfgs["fill_memory"]["root"] = self.model.dataset_cfgs["support"]["root"]
            self.model.dataset_cfgs["fill_memory"]["json_file"] = self.config.test.out_neg_json
            self.model.dataset_cfgs["fill_memory"]["memory_pkl"] = self.config.test.out_neg_pkl
        else:
            pass


    def after_test(self):
        if (
                self.model.test_mode == "fill_memory"
                or self.model.test_mode == "postprocess_memory"
                or self.model.test_mode == "fill_memory_neg"
                or self.model.test_mode == "postprocess_memory_neg"
        ):
            if self.config.test.out_path is not None:
                save_path = self.config.test.out_path
            else:
                raise RuntimeError(
                    "A saving path for the temporary checkpoint is required to store the model with memory bank."
                )
            self.trainer.save_checkpoint(save_path)

            if self.model.test_mode == "fill_memory":
                print("Checkpoint with memory is saved to %s" % save_path)
            elif self.model.test_mode == "postprocess_memory":
                print("Checkpoint with post-processed memory is saved to %s" % save_path)
            elif self.model.test_mode == "fill_memory_neg":
                print("Checkpoint with negative memory is saved to %s" % save_path)
            elif self.model.test_mode == "postprocess_memory_neg":
                print("Checkpoint with post-processed negative memory is saved to %s" % save_path)
            else:
                raise NotImplementedError
        elif self.model.test_mode == "test" or self.model.test_mode == "test_support":
            results = copy.deepcopy(self.trainer.model.output_queue)
            results_all = collect_results_cpu(
                results, size=len(self.trainer.model.eval_dataset)
            )

            if len(self.trainer.model.scalars_queue) > 0:
                scalars = copy.deepcopy(self.trainer.model.scalars_queue)
                scalars_all = collect_results_cpu(
                    scalars, size=len(self.trainer.model.eval_dataset)
                )
            else:
                scalars_all = None

            if not dist.is_initialized() or dist.get_rank() == 0:
                if scalars_all is not None:
                    with open("./scalars_all.pkl", "wb") as f:
                        pickle.dump(scalars_all, f)

                results_unpacked = []
                for results_per_img in results_all:
                    results_unpacked.extend(results_per_img)
                if self.config.test.export_result is not None:
                    with open(self.config.test.export_result, 'w') as f:
                        json.dump(results_unpacked, f)
                if self.model.test_mode == "test":
                    self.trainer.model.eval_dataset.evaluate(results_unpacked)
                elif self.model.test_mode == "test_support":
                    self.trainer.model.eval_dataset.evaluate(results_unpacked)
                    with open(self.config.test.out_support_res, "wb") as f:
                        pickle.dump(results_unpacked, f)

                    # out_pkl = self.config.test.out_neg_pkl
                    # out_json = self.config.test.out_neg_json
                    # n_sample = self.trainer.model.seg_model.mem_length_negative
                    # self.trainer.model.eval_dataset.sample_negative(
                    #     results_unpacked, out_pkl, out_json, n_sample
                    # )
                else:
                    raise NotImplementedError
        elif self.model.test_mode == "vis_memory":
            pass
        else:
            raise NotImplementedError(f"Unrecognized test mode {self.model.test_mode}")



if __name__ == "__main__":
    SAM2RefLightningCLI()

