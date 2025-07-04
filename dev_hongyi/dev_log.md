## Results: 
- Commit: 26119e9b256bdced6c5fdf3fc576780ed2ec8386
  - 5 shot all classes mini-val: 19.5 box AP, 19.2 mask AP (matching_baseline_instance.yaml)
  - 10 shot all few-shot classes mini-val: 22.5 box AP, 22.5 mask AP (matching_baseline_10shot.yaml)

## COCO 

### Ablation STD Few-shot results varying reference sets
- 10 different seeds, 6 different n_shots
  - `zsh ./scripts/ablation_coco_std.sh`


## LVIS steps:
- Download LVIS dataset to path `data/lvis/original`
```bash
cd data/lvis/original
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip
wget https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip
```
- Download LVIS minival json file from `https://huggingface.co/clin1223/GenerateU/blob/main/lvis_v1_minival.json`. Save it as `data/lvis/original/lvis_v1_minival_just_bbox.json`

- Add missing `file_name` attribute to the lvis json file, and add is_crowd attribute to the json file
  - `python3 dev_hongyi/dataset/lvis_add_filename.py`

- This lvis-minival json doesn't have segmentation annotations, so we will copy the segmentation annotations from the lvis-val json file, using annotation id.
  - `python3 dev_hongyi/dataset/lvis_fix_minival_segm.py`

- Run few-shot pipeline.
  - Note: The following assertion must hold `assert (n_classes * n_shots) % world_size == 0`.
    - 1 shot: (1203 * 1) % 3 = 0 --> Use 1 or 3 GPUs 
    - 2 shot: (1203 * 2) % 6 = 0 --> Use 1 or 3 or 6 GPUs
    - 3 shot: (1203 * 3) % 9 = 0 --> Use 1 or 3 or 9 GPUs
    - 5 shot: (1203 * 5) % 15 = 0 --> Use 1 or 3 or 5 or 15 GPUs
    - 10 shot: (1203 * 10) % 30 = 0 --> Use 1 or 3 or 5 or 10 or 30 GPUs
  - lvis (1203 classes) factors: 1,3,401
  - lvis_common (461 classes) factors: 1, 461
  - lvis_frequent (405 classes) factors: 1, 3, 5, 9...
  - lvis_rare (337 classes) factors: 1, 337
  - Command format is as follows:
    - `CUDA_VISIBLE_DEVICES=1,2,3 zsh ./scripts/lvis_pipeline.sh <split> <n_shot> <n_gpus>`
    - Allowed splits are: `lvis`, `lvis_common`, `lvis_frequent`, `lvis_rare`, `lvis_minival`, `lvis_minival_common`, `lvis_minival_frequent`, `lvis_minival_rare`
    - E.g. `CUDA_VISIBLE_DEVICES=1,2,3 zsh ./scripts/lvis_pipeline.sh lvis 10 3`




## PASCAL VOC steps:
- Download PASCAL VOC dataset
```bash
cd data/pascal_voc
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
```

- Install dependency
`pip install mmengine-lite`

- Convert PASCAL VOC to COCO format
  - `python3 dev_hongyi/dataset/pascal_voc_to_coco.py --out-dir data/pascal_voc/annotations --out-format coco data/pascal_voc/VOCdevkit`

<!-- - To avoid choosing very bad reference images, first we filter reference images by using SAM model as upper bound. We prompt SAM with the ground truth bounding boxes and then filter out the reference images that the SAM model by default cannot segment properly given some perfect prompt (i.e. the ground truth).
`python3 dev_hongyi/dataset/pascal_voc_filter_bad_ref.py` -->

- To make our life easier, we are going to put all the JPEG images (VOC2007 and VOC2012) into one folder
Please, join the VOC2007 and VOC2012 JPEG images into a folder in `data/pascal_voc/allimages`.

- We change the filename (path) on the generated json file to the new JPEG images folder
`python3 dev_hongyi/dataset/change_filename_pascal.py`

- Run SAM segmentation with batches for PASCAL VOC (PASCAL VOC does not provide instance-segmentation masks, so we use the bounding boxes as prompt for SAM to generate masks)
  - `CUDA_VISIBLE_DEVICES=2,3 sh scripts/pascal_voc_box_to_segmentation.sh 2,3`
  - or single-gpu: `CUDA_VISIBLE_DEVICES=4 sh scripts/pascal_voc_box_to_segmentation.sh 4`

- Notes:
We will use `voc0712_trainval_with_segm.json` for generating n shot few-shot classes
We will use `voc07_test_with_segm.json` for evaluating on the split test

- Run few-shot pipeline for each PASCAL VOC split
  NOTE: We have 5 novel classes. For the code to run correctly this assertion must hold `assert (n_classes * n_shots) % world_size == 0`. (_The n_shots to run with PASCAL are: 1, 2, 3, 5, 10. Thus, using 5 GPUs is valid for all the n_shots._)
  - `CUDA_VISIBLE_DEVICES=1,2,3,4,5 ./scripts/pascal_split.sh 1 5`
  - `CUDA_VISIBLE_DEVICES=1,2,3,4,5 ./scripts/pascal_split.sh 2 5`
  - `CUDA_VISIBLE_DEVICES=1,2,3,4,5 ./scripts/pascal_split.sh 3 5`




