import numpy as np
import torch
from torch.utils.data import DataLoader

from dev_hongyi.utils import print_dict
from dev_hongyi.dataset.coco_ref_dataset import COCORefTrainDataset, COCOMemoryFillDataset, COCORefTestDataset
from dev_hongyi.models.SAM2Ref import SAM2Ref



if __name__ == "__main__":
    # device = torch.device("cuda")
    #
    # model = SAM2Ref(
    #     model_cfg="sam2_hiera_t.yaml",
    #     checkpoint_path="./checkpoints/sam2_hiera_tiny.pt",
    #     disable_custom_iou_embed=False
    # )
    #
    # model = model.to(device)
    # model.train()
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # exit()

    root = "./data/coco/val2017"
    json = "./data/coco/annotations_refsam2/instances_val2017_tiny_filtered_by_0.6.json"
    image_size = 1024
    remove_bad = True
    max_cat_num = 2
    max_mem_length = 4
    n_pos_points = 16
    neg_ratio = 2

    # dataset = COCOMemoryFillDataset(
    #     root="./data/coco/train2017",
    #     json_file="./data/coco/annotations_refsam2/memory/train2017_allClasses_length4_v1.json",
    #     image_size=1024,
    #     memory_length=4
    # )


    # dataset = COCORefTrainDataset(
    #     root=root,
    #     json_file=json,
    #     image_size=image_size,
    #     remove_bad=remove_bad,
    #     max_cat_num=max_cat_num,
    #     max_mem_length=max_mem_length,
    #     n_pos_points=n_pos_points,
    #     neg_ratio=neg_ratio
    # )

    dataset = COCORefTestDataset(
        root=root,
        json_file=json,
        image_size=image_size
    )

    print_dict(dataset[12])
    exit()

    #
    # print_dict(dataset[11])
    #
    # # 64 * 64 = 4096 masks for 1 target image with 1 class of reference
    # # 4096 * 80 = 327680 masks in total
    # # run NMS on 327680 masks and pick the top 100 remaining masks -> output
    # # output -> dump to json, using COCO evluation tool to get AP AP50
    #
    # import os
    # import cv2
    # from PIL import Image
    #
    # data = dataset[11]
    # img = Image.open(os.path.join("./data/coco/val2017", data["target_img_info"]["file_name"]))
    # img = np.array(img)
    # img = cv2.resize(img, (image_size, image_size))
    #
    # for cat_id in data["tar_anns_by_cat"].keys():
    #     n_pos = data["tar_anns_by_cat"][cat_id]["points_info"]["n_pos"]
    #     n_neg = data["tar_anns_by_cat"][cat_id]["points_info"]["n_neg"]
    #     n_rest = data["tar_anns_by_cat"][cat_id]["points_info"]["n_rest"]
    #
    #     pos_points = data["tar_anns_by_cat"][cat_id]["query_points"][:n_pos]
    #     print(data["tar_anns_by_cat"][cat_id]["bboxes"])
    #     print(pos_points)
    #     for j in range(n_pos):
    #         x = int(pos_points[j][0])
    #         y = int(pos_points[j][1])
    #         cv2.circle(img, (x, y), 5, color=(255, 0, 0))
    #     break
    #
    # img_draw = Image.fromarray(img)
    # img_draw.save("/home/s2139448/test_sam2.png")
    # exit()


    # train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda batch: batch)
    # for batch in train_loader:
    #     print("\nBatch 0")
    #     print_dict(batch[0])
    #     print("\nBatch 1")
    #     print_dict(batch[1])
    #
    #     losses = model(batch)
    #     print(losses)
    #     # print("\n")
    #     # print(pred_masks.shape)
    #     # print(pred_ious.shape)
    #     break

