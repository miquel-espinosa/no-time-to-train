
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from sam2.automatic_mask_generator import (
    SAM2AutomaticMaskGenerator,
    generate_crop_boxes,
    MaskData,
    batch_iterator,
    calculate_stability_score,
    batched_mask_to_box,
    is_box_near_crop_edge,
    uncrop_masks,
    mask_to_rle_pytorch,
    batched_nms,
    uncrop_boxes_xyxy,
    uncrop_points,
    coco_encode_rle,
    rle_to_mask,
    area_from_rle
)



def kmeans_pp(X, K, max_iters=100):
    # if input is numpy array, convert to torch tensor
    if type(X) == np.ndarray:
        X = torch.from_numpy(X).float()
    centers = X[torch.randint(X.size(0), (1,))]
    for i in range(K - 1):
        distances = torch.cdist(X, centers).min(dim=1).values + 1e-6
        probs = distances ** 2 / torch.sum(distances ** 2)
        idx = torch.multinomial(probs, num_samples=1)
        centers = torch.cat([centers, X[idx]], dim=0)
    for i in range(max_iters):
        distances = torch.cdist(X, centers)
        cluster_assignment = torch.argmin(distances, dim=1)
        new_centers2 = torch.stack([torch.mean(X[cluster_assignment == k], dim=0)
                                   for k in range(K)], dim=0)
        new_centers = []
        for k in range(K):
            if torch.sum(cluster_assignment == k) > 0:
                new_centers.append(torch.mean(X[cluster_assignment == k], dim=0))
            else:
                new_centers.append(centers[k])
        new_centers = torch.stack(new_centers, dim=0)
        assert not torch.isnan(new_centers).any()

        if torch.all(torch.eq(new_centers, centers)):
            break
        centers = deepcopy(new_centers)
    return centers, cluster_assignment




class SAM2AutomaticMaskGenerator_Matcher(SAM2AutomaticMaskGenerator):
    @torch.no_grad()
    def generate(
        self,
        image,
        select_point_coords=None,
        select_point_labels=None,
        select_box=None,
        select_mask_input=None,
        dense_pred=False,
        extra_mask_data=None
    ):
        mask_data = self._generate_masks(
            image, select_point_coords, select_point_labels, select_box, select_mask_input, dense_pred, extra_mask_data
        )
        if dense_pred:
            return mask_data

        masks = mask_data["masks"]
        ious = mask_data["iou_preds"]
        return masks, ious

    def _generate_masks(
        self,
        image,
        select_point_coords,
        select_point_labels,
        select_box,
        select_mask_input,
        dense_pred,
        extra_mask_data
    ):
        orig_size = image.shape[:2]
        crop_box = [0, 0, orig_size[-1], orig_size[-2]]

        data = self._process_crop(
            image,
            crop_box,
            0,
            orig_size,
            select_point_coords,
            select_point_labels,
            select_box,
            select_mask_input,
            dense_pred,
            extra_mask_data
        )
        return data

    def _process_crop(
        self,
        image,
        crop_box,
        crop_layer_idx,
        orig_size,
        select_point_coords,
        select_point_labels,
        select_box,
        select_mask_input,
        dense_pred,
        extra_mask_data
    ):
        if dense_pred:
            assert extra_mask_data is None

        x0, y0, x1, y1 = crop_box
        cropped_im_size = (int(y1-y0), int(x1-x0))

        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        data = MaskData()

        if dense_pred:  # dense prediction
            for (points,) in batch_iterator(self.points_per_batch, points_for_image):
                batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size, normalize=True)
                data.cat(batch_data)
                del batch_data
            return data
        else:
            assert select_point_coords is not None
            assert select_point_labels is not None

            points_list_len = len(select_point_coords)

            if select_box is None:
                select_boxes = [None] * points_list_len
            elif isinstance(select_box, list) and len(select_box) == 1:
                select_boxes = select_box * points_list_len
            else:
                raise NotImplementedError

            if select_mask_input == None:
                select_masks_input = [None] * points_list_len
            elif isinstance(select_mask_input, list) and len(select_mask_input) == 1:
                select_masks_input = select_mask_input * points_list_len
            else:
                raise NotImplementedError


            for sel_points, sel_labels, sel_boxes, sel_masks in zip(
                    select_point_coords, select_point_labels, select_boxes, select_masks_input
            ):
                for (point_batch, label_batch) in batch_iterator(
                        self.points_per_batch, sel_points, sel_labels
                ):
                    batch_data = self._process_batch(
                        point_batch,
                        cropped_im_size,
                        crop_box,
                        orig_size,
                        normalize=True,
                        labels=label_batch,
                        boxes=sel_boxes,
                        masks=sel_masks
                    )
                    data.cat(batch_data)
                    del batch_data

        # Remove duplicates within this crop.
        if extra_mask_data is not None:
            data.cat(extra_mask_data)
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        return data

    def _process_batch(
        self,
        points,
        im_size,
        crop_box,
        orig_size,
        normalize=False,
        labels=None,
        boxes=None,
        masks=None,
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        if labels is not None:
            in_labels = torch.as_tensor(
            labels, dtype=torch.int, device=self.predictor.device
        )
        else:
            in_labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )

        if boxes is not None:
            in_boxes = self.predictor._transforms.transform_boxes(
                boxes, normalize=normalize, orig_hw=im_size
            )
        else:
            in_boxes = None

        if masks is not None:
            raise NotImplementedError
        else:
            in_masks = None

        if len(in_points.shape) == 2:
            in_points = in_points.unsqueeze(dim=1)
            in_labels = in_labels.unsqueeze(dim=1)

        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points,
            in_labels,
            boxes=in_boxes,
            mask_input=in_masks,
            multimask_output=self.multimask_output,
            return_logits=True,
        )

        # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            low_res_masks=low_res_masks.flatten(0, 1),
        )
        del masks

        if not self.use_m2m:
            # Filter by predicted IoU
            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            # Calculate and filter by stability score
            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)
        else:
            in_points = self.predictor._transforms.transform_coords(
                points.repeat_interleave(masks.shape[1], dim=0), normalize=normalize, orig_hw=im_size
            )
            labels = torch.ones(
                in_points.shape[0], dtype=torch.int, device=in_points.device
            )
            masks, ious = self.refine_with_m2m(
                in_points, labels, data["low_res_masks"], self.points_per_batch
            )
            data["masks"] = masks.squeeze(1)
            data["iou_preds"] = ious.squeeze(1)

            if self.pred_iou_thresh > 0.0:
                keep_mask = data["iou_preds"] > self.pred_iou_thresh
                data.filter(keep_mask)

            data["stability_score"] = calculate_stability_score(
                data["masks"], self.mask_threshold, self.stability_score_offset
            )
            if self.stability_score_thresh > 0.0:
                keep_mask = data["stability_score"] >= self.stability_score_thresh
                data.filter(keep_mask)

        # Threshold masks and calculate boxes
        data["masks"] = data["masks"] > self.mask_threshold
        data["boxes"] = batched_mask_to_box(data["masks"])

        # Filter boxes that touch crop boundaries
        keep_mask = ~is_box_near_crop_edge(
            data["boxes"], crop_box, [0, 0, orig_w, orig_h]
        )
        if not torch.all(keep_mask):
            data.filter(keep_mask)

        # Compress to RLE
        data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
        return data

