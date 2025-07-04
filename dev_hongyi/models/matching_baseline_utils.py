
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from copy import deepcopy

# import faiss
# import faiss.contrib.torch_utils

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

def fast_l2(x, y, sqrt=True):
    # compute L2 distances between x, y in dim -2

    assert len(x.shape) == len(y.shape)
    assert x.shape[-1] == y.shape[-1]

    x_norms = torch.pow(x, 2.0).sum(dim=-1, keepdim=True)
    y_norms = torch.pow(y, 2.0).sum(dim=-1, keepdim=True)

    dist = x_norms + y_norms.transpose(-1, -2) - 2 * (x @ y.transpose(-1, -2))
    if sqrt:
        dist = torch.sqrt(torch.clamp(dist, min=0.0))
    return dist


def kmeans(feats, k, n_iter=100):
    assert len(feats.shape) == 2

    device = feats.device
    n, c = feats.shape

    centers = feats[torch.randperm(n)[:k].to(device=device)]  # [k, c]
    for i in range(n_iter):
        sim = F.normalize(feats, p=2, dim=-1) @ F.normalize(centers, p=2, dim=-1).t()  # [n, k]
        new_center_inds = torch.argmax(sim, dim=1)  # [n]

        new_centers = []
        for j in range(k):
            new_centers.append(feats[new_center_inds==j].mean(dim=0, keepdim=True))
        centers = torch.cat(new_centers, dim=0)  # [k, c]
    centers = F.normalize(centers, dim=-1)
    return centers


def kmeans_decouple(feats, feats_fore, k, n_iter=100):
    assert len(feats.shape) == 2

    device = feats.device
    n, c = feats.shape

    centers = feats_fore[torch.randperm(n)[:k].to(device=device)]  # [k, c]
    for i in range(n_iter):
        sim = F.normalize(feats, p=2, dim=-1) @ F.normalize(centers, p=2, dim=-1).t()  # [n, k]
        new_center_inds = torch.argmax(sim, dim=1)  # [n]

        new_centers = []
        for j in range(k):
            new_centers.append(feats_fore[new_center_inds==j].mean(dim=0, keepdim=True))
        centers = torch.cat(new_centers, dim=0)  # [k, c]

    sim_fore = F.normalize(feats_fore, p=2, dim=-1) @ F.normalize(centers, p=2, dim=-1).t()
    assign_inds = torch.argmax(sim_fore, dim=-1)

    new_centers = []
    for i in range(k):
        new_centers.append(
            feats[assign_inds == i].mean(dim=0, keepdim=True)
        )
    new_centers = torch.cat(new_centers, dim=0)
    centers = F.normalize(new_centers, dim=-1)
    return centers


def compute_foundpose(feats: torch.Tensor, masks: torch.Tensor, k_kmeans: int, n_pca: int):
    # Reference: https://github.com/facebookresearch/foundpose/

    device = feats.device
    n_class, n_shot = feats.shape[:2]
    mem_dim = feats.shape[-1]

    feats = feats.reshape(n_class * n_shot, -1, mem_dim)
    masks = feats.reshape(n_class * n_shot, -1)

    fore_feats = []
    for i in range(feats.shape[0]):
        fore_feats.append(feats[i][masks[i] > 0])
    fore_feats = torch.cat(fore_feats, dim=0)

    # PCA
    fore_feats_np = fore_feats.cpu().numpy()
    pca = PCA(n_components=n_pca)
    pca.fit(fore_feats_np)
    pca_mean = torch.from_numpy(pca.mean_).to(device=device)
    pca_components = torch.from_numpy(pca.components_).to(device=device)
    fore_feats = (fore_feats - pca_mean.reshape(1, -1)) @ pca_components.t()

    # K-means
    fore_feats_cpu = fore_feats.cpu()
    kmeans = faiss.Kmeans(
        n_pca,
        k_kmeans,
        niter=100,
        gpu=False,
        verbose=True,
        seed=0,
        spherical=False,
    )
    kmeans.train(fore_feats_cpu)

    centers = array_to_tensor(kmeans.centroids).to(device)

    centroid_distances, cluster_ids = kmeans.index.search(fore_feats_cpu, 1)
    centroid_distances = centroid_distances.squeeze(axis=-1).to(device=device)
    cluster_ids = cluster_ids.squeeze(axis=-1).to(device=device)

    # TF-IDF







def vis_kmeans(ref_imgs, ref_masks_ori, ref_cat_ind, ref_feats, ref_masks, feats_centers, encoder_shape_info, device, transparency):
    encoder_h = encoder_shape_info.get("height")
    encoder_w = encoder_shape_info.get("width")
    encoder_patch_size = encoder_shape_info.get("patch_size")

    assert encoder_h == encoder_w
    encoder_img_size = encoder_h * encoder_patch_size

    color_template = torch.tensor([
        [255, 0, 0],
        [148, 33, 146],
        [255, 251, 0],
        [170, 121, 66],
        [4, 51, 255],
        [0, 249, 0],
        [255, 64, 255],
        [0, 253, 255]
    ]).to(device=device, dtype=torch.float32)

    cat_centers = feats_centers[ref_cat_ind]
    n_centers = cat_centers.shape[0]
    assert n_centers <= len(color_template)

    center_assign = (
        F.normalize(ref_feats, p=2, dim=-1)
        @ cat_centers.t()
    ).max(dim=-1)[-1]

    center_assign[ref_masks == 0] = -1
    center_assign = center_assign.to(dtype=torch.long)

    canvas = torch.zeros(
        (encoder_h * encoder_w, encoder_patch_size, encoder_patch_size, 3), device=device
    )
    for i in range(len(center_assign)):
        if center_assign[i].item() != -1:
            canvas[i, :, :, :] += color_template[center_assign[i]]
    canvas = canvas.reshape(encoder_h, encoder_w, encoder_patch_size, encoder_patch_size, 3)
    canvas = canvas.permute(0, 2, 1, 3, 4).reshape((encoder_img_size, encoder_img_size, 3))

    vis_img = ref_imgs[0].permute(1, 2, 0) * 255.0

    color_ws = ref_masks_ori.reshape(encoder_img_size, encoder_img_size, 1) > 0
    color_ws = color_ws.to(dtype=torch.float32) * transparency
    out_vis = vis_img * (1 - color_ws) + canvas * color_ws
    return out_vis


def vis_pca(ref_imgs, ref_masks_ori, ref_cat_ind, ref_feats, ref_masks, pca_means, pca_components, encoder_shape_info, device, transparency):
    encoder_h = encoder_shape_info.get("height")
    encoder_w = encoder_shape_info.get("width")
    encoder_patch_size = encoder_shape_info.get("patch_size")

    assert encoder_h == encoder_w
    encoder_img_size = encoder_h * encoder_patch_size

    foreground_inds = ref_masks > 0

    foreground_feats = ref_feats[foreground_inds]
    pca_mean = pca_means[ref_cat_ind].unsqueeze(dim=0)
    pca_components = pca_components[ref_cat_ind]
    pca_weights = (foreground_feats - pca_mean) @ pca_components.t()
    _max_w = pca_weights.max()
    _min_w = pca_weights.min()
    pca_weights = (pca_weights - _min_w) / (_max_w - _min_w)

    rgb = torch.zeros((encoder_h * encoder_w, 3), device=device)
    rgb[foreground_inds] += pca_weights * 255.0

    canvas = torch.zeros(
        (encoder_h * encoder_w, encoder_patch_size, encoder_patch_size, 3), device=device
    )
    for i in range(encoder_h * encoder_w):
        canvas[i, :, :, :] += rgb[i]
    canvas = canvas.reshape(encoder_h, encoder_w, encoder_patch_size, encoder_patch_size, 3)
    canvas = canvas.permute(0, 2, 1, 3, 4).reshape((encoder_img_size, encoder_img_size, 3))

    vis_img = ref_imgs[0].permute(1, 2, 0) * 255.0

    color_ws = ref_masks_ori.reshape(encoder_img_size, encoder_img_size, 1) > 0
    color_ws = color_ws.to(dtype=torch.float32) * transparency
    out_vis = vis_img * (1 - color_ws) + canvas * color_ws
    return out_vis







class SAM2AutomaticMaskGenerator_MatchingBaseline(SAM2AutomaticMaskGenerator):
    @torch.no_grad()
    def generate(
        self,
        image,
        select_point_coords=None,
        select_point_labels=None,
        select_box=None,
        select_mask_input=None
    ):
        mask_data = self._generate_masks(
            image, select_point_coords, select_point_labels, select_box, select_mask_input
        )
        masks = mask_data["masks"]
        ious = mask_data["iou_preds"]
        return masks, ious, mask_data["low_res_masks"]

    def _generate_masks(
        self,
        image,
        select_point_coords,
        select_point_labels,
        select_box,
        select_mask_input
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
            select_mask_input
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
        select_mask_input
    ):
        x0, y0, x1, y1 = crop_box
        cropped_im_size = (int(y1-y0), int(x1-x0))

        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        data = MaskData()

        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size, normalize=True)
            data.cat(batch_data)
            del batch_data

        # Remove duplicates within this crop.
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
    ) -> MaskData:
        orig_h, orig_w = orig_size

        # Run model on this batch
        points = torch.as_tensor(
            points, dtype=torch.float32, device=self.predictor.device
        )
        in_points = self.predictor._transforms.transform_coords(
            points, normalize=normalize, orig_hw=im_size
        )
        in_labels = torch.ones(
            in_points.shape[0], dtype=torch.int, device=in_points.device
        )
        masks, iou_preds, low_res_masks = self.predictor._predict(
            in_points[:, None, :],
            in_labels[:, None],
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

