import torch
from torch import nn
import torch.nn.functional as F
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.data import MetadataCatalog
from .neck import YOSONeck
from .head import YOSOHead
from .loss import SetCriterion, HungarianMatcher
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.modeling.postprocessing import sem_seg_postprocess

from kornia.contrib import distance_transform

import random

from semantic_sam.utils import box_ops, get_iou
from semantic_sam.modules.criterion_interactive_many_to_many import SetCriterionOsPartWholeM2M
from semantic_sam.modules.many2many_matcher import M2MHungarianMatcher

__all__ = ["YOSO"]


@META_ARCH_REGISTRY.register()
class YOSO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.max_num_instance = 60
        self.num_mask_tokens = 6
        self.regenerate_point = True
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.in_features = cfg.MODEL.YOSO.IN_FEATURES
        self.num_classes = cfg.MODEL.YOSO.NUM_CLASSES
        self.num_proposals = cfg.MODEL.YOSO.NUM_PROPOSALS
        self.object_mask_threshold = cfg.MODEL.YOSO.TEST.OBJECT_MASK_THRESHOLD
        self.overlap_threshold = cfg.MODEL.YOSO.TEST.OVERLAP_THRESHOLD
        self.metadata =  MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        
        self.backbone = build_backbone(cfg)
        self.size_divisibility = cfg.MODEL.YOSO.SIZE_DIVISIBILITY
        if self.size_divisibility < 0:
            self.size_divisibility = self.backbone.size_divisibility
        
        self.sem_seg_postprocess_before_inference = (cfg.MODEL.YOSO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE or cfg.MODEL.YOSO.TEST.PANOPTIC_ON or cfg.MODEL.YOSO.TEST.INSTANCE_ON)
        self.semantic_on = cfg.MODEL.YOSO.TEST.SEMANTIC_ON
        self.instance_on = cfg.MODEL.YOSO.TEST.INSTANCE_ON
        self.panoptic_on = cfg.MODEL.YOSO.TEST.PANOPTIC_ON

        class_weight = cfg.MODEL.YOSO.CLASS_WEIGHT
        dice_weight = cfg.MODEL.YOSO.DICE_WEIGHT
        mask_weight = cfg.MODEL.YOSO.MASK_WEIGHT

        matcher = M2MHungarianMatcher(
            # cost_class=4.0,
            cost_mask=5.0,
            cost_dice=5.0,
            # cost_box=5.0,
            # cost_giou=2.0,
            num_points=12544,
            num_mask_tokens=6
        )
        
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        loss_list = ["labels", "masks"]

        self.criterion = SetCriterionOsPartWholeM2M(
            num_classes=1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=0.1,
            losses=['masks'],
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            dn='seg',
            dn_losses=['masks', 'dn_labels', 'boxes'],
            panoptic_on=False,
            semantic_ce_loss=False,
            num_mask_tokens=6
        )
        
        
        self.yoso_neck = YOSONeck(cfg=cfg, backbone_shape=self.backbone.output_shape()) # 
        self.yoso_head = YOSOHead(cfg=cfg, num_stages=cfg.MODEL.YOSO.NUM_STAGES) # 

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.to(self.device)

    def forward(self, batched_inputs):
        batched_inputs = batched_inputs if type(batched_inputs) == list else batched_inputs['sam']
        
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        backbone_feats = self.backbone(images.tensor)
        features = list()
        for f in self.in_features:
            features.append(backbone_feats[f])
        neck_feats = self.yoso_neck(features)

        prediction_switch = {'part': False, 'whole': False, 'seg': True, 'det': True}
        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                # targets = self.prepare_targets(gt_instances, images)
                targets = self.prepare_targets_interactive(gt_instances, images, prediction_switch=prediction_switch)
            else:
                targets = None

            outputs, mask_dict = self.yoso_head(neck_feats, targets)
            losses = self.criterion(outputs, targets, mask_dict, extra=prediction_switch)
            
            return losses
        else:
            losses, cls_scores, mask_preds = self.yoso_head(neck_feats, None)
            mask_cls_results = cls_scores
            mask_pred_results = mask_preds
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )


            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

            return processed_results

    def prepare_targets_interactive(self, targets, images, prediction_switch, task='seg'):
        """
        prepare targets for interactive segmentation, mainly includes:
            box:
            mask:
            labels: part / instance
        """
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []

        box_start = random.randint(int((self.max_num_instance - 1)/2), self.max_num_instance - 1)  # box based interactive after this number; about 1/4
        for targets_per_image in targets:
            gt_boxes = targets_per_image.gt_boxes if torch.is_tensor(targets_per_image.gt_boxes) else targets_per_image.gt_boxes.tensor
            # pad gt
            h, w = targets_per_image.image_size
            if not self.training:
                h_pad, w_pad = h, w

            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_masks = targets_per_image.gt_masks if torch.is_tensor(targets_per_image.gt_masks) else targets_per_image.gt_masks.tensor
            if not self.training:
                max_num_instance_ori = self.max_num_instance
                self.max_num_instance = len(gt_masks)
                box_start = self.max_num_instance # FIXME all points evaluation
            if len(gt_masks)==0:
                new_targets.append({
                    'boxes': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    'points': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    'boxes_dn': torch.ones(self.max_num_instance, 4).to(gt_masks).float(),
                    "pb": torch.cat([torch.ones(box_start), torch.zeros(self.max_num_instance - box_start)], 0),
                    'box_start': box_start
                })
                if not self.training:
                    self.max_num_instance = max_num_instance_ori
                continue
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            num_mask = targets_per_image.gt_classes.shape[0]

            index = torch.randperm(num_mask)
            if num_mask==0:
                print("wrong empty image! argets_per_image.gt_classes.shape[0] ", targets_per_image.gt_classes.shape[0], "targets_per_image", targets_per_image)
            if self.max_num_instance > num_mask:
                rep = 0 if num_mask==0 else int(self.max_num_instance/num_mask) + 1
                index = index.repeat(rep)
            index = index[:self.max_num_instance]
            box_start = self.max_num_instance
            level_target_inds = []
            # randomly sample one point as the user input
            if self.regenerate_point and box_start>0:
                point_coords = []
                for i in range(box_start):
                    mask = gt_masks[index[i]].clone()
                    center_point = True   # for evaluation sample the center as clicks
                    if not self.training and center_point:
                        mask = mask[None, None, :]
                        n, _, h, w = mask.shape
                        mask_dt = (distance_transform((~F.pad(mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float())[:, :, 1:-1, 1:-1])
                        # selected_index = torch.stack([torch.arange(n*_), mask_dt.max(dim=-1)[1].cpu()]).tolist()
                        selected_point = torch.tensor([mask_dt.argmax()/w, mask_dt.argmax()%w]).long().cuda().flip(0)
                    else:
                        candidate_indices = mask.nonzero()
                        if len(candidate_indices)==0:
                            print('wrong')
                            selected_point = torch.tensor([0, 0]).cuda()
                        else:
                            selected_index = random.randint(0, len(candidate_indices)-1)
                            selected_point = candidate_indices[selected_index].flip(0)
                        # only build level targets for sam data
                        if not prediction_switch['whole'] and not prediction_switch['part']:
                            level_target_ind = []
                            for ind, m in enumerate(gt_masks):
                                if m[tuple(selected_point.flip(0))]:
                                    level_target_ind.append(ind)
                            assert len(level_target_ind) > 0, "each point must have at least one target"
                            # randomly sample some target index if targets exceeds the maximum tokens
                            # FIXME another way is to filter small objects when too many level targets
                            if len(level_target_ind)>self.num_mask_tokens:
                                random.shuffle(level_target_ind)
                                level_target_ind = level_target_ind[:self.num_mask_tokens]
                            level_target_inds.append(level_target_ind)
                    selected_point = torch.cat([selected_point-3, selected_point+3], 0)
                    point_coords.append(selected_point)
                point_coords = torch.stack(point_coords).to('cuda')
            else:
                point_coords = targets_per_image.gt_boxes.tensor[index[:box_start]]
            max_num_tgt_per_click = -1
            if len(level_target_inds)>0:
                num_tgt = [len(l) for l in level_target_inds]
                max_num_tgt_per_click = max(num_tgt)
                if max_num_tgt_per_click>5:
                    print("max number of levels ", max(num_tgt))
            new_target={
                    "ori_mask_num": len(targets_per_image.gt_classes),
                    "level_target_inds": level_target_inds,
                    "max_num_tgt_per_click": max_num_tgt_per_click,
                    "labels": targets_per_image.gt_classes[index] if prediction_switch['whole'] else None,
                    "masks": padded_masks[index],
                    "ori_masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(gt_boxes[index])/image_size_xyxy,
                    "ori_boxes":box_ops.box_xyxy_to_cxcywh(gt_boxes)/image_size_xyxy,
                    "points":box_ops.box_xyxy_to_cxcywh(point_coords)/image_size_xyxy,
                    "pb": torch.cat([torch.ones(box_start), torch.zeros(self.max_num_instance - box_start)], 0),
                    "gt_whole_classes": targets_per_image.gt_whole_classes[index] if targets_per_image.has('gt_whole_classes') and prediction_switch['whole'] else None,
                    "gt_part_classes": targets_per_image.gt_part_classes[index] if targets_per_image.has('gt_part_classes') and prediction_switch['part'] else None,
                }
            # handle coco data format
            if prediction_switch['whole'] and not prediction_switch['part']:
                new_target['gt_whole_classes'] = targets_per_image.gt_classes[index]
                
            if not self.training:
                # transform targets for inference due to padding
                self.max_num_instance = max_num_instance_ori
                new_target["pb"]=torch.zeros_like(new_target["pb"])
                height = images[0].shape[1]
                width = images[0].shape[2]
                padded_h = images.tensor.shape[-2]  # divisable to 32
                padded_w = images.tensor.shape[-1]
                new_target["boxes_dn_ori"] = torch.cat([new_target["points"].clone(), new_target["boxes"][box_start:].clone()], 0)
                new_target['points'] = new_target['points'] * torch.as_tensor([width, height, width, height], dtype=torch.float, device=self.device)/torch.as_tensor([padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
                new_target['boxes'] = new_target['boxes'] * torch.as_tensor([width, height, width, height], dtype=torch.float, device=self.device)/torch.as_tensor([padded_w, padded_h, padded_w, padded_h], dtype=torch.float, device=self.device)
            new_target["boxes_dn"] = torch.cat([new_target["points"], new_target["boxes"][box_start:]], 0)
            new_target['box_start'] = box_start
            new_targets.append(new_target)

        return new_targets

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.num_classes, device=self.device).unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_proposals, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.num_classes # torch.div(topk_indices, self.num_classes, rounding_mode='floor') #
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        # if self.panoptic_on:
            # keep = torch.zeros_like(scores_per_image).bool()
            # for i, lab in enumerate(labels_per_image):
            #     keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            # scores_per_image = scores_per_image[keep]
            # labels_per_image = labels_per_image[keep]
            # mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
