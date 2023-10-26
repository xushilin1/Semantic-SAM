import math
from numpy import pad
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import torch
import torch.nn.functional as F
from torch import nn


from typing import Optional, List
from torch import nn, Tensor

from semantic_sam.body.decoder.utils import inverse_sigmoid, MLP, gen_sineembed_for_position

class FFN(nn.Module):
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, add_identity=True):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(nn.Linear(in_channels, feedforward_channels),
                              nn.ReLU(True),
                              nn.Dropout(0.0)
                              ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(0.0))
        self.layers = nn.Sequential(*layers)
        self.add_identity = add_identity
        self.dropout_layer = nn.Dropout(0.0)

    def forward(self, x, identity=None):
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class KernelUpdator(nn.Module):

    def __init__(self,
                 in_channels=256,
                 feat_channels=64,
                 out_channels=None,
                 input_feat_shape=3,
                 gate_sigmoid=True,
                 gate_norm_act=False,
                 activate_out=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN')):
        super(KernelUpdator, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels_raw = out_channels
        self.gate_sigmoid = gate_sigmoid
        self.gate_norm_act = gate_norm_act
        self.activate_out = activate_out
        if isinstance(input_feat_shape, int):
            input_feat_shape = [input_feat_shape] * 2
        self.input_feat_shape = input_feat_shape
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.out_channels = out_channels if out_channels else in_channels

        self.num_params_in = self.feat_channels
        self.num_params_out = self.feat_channels
        self.dynamic_layer = nn.Linear(
            self.in_channels, self.num_params_in + self.num_params_out)
        self.input_layer = nn.Linear(self.in_channels,
                                     self.num_params_in + self.num_params_out,
                                     1)
        self.input_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        self.update_gate = nn.Linear(self.in_channels, self.feat_channels, 1)
        if self.gate_norm_act:
            self.gate_norm = nn.LayerNorm(self.feat_channels)

        self.norm_in = nn.LayerNorm(self.feat_channels)
        self.norm_out = nn.LayerNorm(self.feat_channels)
        self.input_norm_in = nn.LayerNorm(self.feat_channels)
        self.input_norm_out = nn.LayerNorm(self.feat_channels)

        self.activation = nn.ReLU()

        self.fc_layer = nn.Linear(self.feat_channels, self.out_channels, 1)
        self.fc_norm = nn.LayerNorm(self.feat_channels)

    def forward(self, update_feature, input_feature):
        """
        Args:
            update_feature (torch.Tensor): [bs, num_proposals, in_channels]
            input_feature (torch.Tensor): [bs, num_proposals, in_channels]
        """
        bs, num_proposals, _ = update_feature.shape
        
        parameters = self.dynamic_layer(update_feature)
        param_in = parameters[..., :self.num_params_in]
        param_out = parameters[..., -self.num_params_out:]

        input_feats = self.input_layer(input_feature)
        input_in = input_feats[..., :self.num_params_in]
        input_out = input_feats[..., -self.num_params_out:]

        gate_feats = input_in * param_in
        if self.gate_norm_act:
            gate_feats = self.activation(self.gate_norm(gate_feats))

        input_gate = self.input_norm_in(self.input_gate(gate_feats))
        update_gate = self.norm_in(self.update_gate(gate_feats))
        if self.gate_sigmoid:
            input_gate = input_gate.sigmoid()
            update_gate = update_gate.sigmoid()
        param_out = self.norm_out(param_out)
        input_out = self.input_norm_out(input_out)

        if self.activate_out:
            param_out = self.activation(param_out)
            input_out = self.activation(input_out)

        # param_out has shape (batch_size, feat_channels, out_channels)
        features = update_gate * param_out + input_gate * input_out

        features = self.fc_layer(features)
        features = self.fc_norm(features)
        features = self.activation(features)

        return features

class CrossAttenHead(nn.Module):
    def __init__(self, cfg):
        super(CrossAttenHead, self).__init__()
        self.num_cls_fcs = cfg.MODEL.YOSO.NUM_CLS_FCS
        self.num_mask_fcs = cfg.MODEL.YOSO.NUM_MASK_FCS
        self.num_classes = cfg.MODEL.YOSO.NUM_CLASSES
        self.conv_kernel_size_2d = cfg.MODEL.YOSO.CONV_KERNEL_SIZE_2D

        self.hidden_dim = cfg.MODEL.YOSO.HIDDEN_DIM
        self.num_proposals = cfg.MODEL.YOSO.NUM_PROPOSALS
        self.hard_mask_thr = 0.5

        self.kernel_updator = KernelUpdator(
            in_channels=256,
            feat_channels=256,
            out_channels=256,
            input_feat_shape=3,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN')
        )

        self.s_atten = nn.MultiheadAttention(embed_dim=self.hidden_dim * self.conv_kernel_size_2d**2,
                                             num_heads=8,
                                             dropout=0.0)
        self.s_dropout = nn.Dropout(0.0)
        self.s_atten_norm = nn.LayerNorm(self.hidden_dim * self.conv_kernel_size_2d**2)

        self.ffn = FFN(self.hidden_dim, feedforward_channels=2048, num_fcs=2)
        self.ffn_norm = nn.LayerNorm(self.hidden_dim)

        self.iou_fcs = MLP(256, 256, 1, 3)

        # self.cls_fcs = nn.ModuleList()
        # for _ in range(self.num_cls_fcs):
        #     self.cls_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
        #     self.cls_fcs.append(nn.LayerNorm(self.hidden_dim))
        #     self.cls_fcs.append(nn.ReLU(True))
        # self.fc_cls = nn.Linear(self.hidden_dim, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(self.num_mask_fcs):
            self.mask_fcs.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False))
            self.mask_fcs.append(nn.LayerNorm(self.hidden_dim))
            self.mask_fcs.append(nn.ReLU(True))
        self.fc_mask = nn.Linear(self.hidden_dim, self.hidden_dim)

        prior_prob = 0.01
        self.bias_value = -math.log((1 - prior_prob) / prior_prob)

        self.apply(self._init_weights)
        # nn.init.constant_(self.fc_cls.bias, self.bias_value)

    def _init_weights(self, m):
        # print("init weights")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, features, proposal_kernels, mask_preds, self_attn_mask=None):
        B, C, H, W = features.shape

        soft_sigmoid_masks = mask_preds.sigmoid()
        nonzero_inds = soft_sigmoid_masks > self.hard_mask_thr
        hard_sigmoid_masks = nonzero_inds.float()

        # [B, N, C]
        f = torch.einsum('bnhw,bchw->bnc', hard_sigmoid_masks, features)
        
        num_proposals = proposal_kernels.shape[1]
        k = proposal_kernels.view(B, num_proposals, -1)

        # ----
        k = self.kernel_updator(f, k)
        # ----

        # [N, B, C]
        k = k.permute(1, 0, 2)

        k_tmp = self.s_atten(query = k, key = k, value = k, attn_mask=self_attn_mask)[0]
        k = k + self.s_dropout(k_tmp)
        k = self.s_atten_norm(k.permute(1, 0, 2))

        obj_feat = k.reshape(B, num_proposals, self.hidden_dim)

        obj_feat = self.ffn_norm(self.ffn(obj_feat))

        cls_feat = obj_feat
        mask_feat = obj_feat

        iou_pred = self.iou_fcs(cls_feat).squeeze(-1).view(B, -1, 6)
        
        # for cls_layer in self.cls_fcs:
        #     cls_feat = cls_layer(cls_feat)
        # cls_score = self.fc_cls(cls_feat).view(B, num_proposals, -1)
        cls_score = None

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        mask_kernels = self.fc_mask(mask_feat)
        new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, features)

        return iou_pred, cls_score, new_mask_preds, obj_feat


class YOSOHead(nn.Module):
    def __init__(self, cfg, num_stages):
        super(YOSOHead, self).__init__()
        self.num_all_tokens = 6
        self.num_mask_tokens = 6
        self.label_enc = nn.Embedding(2, 256)
        self.pb_embedding = nn.Embedding(2, 256)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, 256)
        self.pos_linear = nn.Linear(512, 256)

        self.num_stages = num_stages
        self.temperature = cfg.MODEL.YOSO.TEMPERATIRE

        # self.kernels = nn.Embedding(cfg.MODEL.YOSO.NUM_PROPOSALS, cfg.MODEL.YOSO.HIDDEN_DIM)
        self.mask_heads = nn.ModuleList()
        
        for _ in range(self.num_stages):
            self.mask_heads.append(CrossAttenHead(cfg))

    def prepare_for_dn_mo(self, targets, tgt, refpoint_emb, batch_size):
       
        # scalar, noise_scale = self.dn_num, self.noise_scale
        scalar, noise_scale = 100, 0.4

        pb_labels = torch.stack([t['pb'] for t in targets])
        labels = torch.zeros_like(pb_labels).long()
        boxes = torch.stack([t['boxes_dn'] for t in targets])
        box_start = [t['box_start'] for t in targets]

        known_labels = labels
        known_pb_labels = pb_labels

        known_bboxs = boxes
        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        if noise_scale > 0 and self.training:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :, :2] = known_bbox_expand[:, :, 2:] / 2
            diff[:, :, 2:] = known_bbox_expand[:, :, 2:]
            # add very small noise to input points
            sc = 0.01
            for i, st in enumerate(box_start):
                diff[i, :st] = diff[i, :st] * sc
            known_bbox_expand += torch.mul((torch.rand_like(known_bbox_expand) * 2 - 1.0), diff).cuda() * noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        m = known_labels_expaned.long().to('cuda')
        m_pb = known_pb_labels_expaned.long().to('cuda')
        input_label_embed = self.label_enc(m) + self.pb_embedding(m_pb)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        label_embed = input_label_embed.repeat_interleave(self.num_all_tokens,1)
        input_label_embed = label_embed + self.mask_tokens.weight.unsqueeze(0).repeat(input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(self.num_all_tokens,1)

        single_pad = self.num_all_tokens

        scalar = int(input_label_embed.shape[1]/self.num_all_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1]>0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        attn_mask[pad_size:, :pad_size] = True
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'pad_size': pad_size,
            'scalar': scalar,
        }
        return input_query_label,input_query_bbox,attn_mask,mask_dict

    def forward(self, features, targets, self_attn_mask=None):
        
        input_query_label, input_query_bbox, \
            tgt_mask, mask_dict = self.prepare_for_dn_mo(targets, None, None, features.shape[0])
        # object_kernels = self.kernels.weight.repeat(features.shape[0], 1, 1)

        pos_embed = self.pos_linear(gen_sineembed_for_position(input_query_bbox.sigmoid()))
        object_kernels = input_query_label + pos_embed

        all_stage_cls_preds = []
        all_stage_mask_preds = []
        all_stage_iou_preds = []
        mask_preds = torch.einsum('bnc,bchw->bnhw', object_kernels, features)
        for stage in range(self.num_stages):
            mask_head = self.mask_heads[stage]
            iou_preds, cls_scores, mask_preds, object_kernels = mask_head(
                features, object_kernels, mask_preds, self_attn_mask)
            if cls_scores is not None:
                cls_scores = cls_scores / self.temperature
            all_stage_cls_preds.append(cls_scores)
            all_stage_mask_preds.append(mask_preds)
            all_stage_iou_preds.append(iou_preds)
        
        
        out = {
            'pred_logits': all_stage_cls_preds[-1],
            'pred_masks': all_stage_mask_preds[-1],
            'pred_ious': all_stage_iou_preds[-1],
            'aux_outputs': self._set_aux_loss(
                all_stage_cls_preds,
                all_stage_mask_preds, 
                all_stage_iou_preds, 
            )
        }
        return out, mask_dict


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class=None, outputs_seg_masks=None, predictions_iou_score=None):
        return [
            {"pred_logits": a, "pred_masks": b, "pred_ious":c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], predictions_iou_score[:-1])
        ]