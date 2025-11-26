# -*- coding: utf-8 -*-
import torch
from torch import nn
from torchvision.models.resnet import resnet34, resnet50
from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg16
from torchvision.models.resnet import Bottleneck
from network.transformer import SalTransformer
from network.weight_init import trunc_normal_
import torch.utils.model_zoo as model_zoo

from network.feat_modulation_module import AttentionBasedModulator
from network.models_gridms2 import average_representations2
from network.pow_models import SalMorphologyLearner2


import math
import time

from heatmap import putGaussianMaps
import copy
import numpy as np

from functools import partial



def get_sinusoid_position_encoding(d_hid, n_position=12*12):
    ''' Sinusoid position encoding table '''

    # TODO: make it with torch instead of numpy

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class CascadeSalTransformer(nn.Module):
    def __init__(self, num_salT_layers=1, patch_grids=(12, 12), dim=2048, heads=16, qk_same=False, mlp_ratio=2,
                 has_CLS=False, has_pos_embed=True, vis=False, att_type='rbf',
                 apply_post_LN=1, vit_fused_feat_dim=-1, vit_fusion_process='cat', raw_dim=2048, att_modul_on_raw=False, att_modul_type=0, support_fc_type='diag', init_value=1.0, use_kp_proto=True,
                 **sal_mask_cfg):
        super(CascadeSalTransformer, self).__init__()
        # if vit dim is not equal to raw input dim (i.e., CNN feat. dim), we use conv1x1 to transform dim
        self.dim = dim
        self.use_pre_conv1x1 = (dim != raw_dim)
        if self.use_pre_conv1x1:
            self.pre_conv1x1 = nn.Sequential(
                nn.Conv2d(raw_dim, dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(True),
            )


        self.has_CLS = has_CLS
        self.has_pos_embed = has_pos_embed

        if self.has_CLS and num_salT_layers>=1:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        if num_salT_layers>=1:
            if self.has_pos_embed == 'APE1':  # APE1: learnable Absolute PE
                n_patches = patch_grids[0] * patch_grids[1]
                self.position_embeddings = nn.Parameter(torch.zeros(1, int(has_CLS)+n_patches, dim))
                self.pos_dropout = nn.Dropout(p=0.0)
                # init weights for position_embeddings (optional, some works also init it as 0)
                # trunc_normal_(self.position_embeddings, std=1e-6)  # position_embeddings = 0, 1e-6, or sqrt(2/C)
            elif self.has_pos_embed == 'APE2':  # APE2: sinusoidal Absolute PE
                n_patches = patch_grids[0] * patch_grids[1]
                self.position_embeddings = get_sinusoid_position_encoding(d_hid=dim, n_position=int(has_CLS)+n_patches)
                if torch.cuda.is_available():
                    self.position_embeddings = self.position_embeddings.cuda()
                self.pos_dropout = nn.Dropout(p=0.0)
            print('PE type is ', self.has_pos_embed)
        # saliency mask transformers
        self.num_salT_layers = num_salT_layers
        self.vis = vis
        self.salT = SalTransformer(dim, depth=num_salT_layers, heads=heads, qk_same=qk_same, mlp_ratio=mlp_ratio,
                                   has_CLS=has_CLS, vis=self.vis, att_type=att_type, att_pos_enc=self.has_pos_embed, grid_size=patch_grids)
        self.apply_post_LN = apply_post_LN
        if att_type in ['softmax-fg', 'rbf-fg', 'softmax-fg2', 'rbf-fg2', 'softmax-fg3', 'rbf-fg3']:
            assert sal_mask_cfg['sal_mask_type'] != 'N', 'it needs sal_mask (sal_mask_type) to perform masked self-attention in CascadeSalTransformer'
        # -----------------------------------------------------
        # feature modulator
        self.att_modul_type = att_modul_type
        att_modulator_list = []
        self.att_modul_on_raw = att_modul_on_raw
        self.vit_fused_feat_dim = vit_fused_feat_dim  # -1, don't assign dim; 2048, assign fixed dim for fused vit feats
        self.vit_fusion_process = vit_fusion_process  # 'cat' or 'add' for fusing vit features
        self.support_fc_type = support_fc_type
        if self.att_modul_type == 0:  # 0, all vit feats are taken;
            if self.apply_post_LN == 1:
                self.post_vit_LNs = self.add_post_LNs(num_salT_layers, dim)
            elif self.apply_post_LN == 2:
                self.post_raw_LNs = self.add_post_LNs(1, dim)
                self.post_vit_LNs = self.add_post_LNs(num_salT_layers, dim)
            else:
                pass

            if num_salT_layers >= 1 and self.vit_fused_feat_dim > 0:
                self.conv1x1s = self.add_conv1x1s(in_dim=dim, out_dim=self.vit_fused_feat_dim, num=num_salT_layers, fusion_process=self.vit_fusion_process)

            # 'add' fusion needs LN to scale feature
            # if num_salT_layers >= 2 and self.vit_fusion_process == 'add':
            #     if self.vit_fused_feat_dim > 0:
            #         self.feat_fusion_LN = nn.LayerNorm(self.vit_fused_feat_dim, eps=1e-6)
            #     else:
            #         self.feat_fusion_LN = nn.LayerNorm(dim, eps=1e-6)

            self.num_att_modul = 1  # int(num_salT_layers >= 1) + int(self.att_modul_on_raw)
            assert self.num_att_modul >= 1, 'num_att_modul should be >=1, CascadeSalTransformer'

        elif self.att_modul_type == 1:  # last ViT feat is taken
            if self.apply_post_LN == 1:
                self.post_vit_LNs = self.add_post_LNs(1, dim)
            elif self.apply_post_LN == 2:
                self.post_raw_LNs = self.add_post_LNs(1, dim)
                self.post_vit_LNs = self.add_post_LNs(1, dim)
            else:
                pass

            if num_salT_layers >= 1 and self.vit_fused_feat_dim > 0:
                self.conv1x1s = self.add_conv1x1s(in_dim=dim, out_dim=self.vit_fused_feat_dim, num=1, fusion_process=self.vit_fusion_process)

            self.num_att_modul = 1  # int(num_salT_layers >= 1) + int(self.att_modul_on_raw)

        elif self.att_modul_type == 2:  # last and middle vit feats are taken
            if self.apply_post_LN == 1:
                self.post_vit_LNs = self.add_post_LNs(2, dim)
            elif self.apply_post_LN == 2:
                self.post_raw_LNs = self.add_post_LNs(1, dim)
                self.post_vit_LNs = self.add_post_LNs(2, dim)
            else:
                pass

            assert num_salT_layers >= 2, 'att_modul_type==2 requires at least 2 ViTs, CascadeSalTransformer'

            last_index = num_salT_layers - 1
            middle_index= last_index - num_salT_layers // 2
            self.vit_layer_index = [middle_index, last_index]

            if self.vit_fused_feat_dim > 0:
                self.conv1x1s = self.add_conv1x1s(in_dim=dim, out_dim=self.vit_fused_feat_dim, num=2, fusion_process=self.vit_fusion_process)

            # 'add' fusion needs LN to scale feature
            # if num_salT_layers >= 2 and self.vit_fusion_process == 'add':
            #     if self.vit_fused_feat_dim > 0:
            #         self.feat_fusion_LN = nn.LayerNorm(self.vit_fused_feat_dim, eps=1e-6)
            #     else:
            #         self.feat_fusion_LN = nn.LayerNorm(dim, eps=1e-6)

            self.num_att_modul = 1  # int(num_salT_layers >= 1) + int(self.att_modul_on_raw)

        elif self.att_modul_type == 3:  # [last ViT concat CNN feats]
            assert num_salT_layers >= 1, 'att_modul_type==3 requires at least 1 ViTs, CascadeSalTransformer'

            if self.apply_post_LN == 1:
                self.post_vit_LNs = self.add_post_LNs(2 * int(num_salT_layers >= 1), dim)
            elif self.apply_post_LN == 2:
                self.post_raw_LNs = self.add_post_LNs(1, dim)
                self.post_vit_LNs = self.add_post_LNs(2 * int(num_salT_layers >= 1), dim)
            else:
                pass

            if self.vit_fused_feat_dim > 0:
                self.conv1x1s = self.add_conv1x1s(in_dim=dim, out_dim=self.vit_fused_feat_dim, num=2, fusion_process=self.vit_fusion_process)

            self.num_att_modul = 1  # int(num_salT_layers >= 1) + int(self.att_modul_on_raw)
        else:
            print('Error for att_modul_type in CascadeSalTransformer.')
            exit(0)

        assert self.num_att_modul >= 1, 'num_att_modul >= 1, CascadeSalTransformer'
        for i in range(self.num_att_modul):
            att_modulator_list.append(AttentionBasedModulator(dim, support_fc_type=self.support_fc_type, init_value=init_value, num_layers=1))
        self.att_modulators = nn.Sequential(*att_modulator_list)
        self.use_kp_proto = use_kp_proto

        # -----------------------------------------------------
        # mask morphology learners
        # 'N': no mask; 'P': pure mask;
        # 'S': mask shared; 'IS': diff mask, shared morphology unit(MU); 'II', diff mask, diff UI
        self.sal_mask_type = sal_mask_cfg['sal_mask_type']
        if self.sal_mask_type in ['S', 'IS', 'II']:
            self.morphology_learner = SalMorphologyLearner2(
                in_ch=sal_mask_cfg['in_ch'],    # 1
                out_ch=sal_mask_cfg['out_ch'],  # 1
                src_size=sal_mask_cfg['src_size'],  # mask size: (384, 384)
                dst_size=sal_mask_cfg['dst_size'],  # target grid size: (12, 12)
                pre_downscale=sal_mask_cfg['pre_downscale'],  # downsize mask by this ratio (>=2); if <= 1, it means no using of pre-downscale
                num_vits=num_salT_layers,
                sal_mask_type=sal_mask_cfg['sal_mask_type'],  # 'S', 'IS', 'II'
                morphology_type=sal_mask_cfg['morphology_type'],  # 'pow', 'mnn'
                sem_cfg=sal_mask_cfg['sem_cfg'],  # (embedding_dim, hidden_layers (conv3x3), out_dim)
                norm_type=sal_mask_cfg['norm_type'],  # 'bn', 'ln', 'no' (no norm)
                vit_feat_proj_cfg=[dim],  # (vit_feat_dim)
                mpg_cfg=sal_mask_cfg['mpg_cfg'],  # (mpg_hidden_dim, mpg_layer)
                pow_cfg=sal_mask_cfg['pow_cfg'],  # (num_pow_param, regress_type, bound, T, num_conv, conv_ks, conv_reg_type) for 'pow' morphology
                mnn_cfg=sal_mask_cfg['mnn_cfg'],  # (SE_size, type, mnn_layer, T, T2, prob) for MNN
                impose_reg=sal_mask_cfg['impose_reg'],  # impose regularization loss for parameters
                pn_center=sal_mask_cfg['pn_center'],  # the power normalization center for regularization, e.g., (R- center)**2.
                fixed_MC=sal_mask_cfg['fixed_MC'],  # 0, don't fixed; >0 fixed pow for 'pow based MC' or fixed SE for 'MNN'
                vis=vis,  # whether visualize
            )
        self.sal_mask_cfg = sal_mask_cfg

    def add_post_LNs(self, num_LNs=0, dim=2048):
        post_layer_norm_list = []
        for i in range(num_LNs):
            post_layer_norm_list.append(nn.LayerNorm(dim, eps=1e-6))
        return nn.Sequential(*post_layer_norm_list)

    def add_conv1x1s(self, in_dim, out_dim, num, fusion_process='cat'):
        '''
        fusion_process: 'cat' or 'add'.
                        if 'cat', the out_dim will be evenly divided
                        if 'add', each 1x1conv has same out_dim
        '''
        conv_list = []
        if fusion_process == 'cat':
            dim_div = out_dim // num
            dim_first = out_dim - (num-1) * dim_div
            for i in range(num): 
                if i == 0:
                    conv_list.append(
                        nn.Sequential(
                            # nn.Conv2d(in_dim, dim_first, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.Linear(in_dim, dim_first, bias=False),
                            nn.ReLU(True),
                        )
                    )
                else:
                    conv_list.append(
                        nn.Sequential(
                            # nn.Conv2d(in_dim, dim_div, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.Linear(in_dim, dim_div, bias=False),
                            nn.ReLU(True),
                        )
                    )
        elif fusion_process == 'add':
            for i in range(num):
                conv_list.append(
                        nn.Sequential(
                            # nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.Linear(in_dim, out_dim, bias=False),
                            nn.ReLU(True),
                        )
                    )

        return nn.Sequential(*conv_list)

    def apply_conv1x1(self, feat_list, fusion_process='cat'):
        '''
        apply a set of 1x1 conv to fuse features and output has same dim of B x (H*W) x C
        feat_list: a list of features, each with B x (H*W) x C
        '''
        out = []
        for i in range(len(feat_list)):
            o = self.conv1x1s[i](feat_list[i])
            out.append(o)
        if fusion_process == 'cat':
            out = torch.cat(out, dim=2)
        elif fusion_process == 'add':
            out = sum(out)

        return out

    def forward(self, support_feat, support_saliency, support_im, salT_cfg):

        B1, _, h, w = support_feat.shape  # (feature space)
        C = self.dim  # vit_dim may be different from raw_dim (namely support_feat's dim)
        Np = h * w

        # Use pre-conv1x1 to transform from raw_dim to vit_dim
        if self.use_pre_conv1x1:
            support_feat_tmp = self.pre_conv1x1(support_feat)
            # B x C x h x w --> B x Np x C
            support_feat_tmp = support_feat_tmp.flatten(2).transpose(2, 1)
        else:
            # B x C x h x w --> B x Np x C
            support_feat_tmp = support_feat.flatten(2).transpose(2, 1)

        if self.has_CLS and self.num_salT_layers>=1:
            cls_tokens_s = self.cls_token.expand(B1, -1, -1)
            support_feat_tmp = torch.cat((cls_tokens_s, support_feat_tmp), dim=1)  # B1 x (1+Np) x C
        if self.num_salT_layers>=1:
            if self.has_pos_embed in ['APE1', 'APE2']:
                support_feat_tmp = support_feat_tmp + self.position_embeddings
                support_feat_tmp = self.pos_dropout(support_feat_tmp)

        # -----------------------------------------------------
        # pow map learning
        if self.sal_mask_type == 'N':  # no mask
            s_sal_masks = None
            s_morphology_func = None
            s_mask_feats = None
        elif self.sal_mask_type == 'P':  # pure mask
            # B x 1 x H x W (image space)
            assert support_saliency.shape[1] == 1, 'saliency channel should be 1'
            final_scale = self.sal_mask_cfg['src_size'][0] // self.sal_mask_cfg['dst_size'][0]
            # B x 1 x h (H//s) x w (W//s)
            s_sal = torch.nn.functional.avg_pool2d(support_saliency, kernel_size=final_scale, stride=final_scale, padding=0)
            s_sal_masks = [s_sal.reshape(B1, h * w)]  # list, 1 element: B x Np (equal h x w)
            s_morphology_func = None
            s_mask_feats = None
        elif self.sal_mask_type in ['S', 'IS', 'II']:
            s_sal_masks = None
            # B x 1 x H x W (image space)
            assert support_saliency.shape[1] == 1, 'saliency channel should be 1'
            if self.sal_mask_cfg['in_ch'] == 1:  # only take saliency map as input to get saliency embedding
                s_mask_feats, s_low_res_masks = self.morphology_learner.compute_mask_feats(support_saliency)
            elif self.sal_mask_cfg['in_ch'] == 4:  # take saliency map + RGB image as input get saliency embedding
                s_mask_feats, s_low_res_masks = self.morphology_learner.compute_mask_feats(support_saliency, support_im)
            else:
                raise NotImplementedError
            s_morphology_func = partial(self.morphology_learner.forward, mask_feats=s_mask_feats, low_res_masks=s_low_res_masks)



        # -----------------------------------------------------
        # cascaded saliency-guided transformer
        mask_mode = salT_cfg['mask_mode']
        func_mode = salT_cfg['func_mode']
        beta = salT_cfg['sal_beta']
        eps = salT_cfg['sal_eps']
        thresh = salT_cfg['thresh']
        use_max = salT_cfg['use_max']
        rbf_l2norm = salT_cfg['rbf_l2norm']
        rownorm = salT_cfg['rownorm']

        vit_s_feats, vit_s_attn_probs, s_vis_morph_masks, s_morph_reg_losses = self.salT(support_feat_tmp, self.sal_mask_type, s_sal_masks, s_morphology_func, \
                                        mask_mode, func_mode, beta, eps, thresh, use_max, rbf_l2norm, rownorm)

        if self.has_CLS and self.num_salT_layers>=1:
            for i in range(len(self.att_modulators)):
                vit_s_feats[i] = vit_s_feats[i][:, 1:]  # exclude CLS, B x (1+Np) x C

        # unify the morphology regularization loss for support & query masks
        if (self.sal_mask_type in ['S', 'IS', 'II']) and self.morphology_learner.impose_reg>0:
            loss_offset = self.sal_mask_cfg['reg_loss_offset']  # 0., 0.0625
            # s_morph_reg_losses = torch.clamp(s_morph_reg_losses - loss_offset, min=0)
            s_morph_reg_losses = torch.stack(s_morph_reg_losses)  # num_ViTs x B1
            morph_reg_loss = (s_morph_reg_losses-loss_offset).clamp(min=0).mean(dim=0)  # B1
            morph_reg_loss_o = (s_morph_reg_losses).mean(dim=0)  # B1
        else:
            morph_reg_loss = 0
            morph_reg_loss_o = 0

        # -----------------------------------------------------
        # feature post LN & feature selection
        s_feats_select = []
        # assert len(self.att_modulators) >= 1, 'num of att_modulators should >= 1, CascadeSalTransformer'
        s_feats_select = self.select_raw_feats(s_feats_select, support_feat)
        if self.att_modul_type == 0:  # 0, all vit feats are taken;
            if self.num_salT_layers == 0:  # no vits, self.att_modul_on_raw must be true
                pass
            else:
                if self.apply_post_LN == 1 or self.apply_post_LN == 2:  # apply LN for vit feats
                    for i in range(0, len(vit_s_feats)):
                        vit_s_feats[i] = self.post_vit_LNs[i](vit_s_feats[i])
                else:
                    pass

                # reshape
                # B x Np x C --> B x C x h x w
                # for i in range(len(vit_s_feats)):
                #     vit_s_feats[i] = vit_s_feats[i].transpose(2, 1).reshape(B1, C, h, w).contiguous()
                #     vit_q_feats[i] = vit_q_feats[i].transpose(2, 1).reshape(B2, C, h, w).contiguous()

                if self.vit_fused_feat_dim > 0:
                    s_fused_feat = self.apply_conv1x1(vit_s_feats, self.vit_fusion_process)
                else:
                    if self.vit_fusion_process == 'cat':
                        s_fused_feat = torch.cat(vit_s_feats, dim=2)
                    elif self.vit_fusion_process == 'add':
                        s_fused_feat = sum(vit_s_feats)

                # 'add' fusion needs LN to scale feature
                # if self.num_salT_layers >= 2 and self.vit_fusion_process == 'add':
                #     s_fused_feat = self.feat_fusion_LN(s_fused_feat)
                #     q_fused_feat = self.feat_fusion_LN(q_fused_feat)

                # reshape
                # B x Np x C --> B x C x h x w
                s_fused_feat = s_fused_feat.transpose(2, 1).reshape(B1, -1, h, w).contiguous()

                s_feats_select.append(s_fused_feat)

        elif self.att_modul_type == 1:  # last ViT feat is taken
            if self.num_salT_layers == 0:  # no vits, self.att_modul_on_raw must be true
                pass
            else:
                vit_s_feats_rear = vit_s_feats[-1]

                if self.apply_post_LN == 1 or self.apply_post_LN == 2:  # apply LN for vit feats
                    vit_s_feats_rear = self.post_vit_LNs[0](vit_s_feats_rear)

                # # reshape
                # # B x Np x C --> B x C x h x w
                # vit_s_feats_rear = vit_s_feats_rear.transpose(2, 1).reshape(B1, C, h, w).contiguous()
                # vit_q_feats_rear = vit_q_feats_rear.transpose(2, 1).reshape(B2, C, h, w).contiguous()

                if self.vit_fused_feat_dim > 0:
                    vit_s_feats_rear = self.apply_conv1x1([vit_s_feats_rear], self.vit_fusion_process)

                # reshape
                # B x Np x C --> B x C x h x w
                vit_s_feats_rear = vit_s_feats_rear.transpose(2, 1).reshape(B1, -1, h, w).contiguous()

                s_feats_select.append(vit_s_feats_rear)

        elif self.att_modul_type == 2:  # last and middle vit feats are taken
            # when att_modul_type == 2, there are at least 2 vits

            middle_ind, last_ind = self.vit_layer_index
            s_vit_feats_select = [vit_s_feats[middle_ind], vit_s_feats[last_ind]]

            if self.apply_post_LN == 1 or self.apply_post_LN == 2:  # apply LN for vit feats
                for i in range(0, len(s_vit_feats_select)):
                    s_vit_feats_select[i] = self.post_vit_LNs[i](s_vit_feats_select[i])

            # reshape
            # B x Np x C --> B x C x h x w
            # for i in range(len(s_vit_feats_select)):
            #     s_vit_feats_select[i] = s_vit_feats_select[i].transpose(2, 1).reshape(B1, C, h, w).contiguous()
            #     q_vit_feats_select[i] = q_vit_feats_select[i].transpose(2, 1).reshape(B2, C, h, w).contiguous()

            if self.vit_fused_feat_dim > 0:
                s_fused_feat = self.apply_conv1x1(s_vit_feats_select, self.vit_fusion_process)
            else:
                if self.vit_fusion_process == 'cat':
                    s_fused_feat = torch.cat(s_vit_feats_select, dim=2)
                elif self.vit_fusion_process == 'add':
                    s_fused_feat = sum(s_vit_feats_select)

            # 'add' fusion needs LN to scale feature
            # if self.num_salT_layers >= 2 and self.vit_fusion_process == 'add':
            #     s_fused_feat = self.feat_fusion_LN(s_fused_feat)
            #     q_fused_feat = self.feat_fusion_LN(q_fused_feat)

            # reshape
            # B x Np x C --> B x C x h x w
            s_fused_feat = s_fused_feat.transpose(2, 1).reshape(B1, -1, h, w).contiguous()

            s_feats_select.append(s_fused_feat)

        else:
            pass

        # Basically we can fuse all features and just use ONE modulator, as the fiber extraction and modulation
        # operations are linear
        fuse_all_feats = True
        if fuse_all_feats:
            s_feats_select = [torch.cat(s_feats_select, dim=1)]


        return s_feats_select, vit_s_attn_probs, s_vis_morph_masks, morph_reg_loss, morph_reg_loss_o, s_mask_feats


    def select_raw_feats(self, s_feats_select, support_feat):
        '''
        input: B x C0 x h x w
        '''
        B1, C0, h, w = support_feat.shape

        if self.apply_post_LN == 2:  # apply LN for raw S & Q feats
            support_feat_ln = support_feat.flatten(2).transpose(2, 1)
            support_feat_ln = self.post_raw_LNs[0](support_feat_ln)
            support_feat_ln = support_feat_ln.transpose(2, 1).reshape(B1, C0, h, w)

        if self.att_modul_on_raw:
            if self.apply_post_LN == 0 or self.apply_post_LN == 1:
                s_feats_select.append(support_feat)
            elif self.apply_post_LN == 2:
                s_feats_select.append(support_feat_ln)
            else:
                pass

        return s_feats_select

    def support_query_modulation(self, s_feats_select, q_feats_select,
                support_kps, support_kp_mask, support_kps_aux=None, support_kp_mask_aux=None, extract_kp_repres_func=None,
                fused_attention=None, output_attention_maps=False):
        '''
        att_modulators: a sequential contains several 'AttentionBasedModulator'
        '''
        # -----------------------------------------------------
        # feature modulation
        attentive_feats = []
        attention_maps_l2 = []
        use_auxiliary_kps = (support_kps_aux is not None) and (support_kp_mask_aux is not None)
        if use_auxiliary_kps:
            attentive_feats_aux = []
            attention_maps_l2_aux = []
        else:
            attentive_feats_aux = None
            attention_maps_l2_aux = None

        for i in range(len(self.att_modulators)):
            s_feat_per_layer = s_feats_select[i]  # B1 x C x h x w
            q_feat_per_layer = q_feats_select[i]  # B2 x C x h x w

            support_repres, _ = extract_kp_repres_func(features=s_feat_per_layer, labels=support_kps, kp_mask=support_kp_mask)
            if use_auxiliary_kps:
                support_repres_aux, _ = extract_kp_repres_func(features=s_feat_per_layer, labels=support_kps_aux, kp_mask=support_kp_mask_aux)


            if fused_attention is None:
                is_normalized = False
            elif ('spatial-suc' in fused_attention):
                is_normalized = True
            else:
                is_normalized = False
            if is_normalized == True:
                support_repres = torch.nn.functional.normalize(support_repres, p=2, dim=1, eps=1e-12)  # B1 x C x N
                if use_auxiliary_kps:
                    support_repres_aux = torch.nn.functional.normalize(support_repres_aux, p=2, dim=1, eps=1e-12)  # B1 x C x N2

            if self.use_kp_proto:  # compute support keypoint prot first and then modulation
                avg_support_repres = average_representations2(support_repres, support_kp_mask)
                attn_feat, attn_map_l2 = self.att_modulators[i](avg_support_repres, q_feat_per_layer, support_kp_mask, fused_attention, output_attention_maps, is_norm=is_normalized, s_repres=support_repres)
                attentive_feats.append(attn_feat)
                attention_maps_l2.append(attn_map_l2)

                if use_auxiliary_kps:
                    avg_support_repres_aux = average_representations2(support_repres_aux, support_kp_mask_aux)
                    attn_feat_aux, attn_map_l2_aux = self.att_modulators[i](avg_support_repres_aux, q_feat_per_layer, support_kp_mask_aux, fused_attention, output_attention_maps=False, is_norm=is_normalized, s_repres=support_repres_aux)
                    attentive_feats_aux.append(attn_feat_aux)
                    attention_maps_l2_aux.append(attn_map_l2_aux)

            else:  # modulation first and then averaging attentive feats/attention maps
                attn_feat, attn_map_l2 = self.att_modulators[i](support_repres, q_feat_per_layer, support_kp_mask, fused_attention, output_attention_maps)
                attentive_feats.append(attn_feat)
                attention_maps_l2.append(attn_map_l2)

                if use_auxiliary_kps:
                    attn_feat_aux, attn_map_l2_aux = self.att_modulators[i](support_repres_aux, q_feat_per_layer, support_kp_mask_aux, fused_attention, output_attention_maps=False)
                    attentive_feats_aux.append(attn_feat_aux)
                    attention_maps_l2_aux.append(attn_map_l2_aux)

        return attentive_feats, attention_maps_l2, attentive_feats_aux, attention_maps_l2_aux

    def select_raw_feats_backup(self, s_feats_select, q_feats_select, support_feat, query_feat):
        '''
        input: B x C0 x h x w
        '''
        B1, C0, h, w = support_feat.shape
        B2 = query_feat.shape[0]

        if self.apply_post_LN == 2:  # apply LN for raw S & Q feats
            support_feat_ln = support_feat.flatten(2).transpose(2, 1)
            query_feat_ln = query_feat.flatten(2).transpose(2, 1)
            support_feat_ln = self.post_raw_LNs[0](support_feat_ln)
            query_feat_ln = self.post_raw_LNs[0](query_feat_ln)
            support_feat_ln = support_feat_ln.transpose(2, 1).reshape(B1, C0, h, w)
            query_feat_ln = query_feat_ln.transpose(2, 1).reshape(B2, C0, h, w)

        if self.att_modul_on_raw:
            if self.apply_post_LN == 0 or self.apply_post_LN == 1:
                s_feats_select.append(support_feat)
                q_feats_select.append(query_feat)
            elif self.apply_post_LN == 2:
                s_feats_select.append(support_feat_ln)
                q_feats_select.append(query_feat_ln)
            else:
                pass

        return s_feats_select, q_feats_select

    def forward_backup(self, support_feat, query_feat, support_saliency, query_saliency, support_im, query_im, salT_cfg
                ):

        B1, _, h, w = support_feat.shape  # (feature space)
        B2 = query_feat.shape[0]
        C = self.dim  # vit_dim may be different from raw_dim (namely support_feat's dim)
        Np = h * w

        # Use pre-conv1x1 to transform from raw_dim to vit_dim
        if self.use_pre_conv1x1:
            support_feat_tmp = self.pre_conv1x1(support_feat)
            query_feat_tmp = self.pre_conv1x1(query_feat)
            # B x C x h x w --> B x Np x C
            support_feat_tmp = support_feat_tmp.flatten(2).transpose(2, 1)
            query_feat_tmp = query_feat_tmp.flatten(2).transpose(2, 1)
        else:
            # B x C x h x w --> B x Np x C
            support_feat_tmp = support_feat.flatten(2).transpose(2, 1)
            query_feat_tmp = query_feat.flatten(2).transpose(2, 1)

        if self.has_CLS and self.num_salT_layers>=1:
            cls_tokens_s = self.cls_token.expand(B1, -1, -1)
            cls_tokens_q = self.cls_token.expand(B2, -1, -1)
            support_feat_tmp = torch.cat((cls_tokens_s, support_feat_tmp), dim=1)  # B1 x (1+Np) x C
            query_feat_tmp = torch.cat((cls_tokens_q, support_feat_tmp), dim=1)    # B2 x (1+Np) x C
        if self.num_salT_layers>=1:
            if self.has_pos_embed in ['APE1', 'APE2']:
                support_feat_tmp = support_feat_tmp + self.position_embeddings
                query_feat_tmp = query_feat_tmp + self.position_embeddings
                support_feat_tmp = self.pos_dropout(support_feat_tmp)
                query_feat_tmp = self.pos_dropout(query_feat_tmp)

        # -----------------------------------------------------
        # pow map learning
        if self.sal_mask_type == 'N':  # no mask
            s_sal_masks = None
            q_sal_masks = None
            s_morphology_func = None
            q_morphology_func = None
        elif self.sal_mask_type == 'P':  # pure mask
            # B x 1 x H x W (image space)
            assert support_saliency.shape[1] == 1 and query_saliency.shape[1] == 1, 'saliency channel should be 1'
            final_scale = self.sal_mask_cfg['src_size'][0] // self.sal_mask_cfg['dst_size'][0]
            # B x 1 x h (H//s) x w (W//s)
            s_sal = torch.nn.functional.avg_pool2d(support_saliency, kernel_size=final_scale, stride=final_scale, padding=0)
            q_sal = torch.nn.functional.avg_pool2d(query_saliency, kernel_size=final_scale, stride=final_scale, padding=0)
            s_sal_masks = [s_sal.reshape(B1, h * w)]  # list, 1 element: B x Np (equal h x w)
            q_sal_masks = [q_sal.reshape(B2, h * w)]  # list, 1 element: B x Np (equal h x w)
            s_morphology_func = None
            q_morphology_func = None
        elif self.sal_mask_type in ['S', 'IS', 'II']:
            s_sal_masks = None
            q_sal_masks = None
            # B x 1 x H x W (image space)
            assert support_saliency.shape[1] == 1 and query_saliency.shape[1] == 1, 'saliency channel should be 1'
            if self.sal_mask_cfg['in_ch'] == 1:  # only take saliency map as input to get saliency embedding
                s_mask_feats, s_low_res_masks = self.morphology_learner.compute_mask_feats(support_saliency)
                q_mask_feats, q_low_res_masks = self.morphology_learner.compute_mask_feats(query_saliency)
            elif self.sal_mask_cfg['in_ch'] == 4:  # take saliency map + RGB image as input get saliency embedding
                s_mask_feats, s_low_res_masks = self.morphology_learner.compute_mask_feats(support_saliency, support_im)
                q_mask_feats, q_low_res_masks = self.morphology_learner.compute_mask_feats(query_saliency, query_im)
            else:
                raise NotImplementedError
            s_morphology_func = partial(self.morphology_learner.forward, mask_feats=s_mask_feats, low_res_masks=s_low_res_masks)
            q_morphology_func = partial(self.morphology_learner.forward, mask_feats=q_mask_feats, low_res_masks=q_low_res_masks)



        # -----------------------------------------------------
        # cascaded saliency-guided transformer
        mask_mode = salT_cfg['mask_mode']
        func_mode = salT_cfg['func_mode']
        beta = salT_cfg['sal_beta']
        eps = salT_cfg['sal_eps']
        thresh = salT_cfg['thresh']
        use_max = salT_cfg['use_max']
        rbf_l2norm = salT_cfg['rbf_l2norm']
        rownorm = salT_cfg['rownorm']

        vit_s_feats, vit_s_attn_probs, s_vis_morph_masks, s_morph_reg_losses = self.salT(support_feat_tmp, self.sal_mask_type, s_sal_masks, s_morphology_func, \
                                        mask_mode, func_mode, beta, eps, thresh, use_max, rbf_l2norm, rownorm)
        vit_q_feats, vit_q_attn_probs, q_vis_morph_masks, q_morph_reg_losses = self.salT(query_feat_tmp, self.sal_mask_type, q_sal_masks, q_morphology_func, \
                                        mask_mode, func_mode, beta, eps, thresh, use_max, rbf_l2norm, rownorm)

        if self.has_CLS and self.num_salT_layers>=1:
            for i in range(len(self.att_modulators)):
                vit_s_feats[i] = vit_s_feats[i][:, 1:]  # exclude CLS, B x (1+Np) x C
                vit_q_feats[i] = vit_q_feats[i][:, 1:]

        # unify the morphology regularization loss for support & query masks
        if (self.sal_mask_type in ['S', 'IS', 'II']) and self.morphology_learner.impose_reg>0:
            loss_offset = self.sal_mask_cfg['reg_loss_offset']  # 0., 0.0625
            # s_morph_reg_losses = torch.clamp(s_morph_reg_losses - loss_offset, min=0)
            # q_morph_reg_losses = torch.clamp(q_morph_reg_losses - loss_offset, min=0)
            s_morph_reg_losses = torch.stack(s_morph_reg_losses)  # num_ViTs x B1
            q_morph_reg_losses = torch.stack(q_morph_reg_losses)  # num_ViTs x B2
            s_morph_reg_loss_t = (s_morph_reg_losses-loss_offset).clamp(min=0).mean(dim=0)  # B1
            q_morph_reg_loss_t = (q_morph_reg_losses-loss_offset).clamp(min=0).mean(dim=0)  # B2
            morph_reg_N = s_morph_reg_loss_t.shape[0] + q_morph_reg_loss_t.shape[0]
            morph_reg_loss = (s_morph_reg_loss_t.sum() + q_morph_reg_loss_t.sum()) / morph_reg_N

            s_morph_reg_loss_o = (s_morph_reg_losses).mean(dim=0)  # B1
            q_morph_reg_loss_o = (q_morph_reg_losses).mean(dim=0)  # B2
            morph_reg_loss_o = (s_morph_reg_loss_o.sum() + q_morph_reg_loss_o.sum()) / morph_reg_N
        else:
            morph_reg_loss = 0
            morph_reg_loss_o = 0

        # -----------------------------------------------------
        # feature post LN & feature selection
        s_feats_select = []
        q_feats_select = []
        # assert len(self.att_modulators) >= 1, 'num of att_modulators should >= 1, CascadeSalTransformer'
        s_feats_select, q_feats_select = self.select_raw_feats_backup(s_feats_select, q_feats_select, support_feat, query_feat)
        if self.att_modul_type == 0:  # 0, all vit feats are taken;
            if self.num_salT_layers == 0:  # no vits, self.att_modul_on_raw must be true
                pass
            else:
                if self.apply_post_LN == 1 or self.apply_post_LN == 2:  # apply LN for vit feats
                    for i in range(0, len(vit_s_feats)):
                        vit_s_feats[i] = self.post_vit_LNs[i](vit_s_feats[i])
                        vit_q_feats[i] = self.post_vit_LNs[i](vit_q_feats[i])
                else:
                    pass

                # reshape
                # B x Np x C --> B x C x h x w
                # for i in range(len(vit_s_feats)):
                #     vit_s_feats[i] = vit_s_feats[i].transpose(2, 1).reshape(B1, C, h, w).contiguous()
                #     vit_q_feats[i] = vit_q_feats[i].transpose(2, 1).reshape(B2, C, h, w).contiguous()

                if self.vit_fused_feat_dim > 0:
                    s_fused_feat = self.apply_conv1x1(vit_s_feats, self.vit_fusion_process)
                    q_fused_feat = self.apply_conv1x1(vit_q_feats, self.vit_fusion_process)
                else:
                    if self.vit_fusion_process == 'cat':
                        s_fused_feat = torch.cat(vit_s_feats, dim=2)
                        q_fused_feat = torch.cat(vit_q_feats, dim=2)
                    elif self.vit_fusion_process == 'add':
                        s_fused_feat = sum(vit_s_feats)
                        q_fused_feat = sum(vit_q_feats)

                # 'add' fusion needs LN to scale feature
                # if self.num_salT_layers >= 2 and self.vit_fusion_process == 'add':
                #     s_fused_feat = self.feat_fusion_LN(s_fused_feat)
                #     q_fused_feat = self.feat_fusion_LN(q_fused_feat)

                # reshape
                # B x Np x C --> B x C x h x w
                s_fused_feat = s_fused_feat.transpose(2, 1).reshape(B1, -1, h, w).contiguous()
                q_fused_feat = q_fused_feat.transpose(2, 1).reshape(B2, -1, h, w).contiguous()

                s_feats_select.append(s_fused_feat)
                q_feats_select.append(q_fused_feat)

        elif self.att_modul_type == 1:  # last ViT feat is taken
            if self.num_salT_layers == 0:  # no vits, self.att_modul_on_raw must be true
                pass
            else:
                vit_s_feats_rear = vit_s_feats[-1]
                vit_q_feats_rear = vit_q_feats[-1]

                if self.apply_post_LN == 1 or self.apply_post_LN == 2:  # apply LN for vit feats
                    vit_s_feats_rear = self.post_vit_LNs[0](vit_s_feats_rear)
                    vit_q_feats_rear = self.post_vit_LNs[0](vit_q_feats_rear)

                # # reshape
                # # B x Np x C --> B x C x h x w
                # vit_s_feats_rear = vit_s_feats_rear.transpose(2, 1).reshape(B1, C, h, w).contiguous()
                # vit_q_feats_rear = vit_q_feats_rear.transpose(2, 1).reshape(B2, C, h, w).contiguous()

                if self.vit_fused_feat_dim > 0:
                    vit_s_feats_rear = self.apply_conv1x1([vit_s_feats_rear], self.vit_fusion_process)
                    vit_q_feats_rear = self.apply_conv1x1([vit_q_feats_rear], self.vit_fusion_process)

                # reshape
                # B x Np x C --> B x C x h x w
                vit_s_feats_rear = vit_s_feats_rear.transpose(2, 1).reshape(B1, -1, h, w).contiguous()
                vit_q_feats_rear = vit_q_feats_rear.transpose(2, 1).reshape(B2, -1, h, w).contiguous()

                s_feats_select.append(vit_s_feats_rear)
                q_feats_select.append(vit_q_feats_rear)

        elif self.att_modul_type == 2:  # last and middle vit feats are taken
            # when att_modul_type == 2, there are at least 2 vits

            middle_ind, last_ind = self.vit_layer_index
            s_vit_feats_select = [vit_s_feats[middle_ind], vit_s_feats[last_ind]]
            q_vit_feats_select = [vit_q_feats[middle_ind], vit_q_feats[last_ind]]

            if self.apply_post_LN == 1 or self.apply_post_LN == 2:  # apply LN for vit feats
                for i in range(0, len(s_vit_feats_select)):
                    s_vit_feats_select[i] = self.post_vit_LNs[i](s_vit_feats_select[i])
                    q_vit_feats_select[i] = self.post_vit_LNs[i](q_vit_feats_select[i])

            # reshape
            # B x Np x C --> B x C x h x w
            # for i in range(len(s_vit_feats_select)):
            #     s_vit_feats_select[i] = s_vit_feats_select[i].transpose(2, 1).reshape(B1, C, h, w).contiguous()
            #     q_vit_feats_select[i] = q_vit_feats_select[i].transpose(2, 1).reshape(B2, C, h, w).contiguous()

            if self.vit_fused_feat_dim > 0:
                s_fused_feat = self.apply_conv1x1(s_vit_feats_select, self.vit_fusion_process)
                q_fused_feat = self.apply_conv1x1(q_vit_feats_select, self.vit_fusion_process)
            else:
                if self.vit_fusion_process == 'cat':
                    s_fused_feat = torch.cat(s_vit_feats_select, dim=2)
                    q_fused_feat = torch.cat(q_vit_feats_select, dim=2)
                elif self.vit_fusion_process == 'add':
                    s_fused_feat = sum(s_vit_feats_select)
                    q_fused_feat = sum(q_vit_feats_select)

            # 'add' fusion needs LN to scale feature
            # if self.num_salT_layers >= 2 and self.vit_fusion_process == 'add':
            #     s_fused_feat = self.feat_fusion_LN(s_fused_feat)
            #     q_fused_feat = self.feat_fusion_LN(q_fused_feat)

            # reshape
            # B x Np x C --> B x C x h x w
            s_fused_feat = s_fused_feat.transpose(2, 1).reshape(B1, -1, h, w).contiguous()
            q_fused_feat = q_fused_feat.transpose(2, 1).reshape(B2, -1, h, w).contiguous()

            s_feats_select.append(s_fused_feat)
            q_feats_select.append(q_fused_feat)

        else:
            pass

        # Basically we can fuse all features and just use ONE modulator, as the fiber extraction and modulation
        # operations are linear
        fuse_all_feats = True
        if fuse_all_feats:
            s_feats_select = [torch.cat(s_feats_select, dim=1)]
            q_feats_select = [torch.cat(q_feats_select, dim=1)]


        return s_feats_select, q_feats_select, vit_s_attn_probs, vit_q_attn_probs, s_vis_morph_masks, q_vis_morph_masks, morph_reg_loss, morph_reg_loss_o