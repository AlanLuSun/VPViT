# -*- coding: utf-8 -*-
import copy

import torch
from torch import nn
from torch.nn.modules.utils import _pair
from utils import draw_contours
import numpy as np
import math
from heatmap import putGaussianMaps

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out, cfg1=(3, 1, 1), cfg2=(3, 1, 1)):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=cfg1[0],stride=cfg1[1],padding=cfg1[2],bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=cfg2[0],stride=cfg2[1],padding=cfg2[2],bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block_ln(nn.Module):
    def __init__(self,ch_in,ch_out, cfg1=(3, 1, 1), cfg2=(3, 1, 1)):
        super(conv_block_ln,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=cfg1[0],stride=cfg1[1],padding=cfg1[2],bias=True),
            nn.LayerNorm(ch_out, eps=1e-6),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=cfg2[0],stride=cfg2[1],padding=cfg2[2],bias=True),
            nn.LayerNorm(ch_out, eps=1e-6),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block_nonorm(nn.Module):
    def __init__(self,ch_in,ch_out, cfg1=(3, 1, 1), cfg2=(3, 1, 1)):
        super(conv_block_nonorm,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=cfg1[0],stride=cfg1[1],padding=cfg1[2],bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=cfg2[0],stride=cfg2[1],padding=cfg2[2],bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, up_scale=2, cfg=(3,1,1)):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=up_scale),
            nn.Conv2d(ch_in,ch_out,kernel_size=cfg[0],stride=cfg[1],padding=cfg[2],bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class up_conv_ln(nn.Module):
    def __init__(self,ch_in,ch_out, up_scale=2, cfg=(3,1,1)):
        super(up_conv_ln,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=up_scale),
            nn.Conv2d(ch_in,ch_out,kernel_size=cfg[0],stride=cfg[1],padding=cfg[2],bias=True),
		    nn.LayerNorm(ch_out, eps=1e-6),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class up_conv_nonorm(nn.Module):
    def __init__(self,ch_in,ch_out, up_scale=2, cfg=(3,1,1)):
        super(up_conv_nonorm,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=up_scale),
            nn.Conv2d(ch_in,ch_out,kernel_size=cfg[0],stride=cfg[1],padding=cfg[2],bias=True),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class U_Net(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class UNetTiny(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, channels=(32, 64, 128, 256), downscales=(1, 2, 4, 8), norm_type='bn'):
        '''
        U-Net tiny version has THREE depths, where the channels are (32, 64, 128, 256) and
        the resolutions are (1, 2, 4, 8), by default.
        norm_type: 'bn', 'ln', 'no' (no norm)
        '''
        super(UNetTiny, self).__init__()

        # self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        maxpool_list = []
        ratio_list = []  # [2, 2, 2]
        for i in range(len(downscales) - 1):
            scale = int(downscales[i+1] / downscales[i])
            ratio_list.append(scale)
            maxpool_list.append(nn.MaxPool2d(kernel_size=scale, stride=scale))
        self.Maxpools = nn.Sequential(*maxpool_list)

        if norm_type == 'bn':
            conv_block_t = eval('conv_block')
            up_conv_t = eval('up_conv')
        elif norm_type == 'ln':
            conv_block_t = eval('conv_block_ln')
            up_conv_t = eval('up_conv_ln')
        else:  # no norm
            conv_block_t = eval('conv_block_nonorm')
            up_conv_t = eval('up_conv_nonorm')

        self.Conv1 = conv_block_t(ch_in=img_ch, ch_out=channels[0])  # img_ch-->32
        self.Conv2 = conv_block_t(ch_in=channels[0], ch_out=channels[1])  # 32-->64
        self.Conv3 = conv_block_t(ch_in=channels[1], ch_out=channels[2])  # 64-->128
        self.Conv4 = conv_block_t(ch_in=channels[2], ch_out=channels[3])  # 128-->256
        # self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        # self.Up5 = up_conv(ch_in=1024, ch_out=512)
        # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv_t(ch_in=channels[3], ch_out=channels[2], up_scale=ratio_list[-1])
        self.Up_conv4 = conv_block_t(ch_in=channels[3], ch_out=channels[2])

        self.Up3 = up_conv_t(ch_in=channels[2], ch_out=channels[1], up_scale=ratio_list[-2])
        self.Up_conv3 = conv_block_t(ch_in=channels[2], ch_out=channels[1])

        self.Up2 = up_conv_t(ch_in=channels[1], ch_out=channels[0], up_scale=ratio_list[-3])
        self.Up_conv2 = conv_block_t(ch_in=channels[1], ch_out=channels[0])

        self.Conv_1x1 = nn.Conv2d(channels[0], output_ch, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpools[0](x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpools[1](x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpools[2](x3)
        x4 = self.Conv4(x4)

        # x5 = self.Maxpool(x4)
        # x5 = self.Conv5(x5)

        # decoding + concat path
        # d5 = self.Up5(x5)
        # d5 = torch.cat((x4, d5), dim=1)
        # d5 = self.Up_conv5(d5)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

class ConvEmbedding(nn.Module):
    def __init__(self, in_channel=1, patch_size=32, embedding_dim=256, hidden_layers=1, out_channel=1, norm_type='no'):
        '''
        Using strided conv to perform embedding for each patch.
        strided_conv --> conv3x3 (hidden_layers) --> conv1x1
        norm_type: 'bn', 'ln', 'no' (no norm)
        '''

        super(ConvEmbedding, self).__init__()

        self.projection = nn.Conv2d(
                in_channel,
                embedding_dim,
                kernel_size=patch_size,
                stride=patch_size,
        )
        if norm_type == 'bn':
            self.norm_t = nn.BatchNorm2d(embedding_dim, eps=1e-6)
        elif norm_type == 'ln':
            self.norm_t = nn.LayerNorm(embedding_dim, eps=1e-6)
        else:  # no norm
            self.norm_t = None
        self.relu = nn.LeakyReLU(inplace=True)

        self.hidden_convs = nn.Sequential()
        for i in range(hidden_layers):
            self.hidden_convs.add_module('conv%d'%i, nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, stride=1, padding=1))
            if norm_type == 'bn':
                self.hidden_convs.add_module('norm%d'%i, nn.BatchNorm2d(embedding_dim, eps=1e-6))
            elif norm_type == 'ln':
                self.hidden_convs.add_module('norm%d'%i, nn.LayerNorm(embedding_dim, eps=1e-6))
            else:  # no norm
                pass
            self.hidden_convs.add_module('relu%d'%i, nn.LeakyReLU(inplace=True))


        self.conv_1x1 = nn.Conv2d(embedding_dim, out_channel, kernel_size=1, stride=1, padding=0)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # B x in_ch x H x W --> B x out_ch x h x w (h=H/patch_size)
        x = self.projection(x)
        if self.norm_t is not None:
            x = self.norm_t(x)
        x = self.relu(x)
        x = self.hidden_convs(x)
        x = self.conv_1x1(x)

        return x

class PowLearner(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, pow_type='hete-global', bound=0.1,
                 unet_cfg=((32, 64, 128, 256), (1, 2, 4, 8)),
                 patchnet_cfg=(32, 256, 1),
                 homo_cfg=(12, 12),
                 norm_type='bn'):
        '''
        pow_type: 'hete-global' (unet_cfg), 'hete-patch' (patchnet_cfg), 'homo-global'
        bound: (bound~1/bound) for pow, 0<bound<1
        norm_type: 'bn', 'ln', 'no' (no norm)
        '''
        super(PowLearner, self).__init__()

        self.bound = bound
        self.pow_type = pow_type

        if pow_type == 'hete-global':
            # learn pow map conditioned on input saliency
            # encoder-decoder based architecture, same size input and same size output
            channels = unet_cfg[0]
            downscales = unet_cfg[1]
            self.net = UNetTiny(in_ch, out_ch, channels, downscales, norm_type)
        elif pow_type == 'hete-patch':
            # learn pow map conditioned on input saliency
            # patch embedding based, namely projecting each patch using strided conv
            self.patch_size = patchnet_cfg[0]
            embedding_dim = patchnet_cfg[1]
            hidden_layers = patchnet_cfg[2]
            self.net = ConvEmbedding(in_ch, self.patch_size, embedding_dim, hidden_layers, out_ch, norm_type)
        elif pow_type == 'homo-global':
            h = homo_cfg[0]
            w = homo_cfg[1]
            # learn pow map globally
            weight = torch.FloatTensor(1, 1, h, w).fill_(0.0)  # tensor size: n_pixels
            self.pow_params = nn.Parameter(weight, requires_grad=True)
        else:
            print('Error for pow_type in PowLearner!')
            exit(0)

    def forward(self, x):
        '''
        x: B x C x H x W, and x's value should range from 0~1
        note: (bound~1/bound) for pow, 0<bound<1
        '''
        if self.pow_type == 'hete-global':
            H = x #+ 1e-9  # add a small number to avoid 0^R as its gradient may be inf.
            x = self.net(x)  # B x C x H x W
            x = torch.tanh(x)
            R = torch.pow(1 / self.bound, x)  # B x C x H x W
            # print(R[0,0,0, 0])

            H = torch.pow(H, R)  # B x C x H x W
        elif self.pow_type == 'hete-patch':
            H = x #+ 1e-9  # add a small number to avoid 0^R as its gradient may be inf.
            x = self.net(x)  # B x C x (H/patch_size) x (W/patch_size)
            x = torch.tanh(x)
            R = torch.pow(1 / self.bound, x)  # B x C x (H/patch_size) x (W/patch_size)
            # print(R)

            # B x C x H x W, here the 'nearest' upsampling is very important!!!
            # [[1, 2],   --> [[1, 1, 2, 2],
            #  [3, 4]]        [1, 1, 2, 2],
            #                 [3, 3, 4, 4],
            #                 [3, 3, 4, 4]]
            R = nn.functional.interpolate(R, scale_factor=self.patch_size, mode='nearest')
            H = torch.pow(H, R)  # B x C x H x W
        elif self.pow_type == 'homo-global':
            H = x #+ 1e-9  # add a small number to avoid 0^R as its gradient may be inf.
            exponent = torch.tanh(self.pow_params)
            R = torch.pow(1 / self.bound, exponent)  # 1 x 1 x H x W

            H = torch.pow(H, R)  # B x C x H x W
        else:
            print('Error for pow_type in PowLearner!')
            exit(0)

        return H

class SalMorphologyLearner(nn.Module):
    def __init__(
            self,
            in_ch=1, out_ch=1,
            src_size=(384, 384),  # mask size
            dst_size=(12, 12),  # target grid size
            pre_downscale=32,  # downsize mask by this ratio (>=2); if <= 1, it means no using of pre-downscale
            num_learner=1,  # the number of learners
            pow_type='hete-global',  # 'hete-global' (unet_cfg), 'hete-patch' (patchnet_cfg), 'homo-global'
            bound=0.1,             # (bound~1/bound) for pow, 0<bound<1
            unet_cfg=((32, 64, 128, 256), (1, 2, 4, 8)),  # (channels_tuple, downscales_tuple)
            patchnet_cfg=(32, 256, 1),  # (patch_size, embedding_dim, hidden_layers)
            norm_type='bn'  # 'bn', 'ln', 'no' (no norm)
    ):
        super(SalMorphologyLearner, self).__init__()

        self.pre_downscale = pre_downscale
        if pre_downscale >= 2:  # < 2 (namely 1 or 0), it means no usage of pre_downscale
            self.pre_downscale_layer = nn.AvgPool2d(kernel_size=pre_downscale, stride=pre_downscale)

        h_scale = src_size[0] // dst_size[0]
        w_scale = src_size[1] // dst_size[1]
        assert (h_scale == w_scale), 'the h_scale should be equal to w_scale.'
        scale = h_scale
        assert (pre_downscale <= scale), 'pre_downscale is too big.'
        if pre_downscale == scale:  # no need to perform post-downscale
            self.post_downscale = 0
        elif pre_downscale < 2:  # no pre-downscale, we need full scale for post processing
            self.post_downscale = scale
        else:
            self.post_downscale = scale // pre_downscale

        model_list = []
        self.num_learner = num_learner
        for _ in range(num_learner):
            if pow_type == 'hete-global':
                model_list.append(
                    PowLearner(in_ch, out_ch, pow_type, bound,
                               unet_cfg=unet_cfg, patchnet_cfg=None, homo_cfg=None,
                               norm_type=norm_type)
                )
            elif pow_type == 'hete-patch':
                patch_size = patchnet_cfg[0]
                if pre_downscale < 2:
                    assert patch_size <= src_size[0]
                elif pre_downscale >= 2:
                    assert patch_size <= src_size[0] // pre_downscale
                model_list.append(
                    PowLearner(in_ch, out_ch, pow_type, bound,
                               unet_cfg=None, patchnet_cfg=patchnet_cfg, homo_cfg=None,
                               norm_type=norm_type)
                )
            elif pow_type == 'homo-global':
                if pre_downscale < 2:  # no pre_downscale
                    homo_cfg = src_size
                else:
                    homo_cfg = (src_size[0] // pre_downscale, src_size[1] // pre_downscale)
                model_list.append(
                    PowLearner(in_ch, out_ch, pow_type, bound,
                               unet_cfg=None, patchnet_cfg=None, homo_cfg=homo_cfg,
                               norm_type=norm_type)
                )
            else:
                print('Error for pow_type in SalMorphologyLearner!')
                exit(0)
        self.learners = nn.Sequential(*model_list)

        if self.post_downscale >= 2:
            self.post_downscale_layer = nn.AvgPool2d(kernel_size=self.post_downscale, stride=self.post_downscale)

    def forward(self, mask, vis=False):
        '''
        input : mask: B x in_ch x H x W
        output: final_masks, vis_pow_masks, each is a list containing a tensor
        which has size of B x out_ch x h x w
        '''
        if self.pre_downscale >= 2:
            mask = self.pre_downscale_layer(mask)
        # set mask value range; avoid 0^R as its gradient may be inf.
        mask_clamp = torch.clamp(mask, min=0.05, max=0.95)

        final_masks = []
        vis_pow_masks = []
        for i in range(self.num_learner):
            m = self.learners[i](mask_clamp)
            if vis == True:
                # vis_pow_masks.append(m)
                pow_tmp = (m * 255).long()
                origin_tmp = (mask_clamp * 255).long()
                m_draws = self.draw_contours_batch(m=origin_tmp, im=pow_tmp, thresh=int(0.4*255))
                vis_pow_masks.append(m_draws)

            if self.post_downscale >= 2:
                m = self.post_downscale_layer(m)
            final_masks.append(m)

        if self.num_learner == 0:  # directly output pooled masks if no pow learners
            final_masks.append(mask)

        return final_masks, vis_pow_masks

    def draw_contours_batch(self, m, im, thresh=int(0.1*255)):
        '''
        m: B x 1 x H x W
        im: B x C x H x W
        mask or im's value range: 0~255
        '''
        B, _, H, W = m.shape
        m = m.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()
        im = im.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()  # B x H x W x C
        out_list = []
        for i in range(B):
            out = draw_contours(thresh, mask=m[i], im=im[i])
            out_list.append(out)

        outs = np.stack(out_list)
        outs = torch.tensor(outs).permute(0, 3, 1, 2)

        return outs


class ParamGenerator(nn.Module):
    def __init__(self, in_ch, out_ch, hidden_ch, fc_layers=1):
        '''
        fc_layers >= 2
        '''
        super(ParamGenerator, self).__init__()

        fc_list = []
        dims = [in_ch] + [hidden_ch] * (fc_layers - 1) + [out_ch]
        pairs = zip(dims[0:-1:1], dims[1::1])
        for i, pair in enumerate(pairs):
            fc_list.append(nn.Linear(in_features=pair[0], out_features=pair[1], bias=True))
            if i == (fc_layers-1):  # the last linear layer, no relu so that the output has posotive/negative values
                pass
            else:
                # fc_list.append(nn.ReLU(True))
                fc_list.append(nn.GELU())

        self.net = nn.Sequential(*fc_list)

        self.apply(self._init_weights)
        # for i in range(fc_layers):
        #     nn.init.xavier_uniform_(self.net[2*i].weight, gain=1.0)
        #     if i == (fc_layers - 1):
        #         self.net[2 * i].bias.data.fill_(1.0)
        #     else:
        #         self.net[2 * i].bias.data.zero_()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # trunc_normal_(m.weight, std=.02)
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        '''
        B x Np x C
        B x out_ch (the generated params are a vector with out_ch dim)
        '''
        # global averaging
        x = x.mean(dim=1)  # B x C
        # non-linear mapping
        x = self.net(x)

        return x


class PowLearner2(nn.Module):
    def __init__(self, in_ch, hidden_ch, fc_layers=1, out_ch=1, regress_type=0, bound=0.1, T=1.0, impose_reg=0, pn_center=1.0, fixed_MC=0, conv_cfg=(0, 3, 0)):
        '''
        out_ch: if out_ch = 1, it means we use only one R to perform global power;
                if out_ch = h*w, namely the number of patches, it means we use per patch power
        regress_type:
        bound:
        T (temperature):
        impose_reg: whether impose regularization loss over SE weights
        conv_cfg: (num_layers, kernel_size, conv_reg_type)=(1, 3, 0)
        '''
        super(PowLearner2, self).__init__()

        self.out_ch = out_ch
        self.regress_type = regress_type
        self.bound = bound
        self.T = T  # temperature
        self.impose_reg = impose_reg
        self.pn_center = float(pn_center)  # the power normalization center for regularization, e.g., (R- center)**2.
        self.fixed_MC = fixed_MC

        # morphology parameter generator (MPG)
        self.mpg = ParamGenerator(in_ch=in_ch, out_ch=out_ch, hidden_ch=hidden_ch, fc_layers=fc_layers)

        # conv k x k to refine local morphology
        conv_list = []
        self.num_layers = conv_cfg[0]  # if num_layers=0, we don't use conv
        kernel_size = conv_cfg[1]
        self.conv_reg_type = conv_cfg[2]
        if self.num_layers > 0:
            assert kernel_size > 0, "Kernel size should > 0 in PowLearner2."
            for i in range(self.num_layers):
                conv_list.append(nn.Conv2d(1, 1, kernel_size, stride=1, padding=kernel_size//2, bias=True))
                if i != self.num_layers-1:  # not last layer
                    conv_list.append(nn.ReLU(True))
            self.conv= nn.Sequential(*conv_list)
            # initialize the conv weights/bias
            for i in range(self.num_layers):
                self.conv[2*i].weight.data.fill_(1.0/(kernel_size**2))
                self.conv[2*i].bias.data.zero_()

    def forward(self, mask_feats, patch_feats, low_res_masks):
        '''
        mask_feats: B x Np x d
        patch_feats: B x Np x C
        low_res_masks: B x mask_channel x h' x w', usually mask_channel is 1.

        output: transformed low_res_mask, and regularization loss
        '''
        B = mask_feats.shape[0]
        out_ch = self.out_ch
        if self.fixed_MC == 0:  # learnable pow; don't fix pow
            x = torch.cat((mask_feats, patch_feats), dim=2)  # B x Np x (d+C)
            x = self.mpg(x)  # B x out_ch

            _, _, h, w = low_res_masks.shape
            if out_ch == 1:
                x = x.reshape(B, 1, 1, 1)
            elif out_ch == h * w:
                x = x.reshape(B, 1, h, w)
            else:
                print('Error for the shape of x, in PowLearner2.')
                exit(0)

            # morphology changing module (MCM)
            # 0) v = tanh(Tx); R = a^v (a = 1/bound)
            if self.regress_type == 0:
                x = torch.tanh(self.T*x)  # 1.0, 0.5, 0.2
                R = torch.pow(1 / self.bound, x)

            # 1) R = sigmoid(Tx) * a (a = bound)
            elif self.regress_type == 1:
                R = torch.sigmoid(self.T * x) * self.bound

            # 2) R = exp(Tx)
            elif self.regress_type == 2:
                R = torch.exp(self.T * x)

            # 3) R = min(max(x, low), high)  (low=bound, high=1/bound)
            elif self.regress_type == 3:
                R = torch.clamp(x, min=self.bound, max=1 / self.bound)

            else:
                raise NotImplementedError


            H = torch.pow(low_res_masks, R)
            # print('PowLearner2 R: ', R)
        else:  # manually fixed pow
            x = torch.zeros(B, out_ch).fill_(self.fixed_MC)
            if torch.cuda.is_available():
                x = x.cuda()

            _, _, h, w = low_res_masks.shape
            if out_ch == 1:
                x = x.reshape(B, 1, 1, 1)
            elif out_ch == h * w:
                x = x.reshape(B, 1, h, w)
            else:
                print('Error for the shape of x, in PowLearner2.')
                exit(0)
            R = x
            H = torch.pow(low_res_masks, R)

        # conv k x k to refine local morphology
        if self.num_layers > 0:
            H2 = self.conv(H)
            if self.conv_reg_type == 0:
                H2 = torch.clamp(H2, min=0, max=1.0)
            elif self.conv_reg_type == 1:
                H2 = torch.sigmoid(H2)
        else:
            H2 = H
        
        if self.impose_reg == 0:  # no reg_loss
            loss_reg = torch.tensor(0.)
        else:
            if self.impose_reg == 1:  # || M'-M||^2, consistency
                loss_reg = (H2 - low_res_masks)**2
            elif self.impose_reg == 2:  # || M'-M||, consistency
                loss_reg = torch.abs(H2 - low_res_masks)
            elif self.impose_reg == 3:  # || 1 - R ||^2, regularization loss for learned exponent R
                loss_reg = (self.pn_center - R) ** 2
            elif self.impose_reg == 4:  # || 1 - R ||
                loss_reg = torch.abs(self.pn_center - R)
            elif self.impose_reg == 5:  # ||1 - 1/R||^2
                loss_reg = (self.pn_center - 1/R) ** 2
            elif self.impose_reg == 6:  # ||1 - 1/R||
                loss_reg = torch.abs(self.pn_center - 1/R)
            elif self.impose_reg == 7:  # ||(2 - R - 1/R)/2||^2, hook function
                loss_reg = ((2 - 1/R - R) * 0.5) ** 2
            elif self.impose_reg == 8:  # ||(2 - R - 1/R)/2||, hook function
                loss_reg = torch.abs((2 - 1/R - R) * 0.5)
            elif self.impose_reg == 9:  # (|| 1 - R ||^2 + ||1 - 1/R||^2)/2
                loss_reg = ((1 - R) ** 2 + (1 - 1/R) ** 2) / 2
            elif self.impose_reg == 10:  # (|| 1 - R || + ||1 - 1/R||)/2
                loss_reg = (torch.abs(1 - R) + torch.abs(1 - 1/R)) / 2
            else:
                print('Error for impose_reg in MorphologyNeuralNet.')
                exit(0)
            loss_reg = loss_reg.reshape(B, -1).mean(dim=1)  # B

        return H2, loss_reg, H


class DilationLayer(nn.Module):
    def __init__(self, in_ch=1, kernel_size=3, stride=1, padding=1, bias=False, apply_reg=False):
        '''
        Layer-wise dilation. Thus the output channel is equal to input channel.
        This dilation layer has learnable SE, max({x_i + w_i}) + bias
        Optional to use apply_reg, namely regularization loss over weights
        '''
        super(DilationLayer, self).__init__()

        self.in_ch = in_ch
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.apply_reg = apply_reg  # regularization loss
        self.weight = nn.Parameter(torch.Tensor(self.in_ch, *self.kernel_size))  # C x kh x kw
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.in_ch))  # C
        else:
            self.register_parameter('bias', None)
        # initialize SE
        self.reset_parameters()

    def reset_parameters(self):
        # n = self.in_ch * self.kernel_size[0] * self.kernel_size[1]
        # stdv = 1. / math.sqrt(n)
        self.weight.data.fill_(1.0)  # flat square SE with init value 1.0
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        '''
        x: B x C x H x W
        '''
        B, C, H, W = x.shape
        ks_prod = self.kernel_size[0] * self.kernel_size[1]
        # B x C x H x W --> B x (C*kh*kw) x L
        x_unfold = nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        L = x_unfold.shape[-1]  # L sliding windows in image space
        y_unfold = (x_unfold.view(B, C, ks_prod, L) + self.weight.view(1, C, ks_prod, 1)).max(dim=2)[0]  # B x C x L
        if self.bias is not None:
            y_unfold = y_unfold + self.bias.view(1, C, 1)  # B x C x L
        H2 = math.floor((H + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        W2 = math.floor((W + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)
        assert H2 * W2 == L, 'Output size is wrongly computed in DilationLayer'

        # B x (C*kh2*kw2) x L --> B x C x H_out x W_out; "unfold" needs to re-computed and is independent to "fold"
        y = nn.functional.fold(y_unfold, output_size=(H2, W2), kernel_size=(1, 1), stride=1)  # B x C x H2 x W2

        return y

class ErosionLayer(nn.Module):
    def __init__(self, in_ch=1, kernel_size=3, stride=1, padding=1, bias=False, apply_reg=False):
        '''
        Layer-wise erosion. Thus the output channel is equal to input channel.
        This erosion layer has learnable SE, min({x_i - w_i}) + bias
        Optional to use apply_reg, namely regularization loss over weights
        '''
        super(ErosionLayer, self).__init__()

        self.in_ch = in_ch
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.apply_reg = apply_reg  # regularization loss
        self.weight = nn.Parameter(torch.Tensor(self.in_ch, *self.kernel_size))  # C x kh x kw
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.in_ch))  # C
        else:
            self.register_parameter('bias', None)
        # initialize SE
        self.reset_parameters()

    def reset_parameters(self):
        # n = self.in_ch * self.kernel_size[0] * self.kernel_size[1]
        # stdv = 1. / math.sqrt(n)
        self.weight.data.fill_(1.0)  # flat square SE with init value 1.0
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        '''
        x: B x C x H x W
        '''
        B, C, H, W = x.shape
        ks_prod = self.kernel_size[0] * self.kernel_size[1]
        # B x C x H x W --> B x (C*kh*kw) x L
        x_unfold = nn.functional.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
        L = x_unfold.shape[-1]  # L sliding windows in image space
        y_unfold = (x_unfold.view(B, C, ks_prod, L) - self.weight.view(1, C, ks_prod, 1)).min(dim=2)[0]  # B x C x L
        if self.bias is not None:
            y_unfold = y_unfold + self.bias.view(1, C, 1)  # B x C x L
        H2 = math.floor((H + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1)
        W2 = math.floor((W + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1)
        assert H2 * W2 == L, 'Output size is wrongly computed in DilationLayer'

        # B x (C*kh2*kw2) x L --> B x C x H_out x W_out; "unfold" needs to re-computed and is independent to "fold"
        y = nn.functional.fold(y_unfold, output_size=(H2, W2), kernel_size=(1, 1), stride=1)  # B x C x H2 x W2

        return y

class DilationLayerExtSE(nn.Module):
    def __init__(self):
        '''
        Layer-wise dilation . Thus the output channel is equal to input channel.
        This dilation layer has external SE, max({x_i + w_i}) + bias
        Optional to use apply_reg, namely regularization loss over weights
        '''
        super(DilationLayerExtSE, self).__init__()

    def forward(self, x, weight, padding=0, stride=1, bias=None):
        '''
        Layer-wise dilation.
        x: B x C x H x W, a batch of images
        weight: B x C x kh x kw or C x kh x kw, a batch of SE filters
        bias: B x C or C
        '''
        padding = _pair(padding)
        stride = _pair(stride)

        B, C, H, W = x.shape
        kh, kw = weight.shape[-2:]
        ks_prod = kh * kw
        # B x C x H x W --> B x (C*kh*kw) x L
        x_unfold = nn.functional.unfold(x, kernel_size=(kh, kw), padding=padding, stride=stride)
        L = x_unfold.shape[-1]  # L sliding windows in image space
        y_unfold = (x_unfold.view(B, C, ks_prod, L) + weight.view(-1, C, ks_prod, 1)).max(dim=2)[0]  # B x C x L
        if bias is not None:
            y_unfold = y_unfold + bias.view(-1, C, 1)  # B x C x L
        H2 = math.floor((H + 2 * padding[0] - kh) / stride[0] + 1)
        W2 = math.floor((W + 2 * padding[1] - kw) / stride[1] + 1)
        assert H2 * W2 == L, 'Output size is wrongly computed in DilationLayer'

        # B x (C*kh2*kw2) x L --> B x C x H_out x W_out; "unfold" needs to re-computed and is independent to "fold"
        y = nn.functional.fold(y_unfold, output_size=(H2, W2), kernel_size=(1, 1), stride=1)  # B x C x H2 x W2

        return y

class ErosionLayerExtSE(nn.Module):
    def __init__(self):
        '''
        Layer-wise erosion. Thus the output channel is equal to input channel.
        This erosion layer has external SE, min({x_i - w_i}) + bias
        Optional to use apply_reg, namely regularization loss over weights
        '''
        super(ErosionLayerExtSE, self).__init__()

    def forward(self, x, weight, padding=0, stride=1, bias=None):
        '''
        Layer-wise erosion.
        x: B x C x H x W, a batch of images
        weight: B x C x kh x kw or C x kh x kw, a batch of SE filters
        bias: B x C or C
        '''
        padding = _pair(padding)
        stride = _pair(stride)

        B, C, H, W = x.shape
        kh, kw = weight.shape[-2:]
        ks_prod = kh * kw
        # B x C x H x W --> B x (C*kh*kw) x L
        x_unfold = nn.functional.unfold(x, kernel_size=(kh, kw), padding=padding, stride=stride)
        L = x_unfold.shape[-1]  # L sliding windows in image space
        y_unfold = (x_unfold.view(B, C, ks_prod, L) - weight.view(-1, C, ks_prod, 1)).min(dim=2)[0]  # B x C x L
        if bias is not None:
            y_unfold = y_unfold + bias.view(-1, C, 1)  # B x C x L
        H2 = math.floor((H + 2 * padding[0] - kh) / stride[0] + 1)
        W2 = math.floor((W + 2 * padding[1] - kw) / stride[1] + 1)
        assert H2 * W2 == L, 'Output size is wrongly computed in DilationLayer'

        # B x (C*kh2*kw2) x L --> B x C x H_out x W_out; "unfold" needs to re-computed and is independent to "fold"
        y = nn.functional.fold(y_unfold, output_size=(H2, W2), kernel_size=(1, 1), stride=1)  # B x C x H2 x W2

        return y

class UnifiedMorphoLayerExtSE(nn.Module):
    def __init__(self):
        '''
        Layer-wise dilation or erosion (default is dilation). Thus the output channel is equal to input channel.
        This unified layer has external SE, sign(v) * {max({sign(v) * x_i + w_i}) + bias}
        Optional to use apply_reg, namely regularization loss over weights
        '''
        super(UnifiedMorphoLayerExtSE, self).__init__()

    def forward(self, x, weight, padding=0, stride=1, bias=None, sign=None):
        '''
        Layer-wise dilation or erosion (default is dilation).
        x: B x C x H x W, a batch of images
        weight: B x C x kh x kw or C x kh x kw, a batch of SE filters
        bias: B x C or C
        sign: None or tensor size B or 1. Dilation, sign>0; erosion, sign<0.
        '''
        padding = _pair(padding)
        stride = _pair(stride)

        B, C, H, W = x.shape
        kh, kw = weight.shape[-2:]
        ks_prod = kh * kw
        # B x C x H x W --> B x (C*kh*kw) x L
        x_unfold = nn.functional.unfold(x, kernel_size=(kh, kw), padding=padding, stride=stride)
        if sign is not None:
            # if (abs(sign)>=1e-7):  # close to zero will lead image to be 0
            m = (abs(sign)>=1e-7).long().detach()  # create a mask for learned sign
            m = m.view(-1, 1, 1)
            x_unfold = m * sign.view(-1, 1, 1) * x_unfold + (1-m) * x_unfold
        L = x_unfold.shape[-1]  # L sliding windows in image space
        y_unfold = (x_unfold.view(B, C, ks_prod, L) + weight.view(-1, C, ks_prod, 1)).max(dim=2)[0]  # B x C x L
        if bias is not None:
            y_unfold = y_unfold + bias.view(-1, C, 1)  # B x C x L
        if sign is not None:
            # if (abs(sign) >= 1e-7):  # close to zero will lead image to be 0
            y_unfold = m * sign.view(-1, 1, 1) * y_unfold + (1-m) * y_unfold
        H2 = math.floor((H + 2 * padding[0] - kh) / stride[0] + 1)
        W2 = math.floor((W + 2 * padding[1] - kw) / stride[1] + 1)
        assert H2 * W2 == L, 'Output size is wrongly computed in DilationLayer'

        # B x (C*kh2*kw2) x L --> B x C x H_out x W_out; "unfold" needs to re-computed and is independent to "fold"
        y = nn.functional.fold(y_unfold, output_size=(H2, W2), kernel_size=(1, 1), stride=1)  # B x C x H2 x W2

        return y

class UnifiedMorphoLayerExtSE2(nn.Module):
    def __init__(self):
        '''
        Layer-wise dilation or erosion (default is dilation). Thus the output channel is equal to input channel.
        This unified layer has external SE, sign(v) * {max({sign(v) * x_i + w_i}) + bias}
        Optional to use apply_reg, namely regularization loss over weights
        '''
        super(UnifiedMorphoLayerExtSE2, self).__init__()

    def forward(self, x, weight, padding=0, stride=1, bias=None, sign=None, func_type=0, alpha=1.0):
        '''
        Layer-wise dilation or erosion (default is dilation).
        x: B x C x H x W, a batch of images
        weight: B x C x kh x kw or C x kh x kw, a batch of SE filters
        bias: B x C or C
        sign: None or tensor size B or 1. Dilation, sign>0; erosion, sign<0.
        func_type: 0, vanilla max/min operation; 1, SmoothMax; 2, LogSumExp.
        alpha: alpha is the temperature only used in SmoothMax & LogSumExp.
        '''
        padding = _pair(padding)
        stride = _pair(stride)

        B, C, H, W = x.shape
        kh, kw = weight.shape[-2:]
        ks_prod = kh * kw
        # B x C x H x W --> B x (C*kh*kw) x L
        x_unfold = nn.functional.unfold(x, kernel_size=(kh, kw), padding=padding, stride=stride)
        if sign is not None:
            # if (abs(sign)>=1e-7):  # close to zero will lead image to be 0
            m = (abs(sign)>=1e-7).long().detach()  # create a mask for learned sign
            m = m.view(-1, 1, 1)
            x_unfold = m * sign.view(-1, 1, 1) * x_unfold + (1-m) * x_unfold
        L = x_unfold.shape[-1]  # L sliding windows in image space
        if func_type == 0:  # vanilla max/min operation
            y_unfold = (x_unfold.view(B, C, ks_prod, L) + weight.view(-1, C, ks_prod, 1)).max(dim=2)[0]  # B x C x L
        elif func_type == 1:  # SmoothMax
            y_unfold = (x_unfold.view(B, C, ks_prod, L) + weight.view(-1, C, ks_prod, 1))  # B x C x ks_prod x L
            prob = (alpha * y_unfold).softmax(dim=2)
            y_unfold = (y_unfold * prob).sum(dim=2)  # B x C x L
        elif func_type == 2:  # LogSumExp
            y_unfold = (x_unfold.view(B, C, ks_prod, L) + weight.view(-1, C, ks_prod, 1))  # B x C x ks_prod x L
            y_unfold = (1.0/alpha) * torch.logsumexp(alpha * y_unfold, dim=2)  # B x C x L
        else:
            raise NotImplementedError
        if bias is not None:
            y_unfold = y_unfold + bias.view(-1, C, 1)  # B x C x L
        if sign is not None:
            # if (abs(sign) >= 1e-7):  # close to zero will lead image to be 0
            y_unfold = m * sign.view(-1, 1, 1) * y_unfold + (1-m) * y_unfold
        H2 = math.floor((H + 2 * padding[0] - kh) / stride[0] + 1)
        W2 = math.floor((W + 2 * padding[1] - kw) / stride[1] + 1)
        assert H2 * W2 == L, 'Output size is wrongly computed in DilationLayer'

        # B x (C*kh2*kw2) x L --> B x C x H_out x W_out; "unfold" needs to re-computed and is independent to "fold"
        y = nn.functional.fold(y_unfold, output_size=(H2, W2), kernel_size=(1, 1), stride=1)  # B x C x H2 x W2

        return y

def get_symmetric_matrix(mat, mode='central'):
    '''
    mat: B x C x H x W
    dim0: stands for dim of height
    dim1: stands for dim of width
    '''
    B, C, H, W = mat.shape

    y = torch.arange(H)
    x = torch.arange(W)
    if mode == 'central':  # central symmetric
        y = H - 1 - y
        x = W - 1 - x
    elif mode == 'horizontal':  # horizontal symmetric
        x = W - 1 - x
    elif mode == 'vertical':  # vertical symmetric
        y = H - 1 - y
    else:
        raise NotImplementedError

    yy, xx = torch.meshgrid(y, x)
    coords = torch.stack([yy, xx])  # 2 x H x W
    inds = coords.reshape(2, -1)  # 2 x (H*W)
    out = mat[:, :, inds[0], inds[1]].reshape(B, C, H, W)

    return out

class MorphologyNeuralNet(nn.Module):
    def __init__(self, pg_in_ch, pg_hidden_ch, pg_fc_layers=1, kernel_size=3, bias=False, type='both0', mnn_layers=1, T=1.0, T2=1.0, prob=1.0, impose_reg=0, fixed_MC=0):
        '''
        kernel_size: SE size
        bias: whether use bias in dilation and erosion
        type: 'both', unified dilation & erosion; 'dil', dilation; 'ero', erosion
               0, vanilla max/min operation; 1, SmoothMax; 2, LogSumExp.
        T: temperature for sign
        T2: temperature for SmoothMax or LogSumExp
        prob: probability to choose each mnn layer; stachastic layers to increase morphology diversity
        impose_reg: whether impose regularization loss over SE weights
        '''
        super(MorphologyNeuralNet, self).__init__()

        # size of SE
        self.kernel_size = _pair(kernel_size)
        self.impose_reg = impose_reg
        self.bias = bias
        # parse 'type' into 1) morphology type: 'both', 'dil', 'ero';
        #                   2) function type: 0, vanilla max/min operation; 1, SmoothMax; 2, LogSumExp.
        self.type = type[:-1]  # e.g., 'both0'-->'both'
        self.func_type = int(type[-1])  # e.g., 'both0'-->0

        # morphology parameter generator (MPG)
        self.mask_channel = 1
        if self.bias == False:
            out_ch = self.mask_channel * self.kernel_size[0] * self.kernel_size[1]
        else:
            out_ch = self.mask_channel * (self.kernel_size[0] * self.kernel_size[1] + 1)
        if self.type == 'both':  # allocate 1 variable to learn the sign, using tanh(v) = sign(v)
            out_ch = out_ch + 1
        self.mpg = ParamGenerator(in_ch=pg_in_ch, out_ch=out_ch, hidden_ch=pg_hidden_ch, fc_layers=pg_fc_layers)
        self.out_ch = out_ch

        # morphology changing module (MCM)
        # layer_list = []
        # for i in range(mnn_layers):
        #     layer_list.append(UnifiedMorphoLayerExtSE2())
        # self.mcm = nn.Sequential(*layer_list)
        self.num_layer = mnn_layers
        self.mcm = UnifiedMorphoLayerExtSE2()
        self.T = T  # temperature, sign(x)=tanh(Tx)
        self.T2 = T2
        self.prob = prob  # stachastic layers to increase morphology diversity

        self.fixed_MC = fixed_MC

    def forward(self, mask_feats, patch_feats, low_res_masks):
        '''
        mask_feats: B x Np x d
        patch_feats: B x Np x C
        low_res_masks: B x mask_channel x h' x w', usually mask_channel is 1.

        output: transformed low_res_mask, and regularization loss
        '''
        B, mask_channel = low_res_masks.shape[:2]
        assert mask_channel == self.mask_channel

        if self.fixed_MC == 0:  # learnable SE; don't fix SE
            x = torch.cat((mask_feats, patch_feats), dim=2)  # B x Np x (d+C)
            x = self.mpg(x)  # B x out_ch (out_ch = mask_channel*kh*kz)

            # SE weights (should >=0), B x 1 x kh x kw
            num_se_entries = self.mask_channel * self.kernel_size[0] * self.kernel_size[1]
            weights = x[:, :num_se_entries].reshape(B, self.mask_channel, self.kernel_size[0], self.kernel_size[1])
            # 1) Use relu or sigmoid to ensure SE weights >= 0
            # weights = torch.relu(weights)  # ReLu will lead dominant value in SE
            weights = torch.sigmoid(weights)  # avoid dominant elements in SE
            # 2) central symmetric SE
            weights_symmetric = get_symmetric_matrix(weights, mode='central')
            weights = (weights + weights_symmetric) / 2
            # print(weights)
        else:  # manually fix SE
            x = torch.zeros(B, self.out_ch).fill_(self.fixed_MC)
            if torch.cuda.is_available():
                x = x.cuda()
            # SE weights (should >=0), B x 1 x kh x kw
            num_se_entries = self.mask_channel * self.kernel_size[0] * self.kernel_size[1]
            weights = x[:, :num_se_entries].reshape(B, self.mask_channel, self.kernel_size[0], self.kernel_size[1])


        num_bias_entries = 0
        bias = None
        if self.bias:  # True
            num_bias_entries = self.mask_channel
            bias = x[:, num_se_entries:(num_se_entries+num_bias_entries)].reshape(B, self.mask_channel)
        signs = None
        if self.type == 'both':
            signs = x[:, (num_se_entries+num_bias_entries)].reshape(B)  # B
            signs = torch.tanh(self.T*signs)  # sign(x)=tanh(Tx)
            # print("MorphologyNeuralNet sign: ", signs)
        elif self.type == 'dil':
            signs = torch.tensor(1)  # 1
            if torch.cuda.is_available():
                signs = signs.cuda()
        elif self.type == 'ero':
            signs = torch.tensor(-1)  # -1
            if torch.cuda.is_available():
                signs = signs.cuda()
        else:
            print('Error for morphology type in MorphologyNeuralNet.')
            exit(0)
        # print(signs)

        padding=(self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        H = low_res_masks

        # store original max/min value for Contrast-preserving Normalization
        ori_min = torch.min(H.view(B, mask_channel, -1), dim=2)[0]  # per image min
        ori_max = torch.max(H.view(B, mask_channel, -1), dim=2)[0]  # per image max
        ori_min = ori_min.reshape(B, mask_channel, 1, 1).detach()  # no need to learn
        ori_max = ori_max.reshape(B, mask_channel, 1, 1).detach()  # no need to learn
        contrast = ori_max - ori_min

        # for i in range(len(self.mcm)):
        #     H = self.mcm[i](H, weights, padding=padding, stride=1, bias=bias, sign=signs, func_type=self.func_type, alpha=self.T2)
        for i in range(self.num_layer):
            rand_num = np.random.rand(1)
            if rand_num <= self.prob:  # stachastic MNN layers to increase mask diversity
                H = self.mcm(H, weights, padding=padding, stride=1, bias=bias, sign=signs, func_type=self.func_type, alpha=self.T2)

                # Contrast-preserving Normalization, i.e., normalization after dilation or erosion as the gray value has been modified.
                v_min = torch.min(H.view(B, mask_channel, -1), dim=2)[0]  # per image min
                v_max = torch.max(H.view(B, mask_channel, -1), dim=2)[0]  # per image max
                v_min = v_min.reshape(B, mask_channel, 1, 1).detach()  # no need to learn
                v_max = v_max.reshape(B, mask_channel, 1, 1).detach()  # no need to learn
                H = (H - v_min) / (v_max - v_min + 1e-9)  # normalize to 0~1
                H = H * contrast + ori_min

        if self.impose_reg == 0:  # no reg_loss
            loss_reg = torch.tensor(0.)
        else:
            if self.impose_reg == 1:  # || M'-M||^2, consistency
                loss_reg = (H - low_res_masks)**2  # (B x 1 x H x W)
            elif self.impose_reg == 2:  # || M'-M||, consistency
                loss_reg = torch.abs(H - low_res_masks)
            elif self.impose_reg == 3:  # || 0 - W ||^2, regularization loss for SE W
                loss_reg = weights ** 2  # (B x 1 x ks x ks)
            elif self.impose_reg == 4:  # || 0 - W ||
                loss_reg = torch.abs(weights)
            else:
                print('Error for impose_reg in MorphologyNeuralNet.')
                exit(0)
            loss_reg = loss_reg.reshape(B, -1).mean(dim=1)  # B

        return H, loss_reg

class SalMorphologyLearner2(nn.Module):
    '''
    Note: this MorphologyLearner should be used together with the SalTransformer
    '''
    def __init__(
            self,
            in_ch=1, out_ch=1,
            src_size=(384, 384),  # mask size
            dst_size=(12, 12),  # target grid size
            pre_downscale=32,  # downsize mask by this ratio (>=2); if <= 1, it means no using of pre-downscale
            blur_sigma=0.0,  # if sigma > 0, apply blur using sigma with window size 6*sigma (+-3sigma); otherwise no blur
            num_vits=1,  # the number of vits
            sal_mask_type='IS',  # Individual morphological mask, Shared MU (morphology unit = MPG + MCM)
            morphology_type='pow',  # 'pow', 'mnn'
            sem_cfg=(128, 3, 256),  # (embedding_dim, hidden_layers (conv3x3), out_dim)
            norm_type='bn',  # 'bn', 'ln', 'no' (no norm)
            vit_feat_proj_cfg=(2048),  # (vit_feat_dim)
            mpg_cfg=(512, 1),  # (mpg_hidden_dim, mpg_layer)
            pow_cfg=(1, 0, 0.1, 1.0, 0, 3, 0),  # (num_pow_param, regress_type, bound, T, num_conv, conv_ks, conv_reg_type) for 'pow' morphology
            mnn_cfg=(5, 'both0', 1, 1.0, 1.0, 1.0),  # (SE_size, type, mnn_layer, T, T2, prob) for MNN
            impose_reg=0,  # impose regularization loss for parameters
            pn_center=1.0,  # the power normalization center for regularization, e.g., (R- center)**2.
            fixed_MC=0,  # 0, don't fixed; >0 fixed pow for 'pow based MC' or fixed SE for 'MNN'
            vis=False,  # whether visualize
    ):
        super(SalMorphologyLearner2, self).__init__()

        self.pre_downscale = pre_downscale
        if pre_downscale >= 2:  # < 2 (namely 1 or 0), it means no usage of pre_downscale
            self.pre_downscale_layer = nn.AvgPool2d(kernel_size=pre_downscale, stride=pre_downscale)
        # self.low_res_masks = None  # used to store low_res saliency masks
        # self.blur_sigma = blur_sigma
        # if self.blur_sigma > 0:  # Gaussian blur
        #     stride = 1
        #     start = stride / 2 - 0.5
        #     kernel_w_half = round(3 * blur_sigma / stride)
        #     kernel_w = 2 * kernel_w_half + 1
        #     kernel_center = torch.Tensor([0.5, 0.5]).mul(kernel_w - 1).cuda()
        #     gaussian_kernel = torch.zeros(kernel_w, kernel_w).cuda()  # gaussian kernel
        #     gaussian_kernel[:, :], return_flag = putGaussianMaps(kernel_center*stride + start, blur_sigma, kernel_w, kernel_w, stride, normalization=True)
        #     self.gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_w, kernel_w)  # 1 x 1 x kernel_w x kernel_w
        #     self.kernel_w_half = kernel_w_half
        #     # print(self.gaussian_kernel)
        #     print('Gaussian Blur (%f) Used!'%self.blur_sigma)

        h_scale = src_size[0] // dst_size[0]
        w_scale = src_size[1] // dst_size[1]
        assert (h_scale == w_scale), 'the h_scale should be equal to w_scale.'
        scale = h_scale
        assert (pre_downscale <= scale), 'pre_downscale is too big.'
        if pre_downscale == scale:  # no need to perform post-downscale
            self.post_downscale = 0
        elif pre_downscale < 2:  # no pre-downscale, we need full scale for post processing
            self.post_downscale = scale
        else:
            self.post_downscale = scale // pre_downscale

        # 1) SEM: saliency mask embedding module
        self.sal_emb_modul = ConvEmbedding(in_channel=in_ch, patch_size=scale, embedding_dim=sem_cfg[0],
                                           hidden_layers=sem_cfg[1], out_channel=sem_cfg[2], norm_type=norm_type)
        # self.sem_feats = None  # used to store sem features

        # 2) FC: construct projection layer to project semantic features from image data
        vit_dim = vit_feat_proj_cfg[0]
        assert num_vits >= 1, 'num_vit requires >= 1 in SalMorphologyLearner2.'
        if sal_mask_type == 'S':  # only 1 projection layer
            self.vit_feat_proj_layers = nn.Sequential(
                nn.Linear(vit_dim, vit_dim, bias=True)
            )
        else:  # 'IS', 'II'; num_vit projection layers
            proj_layers_list = []
            for i in range(num_vits):
                proj_layers_list.append(nn.Linear(vit_dim, vit_dim, bias=True))
            self.vit_feat_proj_layers = nn.Sequential(*proj_layers_list)
        # init fc weights
        for i in range(len(self.vit_feat_proj_layers)):
            nn.init.xavier_uniform_(self.vit_feat_proj_layers[i].weight, gain=1.0)
            nn.init.constant_(self.vit_feat_proj_layers[i].bias, 0)

        # 3) MU: construct morphology unit
        self.sal_mask_type = sal_mask_type
        self.morphology_type = morphology_type
        self.impose_reg = impose_reg
        if (sal_mask_type == 'S') or (sal_mask_type =='IS'):
            # print('only one MU')
            num_mc = 1
        elif sal_mask_type == 'II':
            # print('num_vit MUs')
            num_mc = num_vits
        else:
            print('Error for sal_mask_type in SalMorphologyLearner2.')
            exit(0)
        mc_list = []
        self.morphology_type = morphology_type
        if morphology_type == 'pow':
            for i in range(num_mc):
                mc_list.append(
                    PowLearner2(
                        in_ch=vit_feat_proj_cfg[0] + sem_cfg[2],  # C + d
                        hidden_ch=mpg_cfg[0],  # 512
                        fc_layers=mpg_cfg[1],  # 1
                        out_ch=pow_cfg[0],  # 1
                        regress_type=pow_cfg[1],  # 0
                        bound=pow_cfg[2],  # 0.1
                        T=pow_cfg[3],  # 1.0
                        impose_reg=impose_reg,
                        pn_center=pn_center,
                        fixed_MC=fixed_MC,
                        conv_cfg=(pow_cfg[4], pow_cfg[5], pow_cfg[6]),
                    )
                )
        elif morphology_type == 'mnn':
            for i in range(num_mc):
                mc_list.append(
                    MorphologyNeuralNet(
                        pg_in_ch=vit_feat_proj_cfg[0] + sem_cfg[2],  # C + d
                        pg_hidden_ch=mpg_cfg[0],  # 512
                        pg_fc_layers=mpg_cfg[1],  # 2
                        kernel_size=mnn_cfg[0],
                        bias=False,
                        type=mnn_cfg[1],  # 'both0', 'dil0', 'ero0'
                        mnn_layers=mnn_cfg[2],
                        T=mnn_cfg[3],
                        T2=mnn_cfg[4],
                        prob=mnn_cfg[5],
                        impose_reg=impose_reg,
                        fixed_MC=fixed_MC,
                    )
                )
        else:
            print('Error for morphology_type in SalMorphologyLearner2.')
            exit(0)
        self.mpg_mcm = nn.Sequential(*mc_list)

        if self.post_downscale >= 2:
            self.post_downscale_layer = nn.AvgPool2d(kernel_size=self.post_downscale, stride=self.post_downscale)

        self.vis = vis

    def compute_mask_feats(self, mask, im=None):
        '''
        Two things are done here: 1) the semantic_input (mask+im) are embedded into feature space using large-strided conv.
                                  (e.g. stride=32); 2) the mask is pre-downscaled for morphology changing (MC).
        input : mask: B x 1 x H x W
        input : im: B x 3 x H x W (optional)
        output: mask_feats ( B x d x h x w-->B x (h*w) x d ), and downscaled masks (B x 1 x h x w)
        '''
        if im is None:
            sem_input = mask
        else:
            sem_input = torch.cat([mask, im], dim=1)

        mask_feats = self.sal_emb_modul(sem_input)
        B, d, h, w = mask_feats.shape
        mask_feats = mask_feats.reshape(B, d, h*w).transpose(2, 1)  # B x (h*w) x d
        if self.pre_downscale >= 2:
            low_res_masks = self.pre_downscale_layer(mask)
        else:
            low_res_masks = mask
        # if self.blur_sigma > 0:  # Gaussian blur
        #     low_res_masks = nn.functional.conv2d(low_res_masks, self.gaussian_kernel, bias=None, stride=1, padding=self.kernel_w_half)

        if self.morphology_type == 'pow':
            # set mask value range; avoid 0^R as its gradient may be inf.
            low_res_masks = torch.clamp(low_res_masks, min=0.02, max=0.98)

        return mask_feats, low_res_masks

    def forward(self, patch_feats, mask_feats, low_res_masks, vit_index=0):
        '''
        patch_feats: B x Np x C
        mask_feats: B x Np x d
        low_res_masks: B x mask_channel x h' x w', usually mask_channel is 1.
        vit_index: i-th vit. MorphologyLearner works together with CascadeTransformer
        output: final_masks, vis_pow_masks, each is a list containing a tensor
        which has size of B x out_ch x h x w
        '''

        if self.sal_mask_type == 'S':
            if vit_index == 0:
                patch_feats = self.vit_feat_proj_layers[0](patch_feats)
                m2, loss_reg, m = self.mpg_mcm[0](mask_feats, patch_feats, low_res_masks)
            else:  # for 'S' type, only compute morphological mask in first vit and shared for other vits.
                return None, None, None

        elif self.sal_mask_type == 'IS':
            patch_feats = self.vit_feat_proj_layers[vit_index](patch_feats)
            m2, loss_reg, m = self.mpg_mcm[0](mask_feats, patch_feats, low_res_masks)

        elif self.sal_mask_type == 'II':
            patch_feats = self.vit_feat_proj_layers[vit_index](patch_feats)
            m2, loss_reg, m = self.mpg_mcm[vit_index](mask_feats, patch_feats, low_res_masks)
        else:
            print('Error for sal_mask_type in SalMorphologyLearner2.')
            exit(0)

        if self.vis == True:
            m_draws = m2.detach().clone()
            # pow_tmp = (m.detach().clone() * 255).long()
            # origin_tmp = (low_res_masks.detach().clone() * 255).long()
            # m_draws = self.draw_contours_batch2(m=origin_tmp, m2=pow_tmp, im=pow_tmp, thresh=int(0.4 * 255))
        else:
            m_draws = None

        if self.post_downscale >= 2:
            m2 = self.post_downscale_layer(m2)

        return m2, loss_reg, m_draws

    def draw_contours_batch(self, m, im, thresh=int(0.4*255)):
        '''
        m: B x 1 x H x W
        im: B x C x H x W
        mask or im's value range: 0~255
        '''
        B, _, H, W = m.shape
        m = m.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()
        im = im.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()  # B x H x W x C
        out_list = []
        for i in range(B):
            out = draw_contours(thresh, mask=m[i], im=im[i])
            out_list.append(out)

        outs = np.stack(out_list)
        outs = torch.tensor(outs).permute(0, 3, 1, 2)

        return outs

    def draw_contours_batch2(self, m, m2=None, im=None, thresh=int(0.4*255)):
        '''
        overlay contours of m (opt. m2) onto im.
        m: B x 1 x H x W
        m2: B x 1 x H x W
        im: B x C x H x W
        mask or im's value range: 0~255
        '''
        B, _, H, W = m.shape
        m = m.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()
        if m2 is not None:
            m2 = m2.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()
        im = im.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()  # B x H x W x C
        out_list = []
        for i in range(B):
            out = draw_contours(thresh, mask=m[i], im=im[i], color='pink')
            if m2 is not None:
                out = draw_contours(thresh, mask=m2[i], im=out, color='white')
            out_list.append(out)

        outs = np.stack(out_list)
        outs = torch.tensor(outs).permute(0, 3, 1, 2)

        return outs

    def draw_contours_batch3(self, m, m2=None, m3=None, im=None, thresh=int(0.4*255), thickness=1):
        '''
        overlay contours of m (opt. m2, opt. m3) onto im.
        m: B x 1 x H x W
        m2: B x 1 x H x W
        m3: B x 1 x H x W
        im: B x C x H x W
        mask or im's value range: 0~255
        '''
        B, _, H, W = m.shape
        m = m.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()
        if m2 is not None:
            m2 = m2.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()
        if m3 is not None:
            m3 = m3.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()
        im = im.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8).copy()  # B x H x W x C
        out_list = []
        for i in range(B):
            out = draw_contours(int(0.4*255), mask=m[i], im=im[i], color='pink', thickness=thickness)
            if m2 is not None:
                out = draw_contours(thresh, mask=m2[i], im=out, color='blue', thickness=thickness)
            if m3 is not None:
                out = draw_contours(thresh, mask=m3[i], im=out, color='white', thickness=thickness)
            out_list.append(out)

        outs = np.stack(out_list)
        outs = torch.tensor(outs).permute(0, 3, 1, 2)  # B x C x H x W

        return outs