import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import os
import cv2
import copy

import logging

import datasets.dataset_utils
import datasets.transforms as mytransforms
from datasets.AnimalPoseDataset.animalpose_dataset import EpisodeGenerator, AnimalPoseDataset, KEYPOINT_TYPE_IDS, KEYPOINT_TYPES, save_episode_before_preprocess, kp_connections
from datasets.AnimalPoseDataset.animalpose_dataset import horizontal_swap_keypoints, get_symmetry_keypoints, HFLIP, FBFLIP, DFLIP, get_auxiliary_paths
from datasets.dataset_utils import draw_skeletons, draw_instance, draw_markers

from network.models_gridms2 import Encoder
from network.models_gridms2 import feature_modulator2, extract_representations, average_representations2, feature_modulator3
from network.transformer import SalTransformer
from network.models_fskd_vit import AttentionBasedModulator, CascadeSalTransformer

from functools import partial

from coco_eval_funs import compute_recall_ap

from utils import print_weights, image_normalize, batch_images_normalize, make_grid_images, make_uncertainty_map, compute_eigenvalues, mean_confidence_interval, mean_confidence_interval_multiple
from utils import get_patches, DTModule, GaussianBlur, AverageMeter, get_model_summary


def val_transductive(model, episode_generator, num_test_episodes=1000, eval_method='method1', using_crop='True'):
    if model.opts['set_eval']:
        model.set_eval_status()
    torch.set_grad_enabled(False)  # disable grad computation
    if episode_generator == None:
        return

    print('==============testing start==============')
    ks_sigmas = np.array([.25, .25, .35, .35, .26,
                          1.07, 1.07, 1.07,
                          1.07, 1.07, 1.07, 1.07,
                          .87, .87, .87, .87,
                          .89, .89, .89, .89]) / 10.0
    # specified_kp_ids = [KEYPOINT_TYPE_IDS[kp_type] for kp_type in episode_generator.support_kp_categories]
    # specified_ks_sigmas = ks_sigmas[specified_kp_ids]

    square_image_length = model.opts['square_image_length']  # 368
    pck_thresh_bbx =  np.array([0.1])  # np.array([0.1, 0.15]), np.linspace(0, 1, 101)
    # pck_thresh_img = np.array([20.0 / 368, 40.0 / 368])  # 0.06 * 384 = 23.04 pixels (23 pixels)
    pck_thresh_img = np.array([0.06, 0.1])  # 0.06 * 384 = 23.04 pixels (23 pixels)
    pck_thresh_type = 'bbx'  # 'bbx' or 'img'
    if pck_thresh_type == 'bbx':  # == 'bbx'
        pck_thresh = pck_thresh_bbx
    else:  # == 'img'
        pck_thresh = pck_thresh_img

    ks_thresh = np.array([0.5, 0.75])
    tps, fps = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]
    tps2, fps2 = [[] for _ in range(len(ks_thresh))], [[] for _ in range(len(ks_thresh))]

    tps3, fps3 = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]
    # tps4, fps4 = [[] for _ in range(len(pck_thresh))], [[] for _ in range(len(pck_thresh))]

    acc_list = [[] for _ in range(len(pck_thresh))]
    acc_list2 = [[] for _ in range(len(pck_thresh))]
    scores_list = [[] for _ in range(len(pck_thresh))]

    # Used to stored datapoints for transductive FSKD
    model.datapoint = {'pred': [], 'gt': [], 'scores': [], 'uc': [], 'kp_mask': [], 'simi_scores': []}
    model.datapoint_list1 = []
    model.datapoint_list2 = []
    #----------------------
    # used to compute relationship between normalized distance error and localization uncertainty
    # d_increment = 0.05
    # d_norm_max = 0.3  # 0~d_norm_max
    # N_bins = d_norm_max / d_increment  # 3/0.05=5.9999999
    # if 1+int(N_bins)-N_bins <= 1e-9:
    #     N_bins = 1+int(N_bins)
    # else:
    #     N_bins = int(N_bins)
    # d_uc_bins = np.zeros((N_bins, 4))  # each row store summation of d_dorm_i, uc_loc_i, uc_sd_i (optional), count
    #----------------------

    episode_i = 0
    sample_failure_cnt = 0
    using_interpolated_kps = False  # model.opts['use_interpolated_kps']
    interpolation_knots = model.opts['interpolation_knots']
    # ---
    finetuning_steps = model.opts['finetuning_steps']
    if finetuning_steps > 0:
        original_params_dict = model.copy_original_params()
    # ---
    if model.opts['use_body_part_protos'] and model.opts['eval_with_body_part_protos']:
        model.load_proto_memory()
        body_part_proto = model.memory['proto']  # C x N
        proto_mask = model.memory['proto_mask']  # N
    #---
    eval_throughput = True
    throughput0 = 0
    throughput = 0
    if eval_throughput:
        throughput_meter0 = AverageMeter()
        throughput_meter = AverageMeter()
    while episode_i < num_test_episodes:
        # roll-out an episode
        if (False == episode_generator.episode_next()):
            sample_failure_cnt += 1
            if sample_failure_cnt % 500 == 0:
                print('sample failure times: {}'.format(sample_failure_cnt))
            continue
        # print(episode_generator.support_kp_categories)

        if using_crop:
            preprocess = mytransforms.Compose([
                mytransforms.RandomCrop(crop_bbox=False),
                mytransforms.Resize(longer_length=model.opts['square_image_length']),  # 368
                mytransforms.CenterPad(target_size=model.opts['square_image_length']),
                mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
            ])
        else:
            preprocess = mytransforms.Compose([
                mytransforms.Resize(longer_length=model.opts['square_image_length']),  # 368
                mytransforms.CenterPad(target_size=model.opts['square_image_length']),
                mytransforms.CoordinateNormalize(normalize_keypoints=True, normalize_bbox=True)
            ])
        image_transform = transforms.Compose([
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            # transforms.RandomGrayscale(p=0.01),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # define a list containing the paths, each path is represented by kp index pair [index1, index2]
        # our paths are subject to support keypoints; then we will interpolated kps for each path. The paths is possible to be empty list []
        num_random_paths = model.opts['num_random_paths']  # only used when auxiliary_path_mode='random'
        # path_mode:  'exhaust', 'predefined', 'random'
        auxiliary_paths = get_auxiliary_paths(path_mode=model.opts['auxiliary_path_mode'], support_keypoint_categories=episode_generator.support_kp_categories, num_random_paths=num_random_paths)

        support_dataset = AnimalPoseDataset(episode_generator.supports,
                                            episode_generator.support_kp_categories,
                                            using_auxiliary_keypoints=using_interpolated_kps,
                                            interpolation_knots=interpolation_knots,
                                            interpolation_mode=model.opts['interpolation_mode'],
                                            auxiliary_path=auxiliary_paths,
                                            hdf5_images_path=model.opts['hdf5_images_path'],
                                            saliency_maps_root=model.opts['saliency_maps_root'],
                                            output_saliency_map=True,  # model.opts['use_pum']
                                            preprocess=preprocess,
                                            input_transform=image_transform
                                            )

        query_dataset = AnimalPoseDataset(episode_generator.queries,
                                          episode_generator.support_kp_categories,
                                          using_auxiliary_keypoints=using_interpolated_kps,
                                          interpolation_knots=interpolation_knots,
                                          interpolation_mode=model.opts['interpolation_mode'],
                                          auxiliary_path=auxiliary_paths,
                                          hdf5_images_path=model.opts['hdf5_images_path'],
                                          saliency_maps_root=model.opts['saliency_maps_root'],
                                          output_saliency_map=True,  # model.opts['use_pum']
                                          preprocess=preprocess,
                                          input_transform=image_transform
                                          )

        support_loader = DataLoader(support_dataset, batch_size=model.opts['K_shot'], shuffle=False)
        query_loader = DataLoader(query_dataset, batch_size=model.opts['M_query'], shuffle=False)

        support_loader_iter = iter(support_loader)
        query_loader_iter = iter(query_loader)
        (supports, support_labels, support_kp_mask, support_scale_trans, _, _, support_saliency, s_bbx_origin, s_w_h_origin) = support_loader_iter.next()
        (queries, query_labels, query_kp_mask, query_scale_trans, _, _, query_saliency, query_bbx_origin, query_w_h_origin) = query_loader_iter.next()

        if eval_throughput:
            throughput_t_start = time.time()

        # make_grid_images(supports, denormalize=True, save_path='grid_image_s.jpg')
        # make_grid_images(queries, denormalize=True, save_path='grid_image_q.jpg')
        # make_grid_images(support_saliency.cuda(), denormalize=False, save_path='./ss.jpg')
        # make_grid_images(query_saliency.cuda(), denormalize=False, save_path='./sq.jpg')
        # print(episode_generator.supports)
        # save_episode_before_preprocess(episode_generator, episode_i, delete_old_files=False, draw_main_kps=True, draw_interpolated_kps=False, interpolation_knots=model.opts['interpolation_knots'], interpolation_mode=model.opts['interpolation_mode'], path_mode='predefined')
        # show_save_episode(supports, support_labels, support_kp_mask, queries, query_labels, query_kp_mask, episode_generator, episode_i,
        #                   is_show=False, is_save=True, delete_old_files=True)
        # support_kp_mask = episode_generator.support_kp_mask  # B1 x N
        # query_kp_mask = episode_generator.query_kp_mask  # B2 x N
        if torch.cuda.is_available():
            supports, queries = supports.cuda(), queries.cuda()
            support_labels, query_labels = support_labels.float().cuda(), query_labels.cuda()  # .float().cuda()
            support_kp_mask = support_kp_mask.cuda()  # B1 x N
            query_kp_mask = query_kp_mask.cuda()  # B2 x N

            # ----------------------------------------------------------
            # Pre-Image Processing for Saliency Maps: DT & GaussianBlur | testing phase
            support_saliency_t = model.dt_module.dt(support_saliency)  # B1 x 1 x H0 x W0 (H0=W0=384)
            query_saliency_t = model.dt_module.dt(query_saliency)  # B1 x 1 x H0 x W0 (H0=W0=384)

            support_saliency, query_saliency = support_saliency.cuda(), query_saliency.cuda()  # original saliency maps
            support_saliency_t, query_saliency_t = support_saliency_t.cuda(), query_saliency_t.cuda()  # transformed ones

            support_saliency_t = model.blur_module.gaussian_blur(support_saliency_t)
            query_saliency_t = model.blur_module.gaussian_blur(query_saliency_t)
            # ----------------------------------------------------------

        if model.opts['use_body_part_protos'] and model.opts['eval_with_body_part_protos']:
            # retrieve index, which makes the retrieved body part order fit current oder, in case "order_fixed = False", namely the dynamic support kp categories
            order_index = [KEYPOINT_TYPES.index(kp_type) for kp_type in episode_generator.support_kp_categories]
            proto_mask_temp = copy.deepcopy(proto_mask)    # N
            proto_mask_temp = proto_mask_temp[order_index] # N, retrieve from standard order to fit current order in case dynamic support_kp_categories
            B1 = supports.shape[0]
            support_kp_mask = proto_mask_temp.repeat(B1, 1)  # B1 x N, overwrite support_kp_mask, union_support_kp_mask

        # compute the union of keypoint types in sampled images, N(union) <= N_way, tensor([True, False, True, ...])
        union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0  # N
        # compute the valid query keypoints, using broadcast
        valid_kp_mask = (query_kp_mask * union_support_kp_mask.reshape(1, -1))  # B2 x N
        num_valid_kps = torch.sum(valid_kp_mask)

        num_valid_kps_for_samples = torch.sum(valid_kp_mask, dim=1)  # B2
        gp_valid_samples_mask = num_valid_kps_for_samples >= episode_generator.least_query_kps_num  # B2
        gp_num_valid_kps = torch.sum(num_valid_kps_for_samples * gp_valid_samples_mask)
        gp_valid_kp_mask = valid_kp_mask * gp_valid_samples_mask.reshape(-1, 1)  # (B2 x N) * (B2, 1) = (B2 x N)

        # check #1    there may exist some wrongly labeled images where the keypoints are outside the boundary
        if torch.any(support_labels > 1) or torch.any(support_labels < -1) or torch.any(query_labels > 1) or torch.any(query_labels < -1):
            continue             # skip current episode directly
        # check #2
        if num_valid_kps ==  0:  # flip transform may lead zero intersecting keypoints between support and query, namely zero valid kps
            continue             # skip directly

        if finetuning_steps > 0:
            model.finetuning_via_gradient_steps(finetuning_steps, original_params_dict, supports, support_labels, support_kp_mask, support_saliency_t)

        if eval_throughput:
            throughput_t_start = time.time()

        if model.encoder_type == 0:  # Encoder
            # feature and semantic distinctiveness, note that x2 = support_saliency.cuda() or query_saliency.cuda()
            support_features, support_lateral_out = model.encoder(x=supports, x2=None, enable_lateral_output=False)   # B1 x C x H x W, B1 x 1 x H x W
            query_features, query_lateral_out = model.encoder(x=queries, x2=None, enable_lateral_output=False)   # B2 x C x H x W, B2 x 1 x H x W

            # attentive_features, attention_maps_l2, attention_maps_l1, _ = \
            #     feature_modulator2(support_features, support_labels, support_kp_mask, query_features, context_mode=model.context_mode,
            #                        sigma=model.opts['sigma'], downsize_factor=model.opts['downsize_factor'], image_length=model.opts['square_image_length'],
            #                        fused_attention=model.opts['use_fused_attention'], output_attention_maps=False)


            if model.opts['use_body_part_protos'] and model.opts['eval_with_body_part_protos']:
                # we should use body_part_proto and proto_mask (updated at above)
                avg_support_repres = copy.deepcopy(body_part_proto)    # C x N
                avg_support_repres = avg_support_repres[:, order_index]  # N, retrieve from standard order to fit current order in case dynamic support_kp_categories
                attentive_features, attention_maps_l2, attention_maps_l1, _ = feature_modulator3(avg_support_repres, query_features, \
                                   fused_attention=model.opts['use_fused_attention'], output_attention_maps=True, compute_similarity=False)
            else:
                # B1 x C x N
                # support_repres, conv_query_features = extract_representations(support_features, support_labels, support_kp_mask, context_mode=model.context_mode,
                #                 sigma=model.opts['sigma'], downsize_factor=model.opts['downsize_factor'], image_length=model.opts['square_image_length'], together_trans_features=None)
                # avg_support_repres = average_representations2(support_repres, support_kp_mask)  # C x N
                # attentive_features, attention_maps_l2, attention_maps_l1, _ = feature_modulator3(avg_support_repres, query_features, \
                #                        fused_attention=model.opts['use_fused_attention'], output_attention_maps=True, compute_similarity=False)

                # testing phase
                salT_att_param_config = {
                    'mask_mode': model.opts['salT_cfg']['mask_mode'],
                    'func_mode': model.opts['salT_cfg']['func_mode'],
                    'sal_beta': model.opts['salT_cfg']['sal_beta'],
                    'sal_eps': model.opts['salT_cfg']['sal_eps'],
                    'thresh': model.opts['salT_cfg']['thresh'],
                    'use_max': model.opts['salT_cfg']['use_max'],
                    'rbf_l2norm': model.opts['salT_cfg']['rbf_l2norm'],
                    'rownorm': model.opts['salT_cfg']['rownorm'],
                }

                # patches, testing
                # patched_sal_s = get_patches(support_saliency_t.cpu(), patch_size=(32, 32), save=True, prefix='s')
                # patched_sal_q = get_patches(query_saliency_t.cpu(),   patch_size=(32, 32), save=True, prefix='q')

                if using_interpolated_kps == False:
                    support_aux_kps, support_aux_kp_mask = None, None
                extract_kp_repres_func = partial(extract_representations, context_mode=model.context_mode, sigma=model.opts['sigma'], \
                                                 downsize_factor=model.opts['downsize_factor'], image_length=model.opts['square_image_length'], together_trans_features=None)

                s_feats_select, vit_s_attn_probs, s_vis_pow_masks, s_morph_reg_loss, s_morph_reg_loss_o, _ = \
                model.cascade_salT(
                    support_features,  # B1 x C x h x w (feature space)
                    support_saliency_t,# B1 x 1 x H x W (image space)
                    supports,          # B1 x 3 x H x W (image space)
                    salT_att_param_config
                    )
                q_feats_select, vit_q_attn_probs, q_vis_pow_masks, q_morph_reg_loss, q_morph_reg_loss_o, _ = \
                model.cascade_salT(
                    query_features,   # B2 x C x h x w (feature space)
                    query_saliency_t, # B2 x 1 x H x W (image space)
                    queries,          # B2 x 3 x H x W (image space)
                    salT_att_param_config
                    )
                if (model.opts['sal_mask_cfg']['sal_mask_type'] in ['S', 'IS', 'II']) and model.opts['sal_mask_cfg']['impose_reg'] > 0:
                    morph_reg_N = s_morph_reg_loss.shape[0] + q_morph_reg_loss.shape[0]  # B1 + B2
                    morph_reg_loss = (s_morph_reg_loss.sum() + q_morph_reg_loss.sum()) / morph_reg_N
                    morph_reg_loss_o = (s_morph_reg_loss_o.sum() + q_morph_reg_loss_o.sum()) / morph_reg_N
                else:
                    morph_reg_loss = 0
                    morph_reg_loss_o = 0

                if eval_throughput:
                    throughput_meter0.update((time.time() - throughput_t_start) / valid_kp_mask.shape[0])  # sec / im
                    throughput_t_start = time.time()

                attentive_feats, attention_maps_l2, attentive_feats_aux, attention_maps_l2_aux = \
                model.cascade_salT.support_query_modulation(
                    s_feats_select,
                    q_feats_select,
                    support_labels,
                    support_kp_mask,
                    support_aux_kps,
                    support_aux_kp_mask,
                    extract_kp_repres_func,
                    fused_attention= model.opts['use_fused_attention'],
                    output_attention_maps=False)

                # show pow-maps, testing
                # if episode_i % 8 == 0:
                # for i in range(len(s_vis_pow_masks)):
                #     s_vis_pow_tmp = torch.nn.functional.interpolate(s_vis_pow_masks[i], size=(384, 384), mode='bilinear', align_corners=False)
                #     q_vis_pow_tmp = torch.nn.functional.interpolate(q_vis_pow_masks[i], size=(384, 384), mode='bilinear', align_corners=False)
                #     s_vis_pow_tmp, q_vis_pow_tmp = (s_vis_pow_tmp*255).long(), (q_vis_pow_tmp*255).long()
                #     s_im_tmp = (batch_images_normalize(supports.clone(), denormalize=True) * 255).long()
                #     q_im_tmp = (batch_images_normalize(queries.clone(), denormalize=True) * 255).long()
                #     s_sal_tmp, q_sal_tmp = (support_saliency * 255).long(), (query_saliency * 255).long()
                #     s_sal_tmp2, q_sal_tmp2 = (support_saliency_t * 255).long(), (query_saliency_t * 255).long()
                #     s_vis_pow_out = model.cascade_salT.morphology_learner.draw_contours_batch3(m=s_sal_tmp, m2=s_sal_tmp2, m3=s_vis_pow_tmp, im=s_im_tmp)
                #     q_vis_pow_out = model.cascade_salT.morphology_learner.draw_contours_batch3(m=q_sal_tmp, m2=q_sal_tmp2, m3=q_vis_pow_tmp, im=q_im_tmp)
                #     s_vis_pow_out2= model.cascade_salT.morphology_learner.draw_contours_batch3(m=s_sal_tmp, m2=s_sal_tmp2, m3=None, im=s_vis_pow_tmp)
                #     q_vis_pow_out2= model.cascade_salT.morphology_learner.draw_contours_batch3(m=q_sal_tmp, m2=q_sal_tmp2, m3=None, im=q_vis_pow_tmp)
                #     make_grid_images(s_vis_pow_out/255, denormalize=False, save_path='./episode_images/vis_pow_maps/s_Learner{}.jpg'.format(i))
                #     make_grid_images(q_vis_pow_out/255, denormalize=False, save_path='./episode_images/vis_pow_maps/q_Learner{}.jpg'.format(i))
                #     make_grid_images(s_vis_pow_out2/255, denormalize=False, save_path='./episode_images/vis_pow_maps/s_Learner{}_t.jpg'.format(i))
                #     make_grid_images(q_vis_pow_out2/255, denormalize=False, save_path='./episode_images/vis_pow_maps/q_Learner{}_t.jpg'.format(i))

                # model.cascade_salT.salT.draw_attn_probs(vit_s_attn_probs[0][0], patch_ind=0, save_path_wo_ext='s', does_show=False)
                # model.cascade_salT.salT.draw_attn_probs2(vit_s_attn_probs[0], save_path_wo_ext='whole_s', does_show=False)

            # show_save_attention_maps(attention_maps_l2, queries, support_kp_mask, query_kp_mask, episode_generator.support_kp_categories,
            #                          episode_num=episode_i, is_show=False, is_save=True, delete_old=False, T=query_scale_trans)
            # show_save_attention_maps(attention_maps_l1, queries, support_kp_mask, query_kp_mask, episode_generator.support_kp_categories,
            #                          episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/attention_maps2', delete_old=False, T=query_scale_trans)

            # p_support_lateral_out = model.numerical_transformation(support_lateral_out, w_computing_mode=2)  # transform into positive number, w^-1
            # show_save_distinctive_maps(p_support_lateral_out, supports, heatmap_process_method=2, episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/distinctive_maps_s', delete_old=False)
            # p_query_lateral_out = model.numerical_transformation(query_lateral_out, w_computing_mode=2)  # transform into positive number, w^-1
            # show_save_distinctive_maps(p_query_lateral_out, queries, heatmap_process_method=2, episode_num=episode_i, is_show=False, is_save=True, save_image_root='./episode_images/distinctive_maps_q', delete_old=False)
            # inv_w_patches, w_patches = model.get_distinctiveness_for_parts(p_support_lateral_out, p_query_lateral_out, support_labels.float(), support_kp_mask, query_labels.float())
            # inv_w_patches = inv_w_patches.cpu().detach().numpy()


        # process attentive_feats
        if model.cascade_map_process == 'catch':
            attentive_feats = torch.cat(attentive_feats, dim=2)  # B2 x N x C' x H x W (C': catched feature channels)
            if using_interpolated_kps:
                attentive_feats_aux = torch.cat(attentive_feats_aux, dim=2)  # B2 x T x C' x H x W
        elif model.cascade_map_process == 'mean':
            attentive_feats = (sum(attentive_feats) / len(attentive_feats))
            if using_interpolated_kps:
                attentive_feats_aux = (sum(attentive_feats_aux) / len(attentive_feats_aux))
        else:
            print('Error for model.cascade_map_process!')
            exit(0)
        keypoint_descriptors = model.descriptor_net(attentive_feats)  # B2 x N x D  or B2 x N x c x h x w
        # if using_interpolated_kps and num_valid_support_aux_kps > 0:
        #     keypoint_descriptors_aux = model.descriptor_net(attentive_feats_aux)  # B2 x T x D or B2 x T x c x h x w


        if model.regression_type == 'direct_regression':
            B2 = query_kp_mask.shape[0]  # model.opts['M_query']
            N =  query_kp_mask.shape[1]  # model.opts['N_way']

            predictions = model.regressor(keypoint_descriptors)  # B2 x N x 2

            # Method 2, masking, using broadcast
            # ignore_mask = (~union_support_kp_mask.reshape(1, N).repeat(B2, 1)) + valid_kp_mask  # B2 x N
            # predictions2 = predictions * ignore_mask.unsqueeze(dim=2)
            predictions2 = predictions * (valid_kp_mask.unsqueeze(dim=2))
            query_labels2 = query_labels * (valid_kp_mask.unsqueeze(dim=2))

            predictions = predictions2
            query_labels = query_labels2

        elif model.regression_type == 'direct_regression_gridMSE':
            B2 = query_kp_mask.shape[0]  # model.opts['M_query']
            N = query_kp_mask.shape[1]  # model.opts['N_way']

            mean_predictions = 0
            # mean_predictions_grid = 0  # only used for testing grid classification
            scale_num = len(model.grid_length)
            mean_grid_var = 0
            multi_scale_covar = []
            multi_scale_predictions = []
            multi_scale_grid_scores = []
            ms_o_grids = []
            ms_o_offsets = []
            ms_o_rhos = []
            for scale_i, grid_length in enumerate(model.grid_length):
                if model.opts['eval_compute_var'] == 0:  # don't compute var or covar
                    predict_gridsO, predict_deviations, _, _= model.regressor[scale_i](keypoint_descriptors, training_phase=False)  # B2 x N x (grid_length ** 2), B2 x N x 2
                else:  # True
                    predict_gridsO, predict_deviations, rho, grid_var, offsetsO, rhoO= model.regressor[scale_i].test(keypoint_descriptors, training_phase=False, compute_grid_var=True)  # B2 x N x (grid_length ** 2), B2 x N x 2



                # compute predicted keypoint locations using grids and deviations
                grid_scores, predict_grids = torch.max(predict_gridsO, dim=2)  # B2 x N, 0 ~ grid_length * grid_length - 1
                predict_gridxy = torch.FloatTensor(B2, N, 2).cuda()
                predict_gridxy[:, :, 0] = predict_grids % grid_length  # grid x
                predict_gridxy[:, :, 1] = predict_grids // grid_length # grid y

                # Method 1, deviation range: -1~1
                predictions = (((predict_gridxy + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
                # Method 2, deviation range: 0~1
                # predictions = ((predict_gridxy + predict_deviations) / grid_length - 0.5) * 2
                mean_predictions += predictions
                # mean_predictions_grid += ((predict_gridxy + 0.5) / grid_length - 0.5) * 2  # only used for testing grid classification
                multi_scale_predictions.append(predictions)
                multi_scale_grid_scores.append(grid_scores)
                ms_o_grids.append(predict_gridsO)
                ms_o_offsets.append(offsetsO)
                ms_o_rhos.append(rhoO)
                if model.opts['eval_compute_var'] == 1:  # compute uncorrelated var, B x N x 2
                    # compute variance for regression
                    # reg_var   = torch.exp(rho) / torch.tensor(np.pi*2)  # sigma^2, variance for Gaussian, rho = log(2PI*vairance)
                    reg_var = (torch.exp(rho.cpu().detach()) ** 2) / 2.0  # 2b^2, variance for laplacian, rho = log(2b)
                    # mean_grid_var += ((square_image_length / grid_length)**2) * grid_var
                    mean_grid_var += ((square_image_length / grid_length) ** 2) * (grid_var+ reg_var / 4)  # deviation regression range -1~1, therefore divided by 2^2=4 to transform to 0~1
                elif model.opts['eval_compute_var'] == 2:  # compute covar, B x N x (2d); note for covar's case grid_var == None
                    d = rho.shape[2] // 2  # covariance latent Q: 2 x d
                    Q = rho.cpu().detach().reshape(B2, N, 2, d)
                    reg_var = torch.zeros(B2, N, 2, 2, requires_grad=False)
                    for i in range(B2):
                        for j in range(N):
                            q = Q[i, j]
                            omega = q.matmul(q.permute(1, 0)) / d  # precision matrix inv(Covar)
                            # reg_var[i, j] = torch.inverse(omega)  # 2 x 2, Covar. Note torch.inverse() is too slow!!!
                            v00, v11 = omega[0, 0], omega[1, 1]
                            v01 = omega[0, 1]
                            det = v00 * v11 - v01 ** 2 + 1e-12
                            reg_var[i, j, 0, 0] = v11 / det
                            reg_var[i, j, 1, 1] = v00 / det
                            reg_var[i, j, 0, 1] = -v01 / det
                            reg_var[i, j, 1, 0] = reg_var[i, j, 0, 1]

                    covar_temp = ((square_image_length / grid_length) ** 2) * (reg_var / 4)
                    mean_grid_var += covar_temp
                    multi_scale_covar.append(covar_temp)

            mean_predictions /= scale_num
            # mean_predictions_grid /= scale_num  # only used for testing grid classification
            if model.opts['eval_compute_var'] != 0:
                mean_grid_var   /= scale_num

            # predictions = predict_gridxy
            # # compute grid groundtruth and deviation
            # gridxy = (query_labels /2 + 0.5) * model.grid_length  # coordinate -1~1 --> 0~model.grid_length, B2 x N x 2
            # gridxy_quantized = gridxy.long().clamp(0, model.grid_length - 1)  # B2 x N x 2
            # query_labels = gridxy_quantized  # using the quantized gridxy as query labels


            # predictions2 = predictions * (valid_kp_mask.unsqueeze(dim=2))
            predictions = mean_predictions * (valid_kp_mask.unsqueeze(dim=2))
            query_labels = query_labels * (valid_kp_mask.unsqueeze(dim=2))
            # predictions_grid = mean_predictions_grid * (valid_kp_mask.unsqueeze(dim=2))  # only used for testing grid classification

            # predictions = predictions2
            # query_labels = query_labels2

        def sample_negative_kp_repres(nkps=3):
            from utils import sample_points_within_rect_but_distant_to_anchors, sample_points_via_interpolate_but_distant_to_anchors
            curr_s_bbx = s_bbx_origin * (support_scale_trans[:, 0]).view(-1, 1)  # B x 4, each row (xmin, ymin, w, h)
            curr_s_bbx[:, 0:2] = curr_s_bbx[:, 0:2] - support_scale_trans[:, 1:3]  # get bbx on squared image
            support_labels_np = ((support_labels/2+0.5) * square_image_length).cpu().numpy()  # B x N x 2, uppport kps on squared image
            support_kp_mask_np = support_kp_mask.cpu().numpy()
            im_index = 0
            anchors = support_labels_np[im_index]  # single image's keypoints
            anchor_mask = support_kp_mask_np[im_index]
            anchor_bbx = curr_s_bbx[im_index]
            longer_edge = anchor_bbx[2] if anchor_bbx[2] >= anchor_bbx[3] else anchor_bbx[3]
            dist_thresh = (longer_edge * 0.1).item()  # 10 pixels
            sample_success_flag, sampled_kps = sample_points_within_rect_but_distant_to_anchors(nkps, anchor_bbx, 'random', anchors, anchor_mask, dist_thresh=dist_thresh)
            # sample_success_flag, sampled_kps = sample_points_via_interpolate_but_distant_to_anchors(3, anchors, anchor_mask, dist_thresh=10)

            # if sample_success_flag:  # draw kps
            #     supports_temp = batch_images_normalize(supports.clone(), denormalize=True)
            #     supports_temp = (supports_temp * 255)
            #     supports_temp = supports_temp.permute(0, 2, 3, 1).cpu().detach().numpy().astype(np.uint8)
            #     im_tmp = supports_temp[im_index]
            #     im_tmp_brg = im_tmp[:, :, [2,1,0]]
            #     image = np.zeros(im_tmp_brg.shape, dtype=np.uint8)
            #     image[:,:,:] = im_tmp_brg[:,:,:]
            #     keypoint_dict = {i:sampled_kps[i] for i in range(len(sampled_kps))}
            #     new_im = draw_markers(image, keypoint_dict, marker='circle', color=[255,255,255], circle_radius=5, thickness=1)
            #     keypoint_dict2 = {i:anchors[i] for i in range(len(anchors)) if anchor_mask[i] == 1}
            #     new_im = draw_markers(new_im, keypoint_dict2, marker='circle', color=[0,0,255], circle_radius=5, thickness=1)
            #     visible_bounds = curr_s_bbx[0]  # (xmin, ymin, w, h)
            #     cv2.rectangle(new_im, (int(visible_bounds[0]), int(visible_bounds[1])),
            #           (int(visible_bounds[0] + visible_bounds[2]), int(visible_bounds[1] + visible_bounds[3])), (0, 255, 0))
            #     cv2.imwrite('view{}.png'.format(str(13)), new_im)

            sampled_feat_map = (s_feats_select[0][im_index]).unsqueeze(0)  # 1 x C x H x W
            sampled_kps = torch.tensor((sampled_kps*1.0/square_image_length-0.5)*2).unsqueeze(0).cuda()    # 1 x nkps x 2, -1~1
            sampled_kp_mask = torch.ones((1, nkps)).cuda()
            neg_kp_repres, _ = extract_kp_repres_func(features=sampled_feat_map, labels=sampled_kps, kp_mask=sampled_kp_mask)  # 1 x C x nps
            sampled_kp_mask = sampled_kp_mask.reshape(nkps)  # nkps
            neg_kp_repres = neg_kp_repres.reshape(-1, nkps)  # C x nkps

            return sample_success_flag, sampled_kps, sampled_kp_mask, neg_kp_repres


        mean_grid_scores = 0  # used to compute stat. avg scores for all correctly predicted kps
        for i in range(scale_num):
            mean_grid_scores += multi_scale_grid_scores[i].exp()  # use exp(.) because original value is from logsoftmax
        mean_grid_scores = (mean_grid_scores / scale_num)  * valid_kp_mask  # B2 x N
        predictionsT = model.iter_transductive(multi_scale_predictions, multi_scale_grid_scores, \
                    s_feats_select[0], q_feats_select[0], support_labels, support_kp_mask, query_labels, valid_kp_mask, extract_kp_repres_func, \
                    ms_o_grids=ms_o_grids, ms_o_offsets=ms_o_offsets, ms_o_rhos=ms_o_rhos, sample_neg_kp_func=sample_negative_kp_repres)
        if eval_throughput:
            num_measured_kps = valid_kp_mask.shape[0] * valid_kp_mask.shape[1]
            throughput_meter.update((time.time() - throughput_t_start) / num_measured_kps)  # sec / kp

        predictionsT = predictionsT.cpu().detach()

        predictions = predictions.cpu().detach()
        # predictions = gp_predictions.cpu().detach()
        # predictions_grid = predictions_grid.cpu().detach()  # only used for testing grid classification
        query_labels = query_labels.cpu().detach()

        # version = 'preprocessed', 'original'
        # show_save_predictions(queries, query_labels, valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=None,
        #                       version='original', is_show=False, is_save=True, folder_name='query_gt', save_file_prefix='eps{}'.format(episode_i))
        # show_save_predictions(queries, predictions, valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=mean_grid_var, confidence_scale=3,
        #                       version='original', is_show=False, is_save=True, folder_name='query_predict', save_file_prefix='eps{}'.format(episode_i), kp_labels_gt=query_labels)
        # # multi-scale covar & predictions
        # for scale_i, grid_length in enumerate(model.grid_length):
        #     show_save_predictions(queries, multi_scale_predictions[scale_i].cpu().detach(), valid_kp_mask, query_scale_trans, episode_generator, square_image_length, kp_var=multi_scale_covar[scale_i], confidence_scale=3,
        #                       version='original', is_show=False, is_save=True, folder_name='query_predict_scale%d'%(grid_length), save_file_prefix='eps{}'.format(episode_i), kp_labels_gt=query_labels)

        # show_save_warp_images(queries, query_labels, predictions, valid_kp_mask, query_scale_trans, support_labels.cpu().detach(), support_scale_trans, episode_generator,
        #                       square_image_length, version='original', is_show=False, is_save=True, save_file_prefix=episode_i)  # 'original' or 'preprocessed'
        # show_save_warp_images(queries, query_labels, predictions, valid_kp_mask, query_scale_trans, support_labels.cpu().detach(), support_scale_trans, episode_generator,
        #                       square_image_length, kp_var=mean_grid_var, confidence_scale=3, version='original', is_show=False, is_save=True, save_file_prefix=episode_i)  # 'original' or 'preprocessed'

        B2 = query_kp_mask.shape[0]  # model.opts['M_query']
        N = query_kp_mask.shape[1]  # model.opts['N_way']

        # square distance diff in original image scale
        predictions_o = recover_kps(predictions, square_image_length, query_scale_trans)
        query_labels_o = recover_kps(query_labels, square_image_length, query_scale_trans)
        square_diff = torch.sum((predictions_o - query_labels_o) ** 2, dim=2).cpu().detach().numpy()  # B2 x M
        # square_diff2 = torch.sum((predictions / 2 - query_labels /2) ** 2, dim=2).cpu().detach().numpy()  # B2 x M

        predictionsT_o = recover_kps(predictionsT, square_image_length, query_scale_trans)
        square_diff2 = torch.sum((predictionsT_o - query_labels_o) ** 2, dim=2).cpu().detach().numpy()  # B2 x M

        if pck_thresh_type == 'bbx':
            longer_edge = np.max(query_bbx_origin[:, [2, 3]].numpy(), axis=1)  # B2, query_bbx_origin's format xmin, ymin, w, h
        else:  # == 'img'
            longer_edge = np.max(query_w_h_origin.numpy(), axis=1)
        longer_edge = longer_edge.reshape(-1, 1)  # B2 x 1

        # square_diff2 = torch.sum((predictions_grid / 2 - query_labels / 2) ** 2, 2).cpu().detach().numpy()
        # correct += np.sum(square_diff < sqaure_error_tolerance)
        # total += query_labels.shape[0] * query_labels.shape[1] # equal to opts['M_query'] * opts['N_way']
        result_mask = valid_kp_mask.cpu().detach().numpy().astype(bool)
        if eval_method == 'method1':
            for ind, thr in enumerate(pck_thresh):
                judges = (square_diff <= (thr * longer_edge)**2)
                # judges = (square_diff2 < (thr ) ** 2)

                # compute statistical mean grid score as a criterion for transductive FSKD
                scores_mask = (judges * result_mask).astype(np.float)
                curr_scores=np.sum(mean_grid_scores.cpu().detach().numpy() * scores_mask) / (np.sum(scores_mask)+1e-9)
                scores_list[ind].append(curr_scores)

                # judges3 = (judges[:, 0:2]).reshape(-1)
                # result_mask3 = result_mask[:, 0:2]
                # judges3 = judges3[result_mask3.reshape(-1)]
                # tps3[ind].extend(judges3)
                # fps3[ind].extend(1 - judges3)
                # judges4 = (judges[:, 2:]).reshape(-1)
                # result_mask4 = result_mask[:, 2:]
                # judges4 = judges4[result_mask4.reshape(-1)]
                # tps4[ind].extend(judges4)
                # fps4[ind].extend(1-judges4)

                judges = judges.reshape(-1)
                # masking
                judges = judges[result_mask.reshape(-1)]
                tps[ind].extend(judges)
                fps[ind].extend(1 - judges)

                acc_cur = np.sum(judges) / len(judges)
                acc_list[ind].append(acc_cur)

                # used in transductive inference
                judges2 = (square_diff2 <= (thr * longer_edge)**2).reshape(-1)
                judges2 = judges2[result_mask.reshape(-1)]
                tps3[ind].extend(judges2)
                fps3[ind].extend(1-judges2)
                acc_cur2 = np.sum(judges2) / len(judges2)
                acc_list2[ind].append(acc_cur2)

            if (episode_i % 20 == 0 or episode_i == (num_test_episodes - 1)) and len(tps[0]) > 0:
                # recall, AP = compute_recall_ap(tps, fps, len(tps[0]))
                acc_mean, interval = mean_confidence_interval_multiple(acc_list)
                acc_mean2, interval2 = mean_confidence_interval_multiple(acc_list2)
                scores_mean, _ = mean_confidence_interval_multiple(scores_list)


                # recall3, AP3 = compute_recall_ap(tps3, fps3, len(tps3[0]))
                # recall4, AP4 = compute_recall_ap(tps4, fps4, len(tps4[0]))

        if eval_method == 'method2':
            # eval method 2, using keypoint similarity based AP
            ks = np.zeros([B2, N])
            # retrieve ks_sigmas based on the dynamic support_kp_categories (which can be allowed to change in each iteration)
            specified_kp_ids = [KEYPOINT_TYPE_IDS[kp_type] for kp_type in episode_generator.support_kp_categories]
            specified_ks_sigmas = ks_sigmas[specified_kp_ids]
            k_var = (specified_ks_sigmas * 2) ** 2  # N
            bbx_areas = (query_bbx_origin[:, 2] * query_bbx_origin[:, 3]).numpy()  # B2, query_bbx_origin's format xmin, ymin, w, h
            bbx_areas = bbx_areas + np.spacing(1)
            e = square_diff / (2.0 * bbx_areas.reshape(-1, 1) * k_var.reshape(1, -1))
            ks = np.exp(-e)  # ks = exp(- d^2/ (2 * s^2 * k^2) )

            for ind, thr in enumerate(ks_thresh):
                judges = (ks >= thr).reshape(-1)
                # masking
                judges = judges[result_mask.reshape(-1)]
                tps2[ind].extend(judges)
                fps2[ind].extend(1 - judges)

                acc_cur = np.sum(judges) / len(judges)
                acc_list[ind].append(acc_cur)

            if (episode_i % 20 == 0 or episode_i == (num_test_episodes - 1)) and len(tps2[0]) > 0:
                # recall2, AP2 = compute_recall_ap(tps2, fps2, len(tps2[0]))
                acc_mean, interval = mean_confidence_interval_multiple(acc_list)

                # ------
                # redundancy, just for below printing and clarification that this is another eval method
                # recall, AP = recall2, AP2
                # tps = tps2
                # ------

        # ----------------------------------------------------------------------------
        # used to compute relationship between normalized distance error and localization uncertainty
        # np_mean_grid_var = mean_grid_var.numpy()
        # d_error = np.sqrt(square_diff) / longer_edge
        # count_mask = (d_error < d_norm_max) * result_mask  # B x N
        # for ii in range(B2):
        #     for jj in range(N):
        #         if count_mask[ii, jj] == False:
        #             continue
        #         bin_i = int(d_error[ii, jj] / d_increment)
        #         d_uc_bins[bin_i, 0] += d_error[ii, jj]
        #         uc_tr = np_mean_grid_var[ii, jj, 0, 0] + np_mean_grid_var[ii, jj, 1, 1]
        #         uc_det = np_mean_grid_var[ii, jj, 0, 0]*np_mean_grid_var[ii, jj, 1, 1] - np_mean_grid_var[ii, jj, 0, 1]*np_mean_grid_var[ii, jj, 1, 0]
        #         uc_energy = 3 * np.sqrt(uc_tr + 2*np.sqrt(uc_det)) / longer_edge[ii, 0]
        #         d_uc_bins[bin_i, 1] += uc_energy
        #         d_uc_bins[bin_i, 3] += 1
        #         d_uc_bins[bin_i, 2] += inv_w_patches[ii, jj]  # semantic uncertainty
        # ----------------------------------------------------------------------------

        if episode_i % 20 == 0 or episode_i == num_test_episodes-1:
            # sum_tps = np.sum(np.array(tps), axis=1)
            # in order to display results property, like 200/200
            if episode_i == (num_test_episodes - 1):
                episode_i = episode_i + 1
            if model.regression_type == 'direct_regression'  or model.regression_type == 'direct_regression_gridMSE' or model.regression_type == 'multiple_regression' or model.regression_type=='composite_regression':
                print('episode {}/{}, Acc {}, Int. {}, time: {}'.format(episode_i, num_test_episodes, acc_mean, interval, datetime.datetime.now()))
                print('episode {}/{}, Acc {}, Int. {}, time: {}'.format(episode_i, num_test_episodes, acc_mean2, interval2, datetime.datetime.now()))
                print('episode {}/{}, Avg score {}'.format(episode_i, num_test_episodes, scores_mean))
                # if eval_method == 'method1':
                #     print('episode {}/{}, Acc {}, AP {}'.format(episode_i, num_test_episodes, recall3, AP3))
                #     print('episode {}/{}, Acc {}, AP {}'.format(episode_i, num_test_episodes, recall4, AP4))

        # increment in episode_i
        episode_i += 1

    if eval_method == 'method2':  # redundancy, just for below printing and clarification that this is another eval method
        tps, fps = tps2, fps2
    sum_tps = np.sum(np.array(tps), axis=1)
    recall, AP = compute_recall_ap(tps, fps, len(tps[0]))
    print('episode {}/{}, Acc2 {}, AP2 {}, {}/{}, time: {}'.format(num_test_episodes, num_test_episodes, recall, AP, sum_tps, len(tps[0]), datetime.datetime.now()))

    sum_tps2 = np.sum(np.array(tps3), axis=1)
    recall2, AP2 = compute_recall_ap(tps3, fps3, len(tps3[0]))
    print('episode {}/{}, Acc2 {}, AP2 {}, {}/{}, time: {}'.format(num_test_episodes, num_test_episodes, recall2, AP2, sum_tps2, len(tps3[0]), datetime.datetime.now()))
    if eval_throughput:
        throughput0 = throughput_meter0.avg
        throughput = throughput_meter.avg
        print('Inference throughput over %s episodes: %s sec/im, %s sec/kp'%(num_test_episodes, throughput_meter0.avg, throughput_meter.avg))
    print('==============testing end================')
    if model.logging != None and episode_i == num_test_episodes-1:
        logging.info('==============testing start==============')
        if model.regression_type == 'direct_regression' or model.regression_type == 'direct_regression_gridMSE' or model.regression_type == 'multiple_regression' or model.regression_type == 'composite_regression':
            logging.info('episode {}/{}, Acc {}, AP {}, {}/{}, time: {}'.format(episode_i,
                num_test_episodes, recall, AP, sum_tps, len(tps[0]), datetime.datetime.now()))
        logging.info('==============testing end===============')

    torch.set_grad_enabled(True)  # enable grad computation
    if model.opts['set_eval']:
        model.set_train_status()

    # # ----------------------------------------------------------------------------
    # fig = plt.figure(111)
    # uc_data = np.copy(d_uc_bins)
    # uc_data[:, [0, 1]] /= uc_data[:, 3].reshape(-1, 1)  # compute average
    # print("bin count: ", uc_data[:, 3])
    # plt.plot(uc_data[:, 0], uc_data[:, 1], marker='.', color='black')
    # ax = fig.get_axes()
    # ax[0].set_xticks(np.linspace(0, d_norm_max, N_bins+1))
    # plt.xlim([0, d_norm_max])
    # plt.grid(True)
    # # plt.show()
    # plt.savefig('d-uc.pdf', bbox_inches='tight')
    # # np.save('d-uc.npy', d_uc_bins)
    # # ----------------------------------------------------------------------------

    # save datapoints
    # for key in model.datapoint:
    #     model.datapoint[key] = torch.stack(model.datapoint[key], dim=0)
    # save_path = os.path.join('./transductive-data/datapoint_four2dog_base2.pt')
    # torch.save(model.datapoint, save_path)
    # print('mean: ', sum(model.datapoint_list1)*1.0 / (len(model.datapoint_list1)+1e-6))
    # print('mean: ', sum(model.datapoint_list2) * 1.0 / (len(model.datapoint_list1)+1e-6))

    return acc_mean, interval, recall, AP, acc_mean2, interval2, recall2, AP2, throughput0, throughput #, d_uc_bins


def iter_transductive(model, ms_preds, ms_grid_scores, s_feats, q_feats, support_kps, support_kp_mask, query_kps, valid_kp_mask, extract_kp_repres_func, **nargs):
    ms_o_grids = nargs['ms_o_grids']
    ms_o_offsets = nargs['ms_o_offsets']
    ms_o_rhos = nargs['ms_o_rhos']
    square_image_length = model.opts['square_image_length']
    scale_num = len(ms_o_grids)
    top_k_grids = int(model.opts['tfskd_config'][0])   # 3
    sample_z_kps = int(model.opts['tfskd_config'][1])  # 0
    strategy = int(model.opts['tfskd_config'][2])  # 0
    refine_method = int(model.opts['tfskd_config'][3])  # 0
    B2, N = valid_kp_mask.shape

    num_iter = int(model.opts['transductive'][0])
    ratio = [1 / 8, 1 / 3]  # 1/8, 1/3
    growth = model.opts['transductive'][1:]  # [3, 2]
    score_threshs = [0.8]

    fused_attention = model.opts['use_fused_attention']
    # num_valid_kps_per_type = torch.sum(valid_kp_mask, dim=0)  # N

    def process_preds(ms_o_grids, ms_o_offsets, ms_o_rhos, topk=1, samplez=0):
        '''
        topk: top-k grids with highest probability
        samplez: sample z points with N(x; u, Sigma)
        mean_kps: B x N x k x 2  (k=topk + samplez, each kp expand k additional kps)
        mean_scores: B x N x k
        mean_uc_energy: B x N x k
        '''
        from torch.distributions.multivariate_normal import MultivariateNormal, _batch_mahalanobis

        # tp1=time.time()

        scale_num = len(ms_o_grids)
        mean_kps = 0        # B x N x k x 2
        mean_scores = 0     # B x N x k
        ms_covar = []
        for i in range(len(ms_o_grids)):
            B2, N, L_square = ms_o_grids[i].shape
            L = int(np.sqrt(L_square))
            grid_length = L

            predict_gridsO = ms_o_grids[i]  # B x N x (L*L)
            predict_deviationsO = ms_o_offsets[i]  # (B*N*L*L) x 2
            predict_rhosO = ms_o_rhos[i]  # (B*N) x (L*L*2d)
            predict_rhosO = predict_rhosO.view(B2 * N * L * L, -1)  # (B*N*L*L) x (2d)

            grid_scores, predict_grids = torch.topk(predict_gridsO, k=topk, dim=2)  # B x N x k
            grid_scores = grid_scores.exp()
            # retrieve offsets and uncertainty according to topk grids
            retrieval_index = predict_grids.view(B2*N, topk)  # (B*N) x k
            ind_temp = torch.linspace(0, B2 * N - 1, B2 * N).long().view(-1).cuda()
            predict_deviations = []
            rho = []
            for t in range(topk):
                retrieval_index_k = retrieval_index[:, t]
                offsets_temp = predict_deviationsO[ind_temp * L* L + retrieval_index_k, :]  # (B*N) * 2
                rhos_temp = predict_rhosO[ind_temp * L* L + retrieval_index_k, :]  # (B*N) * (2 * d)
                predict_deviations.append(offsets_temp)
                rho.append(rhos_temp)
            predict_deviations = torch.cat(predict_deviations, dim=1).reshape(B2, N, topk, 2)
            rho = torch.cat(rho, dim=1).reshape(B2*N, topk, -1)  # (B*N) * k * (2 * d)

            # process offsets
            predict_gridxy = torch.FloatTensor(B2, N, topk, 2).cuda()
            predict_gridxy[:, :, :, 0] = predict_grids % grid_length   # grid x
            predict_gridxy[:, :, :, 1] = predict_grids // grid_length  # grid y
            predictions = (((predict_gridxy + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
            mean_kps += predictions     # B x N x k x 2
            mean_scores += grid_scores  # B x N x k

            # process uncertainty, i.e., covariance matrix
            d = rho.shape[-1] // 2
            Q = rho.detach().reshape(B2*N, topk, 2, d)
            reg_var = torch.zeros(B2*N, topk, 2, 2, requires_grad=False).cuda()
            for s in range(B2*N):
                for t in range(topk):
                    q = Q[s, t]
                    omega = q.matmul(q.permute(1, 0)) / d  # precision matrix inv(Covar)
                    # reg_var[s, t] = torch.inverse(omega)  # 2 x 2, Covar. Note torch.inverse() is too slow!!!
                    v00, v11 = omega[0, 0], omega[1, 1]
                    v01 = omega[0, 1]
                    det = v00*v11 - v01**2 + 1e-12
                    reg_var[s, t, 0, 0] =  v11 / det
                    reg_var[s, t, 1, 1] =  v00 / det
                    reg_var[s, t, 0, 1] = -v01 / det
                    reg_var[s, t, 1, 0] = reg_var[s, t, 0, 1]

            # covar_temp = ((square_image_length / grid_length) ** 2) * (reg_var / 4)
            covar_temp = ((2 / grid_length) ** 2) * (reg_var / 4)  # use 2 because kp is in range -1~1
            ms_covar.append(covar_temp)

        # tp2 = time.time()
        # print('tp1: ', tp2-tp1)

        mean_kps /= scale_num     # B x N x k x 2
        mean_scores /= scale_num  # B x N x k
        mean_covar = sum(ms_covar) / scale_num  # (B*N) x k x 2 x 2
        # process uncertainty energy J = ab*\pi (note this UC is subject to image size 384)
        B2, N = mean_kps.shape[:2]
        mean_uc_energy = torch.zeros(B2*N, topk, requires_grad=False).cuda()
        confidence_scale = 3
        for s in range(B2 * N):
            for t in range(topk):
                single_kp_covar = mean_covar[s, t]  # 2 x 2
                # e, _, angle = compute_eigenvalues(single_kp_covar)
                # stds = torch.sqrt(e[:, 0])  # sqrt(major_axis, minor_axis)
                # mean_uc_energy[s, t] = (confidence_scale ** 2) * stds[0] * stds[1] * 3.1415926  # (3a)(3b)*\pi
                v00, v11 = single_kp_covar[0, 0], single_kp_covar[1, 1]
                v01 = single_kp_covar[0, 1]
                det = v00 * v11 - v01**2  # == (std0^2 * std1^2)
                trace = v00 + v11  # == (std0^2 + std1^2)
                # mean_uc_energy[s, t] = (confidence_scale ** 2) * torch.sqrt(det) * 3.1415926  # (3a)(3b)*\pi
                mean_uc_energy[s, t] = confidence_scale * (trace + 2*det.sqrt()).sqrt()  # 3(a+b)
        mean_uc_energy = mean_uc_energy.reshape(B2, N, topk)  # B x N x k

        # tp3 = time.time()
        # print('tp2: ', tp3 - tp2)

        # here we add sample z additional kps (range -1~1)
        if samplez > 0:
            sampled_kps = torch.zeros(B2 * N, samplez, 2, requires_grad=False)
            loc_u = mean_kps[:, :, 0, :].view(B2 * N, 2)  # (B2*N) x 2, top-1 kps
            loc_cov_mat = mean_covar[:, 0, :, :]  # (B2*N) x 2 x 2
            distrib = MultivariateNormal(loc=loc_u, covariance_matrix=loc_cov_mat)
            sampled_kps = distrib.sample(sample_shape=[samplez])  # samplez x (B2*N) x 2, sample a batch of kps
            # ------------------------------------
            # compute unnormalized density exp(-0.5*(x-u)^T*Sigma^{-1}*(x-u)) = exp(-0.5 * M) (normalized density may >1)
            # below code is borrowed and modified from distrib.log_prob(model, value)
            diff = sampled_kps - loc_u.unsqueeze(0)  # samplez x (B2*N) x 2
            M = _batch_mahalanobis(distrib._unbroadcasted_scale_tril, diff)
            density = torch.exp(-0.5 * M)  # samplez x (B2*N)
            # ------------------------------------
            sampled_kps = sampled_kps.permute(1, 0, 2).reshape(B2, N, samplez, 2)  # B x N x samplez x 2
            # construct uncertainty and scores for sampled kps
            density = density.permute(1, 0).reshape(B2, N, samplez)  # B x N x samplez
            sampled_kps_scores = (mean_scores[:, :, 0]).clone().reshape(B2, N, 1) * density  # use density to down-weight the scores for sampled kps
            sampled_kps_uc_energy = (mean_uc_energy[:, :, 0]).clone().reshape(B2, N, 1).expand(B2, N, samplez)

            # combine topk grid points and sampled points
            mean_kps = torch.cat([mean_kps, sampled_kps], dim=2)  # B x N x (k+samplez) x 2
            mean_scores = torch.cat([mean_scores, sampled_kps_scores], dim=2)  # B x N x (k+samplez)
            mean_uc_energy = torch.cat([mean_uc_energy, sampled_kps_uc_energy], dim=2)  # B x N x (k+samplez)

        # tp4 = time.time()
        # print('tp3: ', tp4 - tp3)

        return mean_kps, mean_scores, mean_uc_energy, mean_covar.reshape(B2, N, topk, 4)

    support_repres, _ = extract_kp_repres_func(features=s_feats, labels=support_kps, kp_mask=support_kp_mask)  # B1 x C x N

    # t0 = time.time()

    # 0) expand pseudo label set (select top 2 or 3 grids per keypoint + sample z kps around top-k kp)
    # B2 x N x k x 2; B2 x N x k; B2 x N x k; B2 x N x k x 2 x 2
    curr_preds, scores, curr_ucs, curr_covar = process_preds(ms_o_grids, ms_o_offsets, ms_o_rhos, topk=top_k_grids, samplez=sample_z_kps)
    k = scores.shape[-1]  # each kp expanded k kps (note k may not be same to top_k_grids), which depends sampling/selecting
    candidate_kp_mask = valid_kp_mask.unsqueeze(-1).repeat(1, 1, k)  # B2 x N x k
    scores = scores * candidate_kp_mask  # masking before sorting candidate kps
    curr_ucs = curr_ucs * candidate_kp_mask  # masking before sorting candidate kps
    curr_preds = curr_preds * candidate_kp_mask.unsqueeze(-1)

    # scale_num = len(ms_preds)
    # curr_preds = (sum(ms_preds) / scale_num) * (valid_kp_mask.unsqueeze(dim=2))  # B2 x N x 2, our kp's range is -1~1
    # scores = 0
    # for i in range(scale_num):
    #     scores += ms_grid_scores[i].exp()  # use exp(.) because original value is from logsoftmax
    # scores = (scores / scale_num)  * valid_kp_mask  # B2 x N

    for t in range(num_iter):
        # scores = valid_kp_mask
        # curr_preds = query_kps

        # t1 = time.time()
        # print('0: ', t1-t0)



        # 1) compute ranking criterion
        if strategy == 0:
            rank_criterion = scores.clone()  # B2 x N x k, use scores as criterion
            # rank_criterion = (1 / (curr_ucs + 1e-12)).clone()  # B2 x N x k, use uc as criterion
            # rank_criterion = (scores * 1 / (curr_ucs + 1e-12)).clone()  # B2 x N x k
        elif strategy == 1:
            strategy1_w = 0.5
            if t == 0:  # initial iteration
                SKPs = average_representations2(support_repres, support_kp_mask)  # C x N
            else:
                SKPs = avg_support_repres
            curr_preds_tmp = curr_preds.reshape(B2, N*k, 2)
            preds_tmp_mask = candidate_kp_mask.reshape(B2, N*k)
            query_repres, _ = extract_kp_repres_func(features=q_feats, labels=curr_preds_tmp, kp_mask=preds_tmp_mask)  # B2 x C x (N*k)
            query_repres = query_repres.reshape(B2, -1, N, k)
            SKPs_norm = torch.nn.functional.normalize(SKPs, p=2, dim=0, eps=1e-12)  # C x N
            query_repres_norm = torch.nn.functional.normalize(query_repres, p=2, dim=1, eps=1e-12)  # B2 x C x N x k
            simi = (SKPs_norm.unsqueeze(0).unsqueeze(-1)*query_repres_norm).sum(1)  # B2 x N x k
            rank_criterion = strategy1_w * scores.clone() + (1-strategy1_w) * simi  # B2 x N x k
            # rank_criterion = scores.clone() * simi  # B2 x N x k

            # rank once (rank for expanded k grids)
            v_sort, v_inds = torch.sort(rank_criterion, dim=2, descending=True)  # sort for expanded kps
            curr_preds2 = torch.gather(curr_preds, dim=2, index=v_inds.unsqueeze(-1).expand_as(curr_preds))
            curr_preds = curr_preds2[:, :, 0, :].unsqueeze(2)  # B2 x N x 1 x 2
            # candidate_kp_mask2 = torch.gather(candidate_kp_mask, dim=2, index=v_inds)
            # candidate_kp_mask = candidate_kp_mask2[:,:,0].unsqueeze(2)  # B2 x N x 1
            candidate_kp_mask = candidate_kp_mask[:, :, 0].unsqueeze(2)
            rank_criterion = v_sort[:, :, 0].unsqueeze(2)  # B2 x N x 1
            k = 1


        # ---------------------------------------------------------
        # "mean" strategy
        # scores_sum = scores.sum(dim=2, keepdim=True) + 1e-12  # B x N x 1
        # curr_preds = ((curr_preds * scores.unsqueeze(dim=3)).sum(dim=2)/scores_sum).unsqueeze(2)  # B2 x N x 1 x 2
        # candidate_kp_mask = candidate_kp_mask.mean(dim=2, keepdim=True)  # B2 x N x 1
        # rank_criterion = scores_sum  # B2 x N x 1
        # k = 1
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # # A) store datapoints to train a selection classifier
        # curr_preds_tmp = curr_preds.reshape(B2, N*k, 2)
        # candidate_kp_mask_tmp = candidate_kp_mask.reshape(B2, N*k)
        # query_repres, _ = extract_kp_repres_func(features=q_feats, labels=curr_preds_tmp, kp_mask=candidate_kp_mask_tmp)  # B x C x (N*k)
        # query_repres = query_repres.permute(0, 2, 1).reshape(B2, N, k, -1)  # B x N x k x C
        # simi_scores = torch.zeros(B2, N, k, k).cuda()
        # for n in range(k):
        #     for m in range(k):
        #         feat_map_tmp1 = query_repres[:, :, n, :]
        #         feat_map_tmp2 = query_repres[:, :, m, :]
        #         simi_scores[:, :, n, m] = torch.cosine_similarity(feat_map_tmp1, feat_map_tmp2, dim=2)
        # model.datapoint['pred'].append(curr_preds.detach().cpu())
        # model.datapoint['gt'].append(query_kps.detach().cpu())
        # model.datapoint['scores'].append(scores.detach().cpu())
        # model.datapoint['uc'].append(curr_covar.detach().cpu())
        # model.datapoint['kp_mask'].append(candidate_kp_mask.detach().cpu())
        # model.datapoint['simi_scores'].append(simi_scores.detach().cpu())
        # break

        # # B) use trained classifier to select datapoint
        # import lightgbm as lgb
        # root = '/home/changsheng/KeypointDetectionWithFSL/FSKD-ViT/transductive-data'
        # data_path = os.path.join(root, 'datapoint_four2dog.pt')
        # save_path = os.path.join(root, 'select_four2dog_base.txt')
        # # 1) sort datapoint
        # scores_sort, inds = torch.sort(scores, dim=2, descending=True)  # B2 x N x k
        # curr_preds_sort = torch.gather(curr_preds, dim=2, index=inds.unsqueeze(-1).expand_as(curr_preds))  # B2 x N x k x 2
        # curr_ucs_sort = torch.gather(curr_ucs, dim=2, index=inds)  # B2 x N x k
        # candidate_kp_mask_sort = torch.gather(candidate_kp_mask, dim=2, index=inds)  # B2 x N x k
        # # 2) compute gt index
        # neg_dis = -((curr_preds_sort - query_kps.unsqueeze(dim=2)) ** 2).sum(dim=-1)  # B2 x N x k
        # gt_inds = torch.max(neg_dis, dim=2)[1]  # B2 x N
        # # 3) center datapoint
        # curr_preds_centered = curr_preds_sort - (curr_preds_sort[:, :, 0, :]).unsqueeze(dim=2)  # B2 x N x k x 2
        # # 4) get datapoints by masking
        # datapoints = torch.cat([curr_preds_centered, scores_sort.unsqueeze(-1)], dim=-1)  # B2 x N x k x d (d=2+1+...)
        # datapoints = datapoints.reshape(B2*N, -1)  # (B2*N) x (k*d)
        # gt_inds = gt_inds.reshape(-1)
        # mask = candidate_kp_mask_sort[:,:,0].reshape(-1).bool()
        # # if torch.sum(mask) == 0:
        # #         print('Error for no valid datapoints!')
        # # datapoints = datapoints[mask, :]  # n x d
        # # labels = gt_inds[mask]  # n ( each entry is within 0~(k-1) )
        #
        # booster_model = lgb.Booster(model_file=save_path)  # load model
        # y_prob = booster_model.predict(datapoints.detach().cpu().numpy())  # (B2*N) x k
        # y_prob = torch.tensor(y_prob).cuda()
        # y_max_prob, y_pred_inds = torch.max(y_prob, dim=1)  # (B2*N)
        # acc = torch.sum((gt_inds == y_pred_inds)[mask]) * 1.0 / sum(mask)
        # acc2 = torch.sum((gt_inds == 0)[mask]) * 1.0 / sum(mask)
        # print("acc: ", acc)
        # print('acc2: ', acc2)
        # model.datapoint_list1.append(acc)
        # model.datapoint_list2.append(acc2)
        #
        # y_pred_inds = y_pred_inds.reshape(B2, N, 1)
        # curr_preds = torch.gather(curr_preds_sort, dim=2, index=y_pred_inds.unsqueeze(-1).expand(B2, N, 1, 2))  # B2 x N x 1 x 2
        # candidate_kp_mask = candidate_kp_mask[:,:,0].unsqueeze(2)  # B2 x N x 1
        # # rank_criterion = y_max_prob.reshape(B2, N, 1)
        # # rank_criterion = torch.gather(scores_sort, dim=2, index=y_pred_inds)  # B2 x N x 1
        # rank_criterion = torch.gather(scores_sort, dim=2, index=y_pred_inds) * y_max_prob.reshape(B2, N, 1)  # B2 x N x 1
        # k = 1
        # ---------------------------------------------------------

        elif strategy == 9:
            # ---------------------------------------------------------
            # use "gt" to select closest datapoint among k datapoints
            neg_dis =  -((curr_preds - query_kps.unsqueeze(dim=2))**2).sum(dim=-1)  # B2 x N x k
            neg_dis_sort, neg_dis_inds = torch.sort(neg_dis, dim=2, descending=True)  # sort for expanded kps
            curr_preds2 = torch.gather(curr_preds, dim=2, index=neg_dis_inds.unsqueeze(-1).expand_as(curr_preds))
            curr_preds = curr_preds2[:, :, 0, :].unsqueeze(2)  # B2 x N x 1 x 2
            candidate_kp_mask2 = torch.gather(candidate_kp_mask, dim=2, index=neg_dis_inds)
            candidate_kp_mask = candidate_kp_mask2[:,:,0].unsqueeze(2)  # B2 x N x 1
            rank_criterion = neg_dis_sort[:, :, 0].unsqueeze(2)  # B2 x N x 1
            k = 1
            # ---------------------------------------------------------
        else:
            raise NotImplementedError

        rank_criterion = rank_criterion * candidate_kp_mask
        rank_criterion = rank_criterion.permute(0, 2, 1)  # B2 x k x N
        rank_criterion = rank_criterion.reshape(B2 * k, N)  # (B2*k) x N
        num_valid_kps_per_type=candidate_kp_mask.permute(0, 2, 1).reshape(B2 * k, N).sum(dim=0) # N
        num_valid_images_per_type = valid_kp_mask.sum(dim=0)  # N

        # t2 = time.time()
        # print('1: ', t2 - t1)

        # 2) select high-quality pseudo labels (selection strategy is imortant!)
        scores_sort, inds = torch.sort(rank_criterion, dim=0, descending=True)  # (B2 * k) x N
        select_indicator = torch.zeros(B2 * k, N).cuda()  # (B2 * k) x N

        # ----------------------------
        # selection strategy 1, top m where m is determined by (number of valid kps x ratio)
        # num_select_kps_per_type = (num_valid_kps_per_type * ratio[t]).long()  # N
        # for n in range(N):
        #     for m in range(num_select_kps_per_type[n]):
        #         select_indicator[inds[m, n], n] = 1
        # ----------------------------
        # selection strategy 2, top m, where m is fixed by list
        for n in range(N):
            for m in range(int(growth[t])):
                # select_indicator[inds[m, n], n] = 1
                if m <= 2:  # m=0, 1, 2
                    if (num_valid_images_per_type[n] >= 3 * (m+1)):
                        select_indicator[inds[m, n], n] = 1
                else:
                    if (num_valid_images_per_type[n] >= 2 * (m+1)):
                        select_indicator[inds[m, n], n] = 1

        # ----------------------------
        # selection strategy 3, top m where m is determined by (score_thresh)
        # for n in range(N):
        #     for m in range(B2):
        #         if scores[m, n] >= score_threshs[t]:
        #             select_indicator[m, n] = 1

        # ----------------------------

        select_indicator = select_indicator.reshape(B2, k, N)
        select_indicator = select_indicator.permute(0, 2, 1)  # B2 x N x k
        select_indicator = select_indicator * candidate_kp_mask  # guarantee the selected kps are valid
        # print(torch.sum(select_indicator))

        # t3 = time.time()
        # print('2: ', t3 - t2)

        # 3) add the pseudo kp labels to the set of support kp labels, and refine kp prototypes
        curr_preds_tmp = curr_preds.reshape(B2, N*k, 2)
        select_indicator_tmp = select_indicator.reshape(B2, N*k)
        query_repres, _ = extract_kp_repres_func(features=q_feats, labels=curr_preds_tmp, kp_mask=select_indicator_tmp)  # B x C x (N*k)
        query_repres = query_repres.permute(0, 2, 1).reshape(B2, N, k, -1)  # B x N x k x C
        query_repres = query_repres.permute(0, 2, 1, 3).reshape(B2*k, N, -1)  # (B * k) x N x C
        query_repres = query_repres.permute(0, 2, 1)  # (B * k) x C x N == n_kps x C x N
        # select_indicator = select_indicator * scores  # (optional) score-weighted pseudo labels, B x N x k
        select_indicator = select_indicator.permute(0, 2, 1).reshape(B2*k, N)  # (B * k) x N == n_kps x N
        # select_indicator = select_indicator / (select_indicator.max(dim=0, keepdim=True)[0] + 1e-12)  # (optional) score-weighted pseudo labels, B x N x k


        # context_mode = 'hard_fiber'
        # query_repres, _ = extract_representations(q_feats, curr_preds, select_indicator, context_mode, sigma=model.opts['sigma'], \
        #                     downsize_factor=model.opts['downsize_factor'], image_length=model.opts['square_image_length'], together_trans_features=None)
        # ----------------------------
        # refinement method0: simple averaging
        if refine_method == 0:
            combined_repres = torch.cat([support_repres, query_repres], dim=0)  # (B1 + n_kps) x C x N
            combined_kp_mask = torch.cat([support_kp_mask, select_indicator], dim=0)  # (B1 + n_kps) x N
            avg_support_repres = average_representations2(combined_repres, combined_kp_mask)  # C x N

        # refinement method1: soft assignment (each prototype uses all query keypoints for update)
        elif refine_method == 1:
            soft_assign_sigma = model.opts['tfskd_config'][4]  # control assignment, 0.04/0.05
            soft_assign_w = model.opts['tfskd_config'][5]  # control the importance between support repres and PL's repres, 0.8
            mask_far_away_points = False  # no using is better
            mask_far_away_points_weight = 1.0  # w * max_dist
            use_neg_kp = False  # use hard negative keypoints as fake SKP
            neg_kp_num = 20

            if t == 0:  # initial iteration
                SKPs = average_representations2(support_repres, support_kp_mask)  # C x N
            else:
                SKPs = avg_support_repres

            sum_support_mask = torch.sum(support_kp_mask, dim=0)  # N
            union_support_kp_mask = sum_support_mask > 0  # N
            if use_neg_kp:  # whether use negative keypoints
                sample_success_flag, neg_kps, neg_kp_mask, neg_kp_repres = nargs['sample_neg_kp_func'](neg_kp_num)
                if sample_success_flag:
                    SKPs = torch.cat([SKPs, neg_kp_repres], dim=1)  # C x (N + neg_num)
                    union_support_kp_mask = torch.cat([union_support_kp_mask, neg_kp_mask.bool()])  # (N + neg_num)
            softmax_mask = -(1-union_support_kp_mask.float())*100  # N

            candidate_repres = query_repres.permute(0, 2, 1).reshape(B2*k*N, -1)  # (B*k*N) x C
            candidate_mask = select_indicator.reshape(-1)  # (B*k*N)
            SKPs_norm = torch.nn.functional.normalize(SKPs, p=2, dim=0, eps=1e-12)  # C x N
            candidate_repres_norm = torch.nn.functional.normalize(candidate_repres, p=2, dim=1, eps=1e-12)  # (B*k*N) x C
            cos_distance = 1 - candidate_repres_norm.matmul(SKPs_norm)  # (B*k*N) x N
            soft_assign_dist = -cos_distance/(soft_assign_sigma**2)  # (B*k*N) x N
            soft_assign_prob = torch.softmax(soft_assign_dist+softmax_mask.unsqueeze(0), dim=1)  # masked softmax, (B*k*N) x N

            if use_neg_kp:  # whether use negative keypoints
                if sample_success_flag:
                    union_support_kp_mask = union_support_kp_mask[:N]
                    softmax_mask = softmax_mask[:N]
                    cos_distance = cos_distance[:, :N]
                    soft_assign_prob = soft_assign_prob[:, :N]

            # compute max distance between prototypes and masking far-away candidate_repres
            if True == mask_far_away_points:
                max_dist = -1
                for n in range(N-1):
                    for m in range(n+1, N):
                        if (union_support_kp_mask[n] == True) and (union_support_kp_mask[m] == True):
                            dist_temp = torch.sum(SKPs_norm[:,n]*SKPs_norm[:,m])
                            if dist_temp >= max_dist:
                                max_dist = dist_temp
                far_away_indicator = cos_distance < (max_dist*mask_far_away_points_weight)  # (B*k*N) x N
                soft_assign_prob = soft_assign_prob * far_away_indicator

            soft_assign_prob = soft_assign_prob * candidate_mask.unsqueeze(dim=1)  # (B*k*N) x N, filter not selected fibers
            candidate_repres = candidate_repres * candidate_mask.unsqueeze(dim=1)  # (B*k*N) x C
            sum_SKPs = (support_repres * support_kp_mask.unsqueeze(dim=1)).sum(dim=0)  # C x N
            sum_SKPs = sum_SKPs.permute(1, 0)  # N x C
            sum_assigned_repres = soft_assign_prob.permute(1,0).matmul(candidate_repres)  # N x C
            sum_assigned_probs = soft_assign_prob.sum(dim=0)  # N
            # N x C, refine using support keypoint repres (SKRs) + assigned_repres
            avg_support_repres = (soft_assign_w * sum_SKPs + (1-soft_assign_w) * sum_assigned_repres) / \
                                 (soft_assign_w * sum_support_mask + (1-soft_assign_w) * sum_assigned_probs + 1e-12).unsqueeze(dim=1)
            # N x C, refine using current prototypes (SKPs) + assigned_repres
            # avg_support_repres = (soft_assign_w * SKPs.permute(1, 0) + (1-soft_assign_w) * sum_assigned_repres) / \
            #                      (soft_assign_w * union_support_kp_mask.float() + (1-soft_assign_w) * sum_assigned_probs + 1e-12).unsqueeze(dim=1)
            avg_support_repres = avg_support_repres.permute(1, 0)  # C x N

        # refinement method2: soft assignment (each prototype uses each type of query keypoints for update)
        elif refine_method == 2:
            soft_assign_sigma = model.opts['tfskd_config'][4]  # control assignment, 0.04/0.05
            soft_assign_w = model.opts['tfskd_config'][5]  # control the importance between support repres and PL's repres, 0.8
            use_neg_kp = False  # use hard negative keypoints as fake SKP
            neg_kp_num = 10

            if (fused_attention is not None) and ('spatial-suc' in fused_attention):  # firstly normalize; secondly average
                support_repres = torch.nn.functional.normalize(support_repres, p=2, dim=1, eps=1e-12)  # B1 x C x N
                query_repres = torch.nn.functional.normalize(query_repres, p=2, dim=1, eps=1e-12)  # (B * k) x C x N == n_kps x C x N
            else:  # channel attention or pure spatial attention <x, u_c>
                pass

            if t == 0:  # initial iteration
                SKPs = average_representations2(support_repres, support_kp_mask)  # C x N
            else:
                SKPs = avg_support_repres

            sum_support_mask = torch.sum(support_kp_mask, dim=0)  # N
            union_support_kp_mask = sum_support_mask > 0  # N
            if use_neg_kp:  # whether use negative keypoints
                sample_success_flag, neg_kps, neg_kp_mask, neg_kp_repres = nargs['sample_neg_kp_func'](neg_kp_num)
                if sample_success_flag:
                    SKPs = torch.cat([SKPs, neg_kp_repres], dim=1)  # C x (N + neg_num)
                    union_support_kp_mask = torch.cat([union_support_kp_mask, neg_kp_mask.bool()])  # (N + neg_num)
            softmax_mask = -(1 - union_support_kp_mask.float()) * 100  # N

            if (fused_attention is not None) and ('spatial-suc' in fused_attention):  # firstly normalize; secondly average
                candidate_repres = query_repres.permute(0, 2, 1).reshape(B2*k*N, -1)  # (B*k*N) x C
                SKPs_norm = SKPs  # C x N, already normalized
                candidate_repres_norm = candidate_repres  # (B*k*N) x C, already normalized
                cos_distance = ((candidate_repres_norm.unsqueeze(-1) - SKPs_norm.unsqueeze(0)) ** 2).sum(1)  # (B*k*N) x N
                soft_assign_dist = -cos_distance/(soft_assign_sigma**2)  # (B*k*N) x N
                soft_assign_prob = torch.softmax(soft_assign_dist+softmax_mask.unsqueeze(0), dim=1)  # masked softmax, (B*k*N) x N
            else:  # channel attention or pure spatial attention <x, u_c>
                candidate_repres = query_repres.permute(0, 2, 1).reshape(B2*k*N, -1)  # (B*k*N) x C
                SKPs_norm = torch.nn.functional.normalize(SKPs, p=2, dim=0, eps=1e-12)  # C x N
                candidate_repres_norm = torch.nn.functional.normalize(candidate_repres, p=2, dim=1, eps=1e-12)  # (B*k*N) x C
                cos_distance = 1 - candidate_repres_norm.matmul(SKPs_norm)  # (B*k*N) x N
                soft_assign_dist = -cos_distance/(soft_assign_sigma**2)  # (B*k*N) x N
                soft_assign_prob = torch.softmax(soft_assign_dist+softmax_mask.unsqueeze(0), dim=1)  # masked softmax, (B*k*N) x N

            if use_neg_kp:  # whether use negative keypoints
                if sample_success_flag:
                    SKPs = SKPs[:, :N]
                    union_support_kp_mask = union_support_kp_mask[:N]
                    softmax_mask = softmax_mask[:N]
                    cos_distance = cos_distance[:, :N]
                    soft_assign_prob = soft_assign_prob[:, :N]

            soft_assign_prob = soft_assign_prob.reshape(B2*k, N, N)  # (B*k) x N x N
            soft_assign_prob = torch.diagonal(soft_assign_prob, offset=0, dim1=1, dim2=2)  # (B*k) x N
            soft_assign_prob = soft_assign_prob * select_indicator  # (B*k) x N, filter not selected fibers


            candidate_repres = query_repres * select_indicator.unsqueeze(dim=1)  # (B*k) x C x N
            sum_SKPs = (support_repres * support_kp_mask.unsqueeze(dim=1)).sum(dim=0)  # C x N
            sum_SKPs = sum_SKPs.permute(1, 0)  # N x C
            sum_assigned_repres = (candidate_repres * soft_assign_prob.unsqueeze(dim=1)).sum(dim=0)  # C x N
            sum_assigned_repres = sum_assigned_repres.permute(1, 0)  # N x C
            sum_assigned_probs = soft_assign_prob.sum(dim=0)  # N
            # N x C, refine using support keypoint repres (SKRs) + assigned_repres
            avg_support_repres = (soft_assign_w * sum_SKPs + (1-soft_assign_w) * sum_assigned_repres) / \
                                 (soft_assign_w * sum_support_mask + (1-soft_assign_w) * sum_assigned_probs + 1e-12).unsqueeze(dim=1)
            # N x C, refine using current prototypes (SKPs) + prototype drift (candidate_repres - prototype)
            # avg_support_repres = (2*soft_assign_w-1) * SKPs.permute(1, 0) + (1-soft_assign_w) * sum_assigned_repres / \
            #                      (sum_assigned_probs + 1e-12).unsqueeze(dim=1)
            avg_support_repres = avg_support_repres.permute(1, 0)  # C x N

        else:
            raise NotImplementedError
        # ----------------------------

        # t4 = time.time()
        # print('3: ', t4 - t3)

        # 4) predict new kp labels (feature modulation + descriptors + kp localization)
        if (fused_attention is None):  # channel attention
            attn_feat, attn_map_l2 = model.cascade_salT.att_modulators[0](avg_support_repres, q_feats, None, model.opts['use_fused_attention'], False)
        elif ('channel' in fused_attention):  # don't use
            attn_feat, attn_map_l2 = model.cascade_salT.att_modulators[0](avg_support_repres, q_feats, support_kp_mask, model.opts['use_fused_attention'], s_repres=support_repres)
        elif ('spatial-suc' in fused_attention):
            is_normalized = True
            # trial = 2
            trial = model.opts['tfskd_config'][6]
            # use_pre_U = False
            use_pre_U = bool(model.opts['tfskd_config'][7])

            if trial == 0:
                attn_feat, attn_map_l2 = model.cascade_salT.att_modulators[0](avg_support_repres, q_feats, support_kp_mask, model.opts['use_fused_attention'], is_norm=is_normalized, s_repres=support_repres, use_pre_U=use_pre_U)
            elif trial == 1:
                combined_repres = torch.cat([support_repres, query_repres], dim=0)  # (B1 + n_kps) x C x N
                combined_kp_mask = torch.cat([support_kp_mask, select_indicator], dim=0)  # (B1 + n_kps) x N
                attn_feat, attn_map_l2 = model.cascade_salT.att_modulators[0](avg_support_repres, q_feats, combined_kp_mask, model.opts['use_fused_attention'], is_norm=is_normalized, s_repres=combined_repres, use_pre_U=use_pre_U)
            elif trial == 2:
                '''
                need to provide: 
                avg_support_repres, support_repres + support_kp_mask, query_repres + soft_assign_prob
                '''
                str_splits = fused_attention.split('-')
                scale_f = float(str_splits[2]) # 1.0
                alpha = float(str_splits[3]) # 0.5
                k_subspace = int(str_splits[4]) # 5
                Temporature = float(str_splits[5])  # 0.5
                sample_type = int(str_splits[6])  # 0, no sampling; 1, independent noise (1shot-->B1); 2, independent noise (<B1 --> B1);

                from network.feat_modulation_module import fibers_augmentation, fibers_centering, get_subspaces_by_svd, get_distance_by_subspaces, get_dot_product_similarity

                avg_repres = average_representations2(support_repres, support_kp_mask)  # C x N
                s_repres, support_kp_mask = fibers_augmentation(support_repres, avg_repres, support_kp_mask, sample_type)
                s_repres_dev = fibers_centering(s_repres, avg_repres)
                U1, S1 = get_subspaces_by_svd(s_repres_dev, support_kp_mask)
                U1 = U1.to(torch.float32)
                subspace1 = U1[:,:,:k_subspace]

                sum_assigned_probs = soft_assign_prob.sum(dim=0)  # N
                sum_assigned_repres = (query_repres * soft_assign_prob.unsqueeze(dim=1)).sum(dim=0)  # C x N
                avg_assigned_repres = sum_assigned_repres / (sum_assigned_probs + 1e-12).unsqueeze(0)  # C x N
                # n_kps x C x N, n_kps x N
                pseudo_repres, pseudo_kp_mask = fibers_augmentation(query_repres, avg_assigned_repres, select_indicator, sample_type)
                pseudo_repres_dev = fibers_centering(pseudo_repres, avg_assigned_repres)
                pseudo_kp_prob_normalized = soft_assign_prob / (sum_assigned_probs + 1e-12).unsqueeze(0)  # n_kps x N
                U2, S2 = get_subspaces_by_svd(pseudo_repres_dev, pseudo_kp_prob_normalized)
                U2 = U2.to(torch.float32)
                subspace2 = U2[:, :, :k_subspace]

                # normalize
                q_feats_norm = torch.nn.functional.normalize(q_feats, p=2, dim=1, eps=1e-12)  # B2 x C x H x W
                if is_normalized:  # already normalized support_kp_repres outside this function
                    q_feats_tmp = q_feats_norm  # used to compute distance by subspaces
                    avg_support_repres_norm = avg_support_repres  # C x N, used to compute dot-product similarity
                else:
                    q_feats_tmp = q_feats  # used to compute distance by subspaces
                    avg_support_repres_norm = torch.nn.functional.normalize(avg_support_repres, p=2, dim=0, eps=1e-12)  # C x N, used to compute dot-product similarity

                distance1 = get_distance_by_subspaces(avg_repres, q_feats_tmp, subspace1)
                attention_map1 = torch.exp(-distance1/Temporature)  # B2 x N x H x W
                # attention_map1np = attention_map1.detach().cpu().numpy()
                distance2 = get_distance_by_subspaces(avg_assigned_repres, q_feats_tmp, subspace2)
                attention_map2 = torch.exp(-distance2 / Temporature)  # B2 x N x H x W
                # attention_map2np = attention_map2.detach().cpu().numpy()

                # dot-product based similarity
                attention_map_l2 = get_dot_product_similarity(avg_support_repres_norm, q_feats_norm)  # B2 x N x H x W
                # attention_map_l2np = attention_map_l2.detach().cpu().numpy()

                gamma = (soft_assign_w * sum_support_mask + (1-soft_assign_w) * sum_assigned_probs + 1e-12)  # N
                att_w1 = soft_assign_w * sum_support_mask / gamma  # N
                att_w2 = (1-soft_assign_w) * sum_assigned_probs / gamma  # N
                att_w1 = att_w1.view(1, -1, 1, 1)  # 1 x N x 1 x 1
                att_w2 = att_w2.view(1, -1, 1, 1)  # 1 x N x 1 x 1
                attention_map_final = alpha * attention_map_l2 + (1-alpha) * (att_w1 * attention_map1 + att_w2 * attention_map2)  # B2 x N x H x W
                # attention_map_finalnp = attention_map_final.detach().cpu().numpy()

                attn_feat = q_feats.unsqueeze(1)*attention_map_final.unsqueeze(2) * scale_f  # B2 x N x C x H x W
            else:
                raise NotImplementedError
        elif ('dual' in fused_attention):
            # trial = 0
            trial = model.opts['tfskd_config'][6]

            if trial == 0:
                attn_feat, attn_map_l2 = model.cascade_salT.att_modulators[0](avg_support_repres, q_feats, support_kp_mask, model.opts['use_fused_attention'], s_repres=support_repres)
            elif trial == 1:
                combined_repres = torch.cat([support_repres, query_repres], dim=0)  # (B1 + n_kps) x C x N
                combined_kp_mask = torch.cat([support_kp_mask, select_indicator], dim=0)  # (B1 + n_kps) x N
                attn_feat, attn_map_l2 = model.cascade_salT.att_modulators[0](avg_support_repres, q_feats, combined_kp_mask, model.opts['use_fused_attention'], s_repres=combined_repres)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        keypoint_descriptors = model.descriptor_net(attn_feat)

        mean_predictions = 0
        mean_grid_var = 0
        multi_scale_covar = []
        multi_scale_predictions = []
        multi_scale_grid_scores = []
        ms_o_grids = []
        ms_o_offsets = []
        ms_o_rhos = []
        for scale_i, grid_length in enumerate(model.grid_length):
            if model.opts['eval_compute_var'] == 0:  # don't compute var or covar
                predict_gridsO, predict_deviations, _, _= model.regressor[scale_i](keypoint_descriptors, training_phase=False)  # B2 x N x (grid_length ** 2), B2 x N x 2
            else:  # True
                predict_gridsO, predict_deviations, rho, grid_var, offsetsO, rhoO= model.regressor[scale_i].test(keypoint_descriptors, training_phase=False, compute_grid_var=True)  # B2 x N x (grid_length ** 2), B2 x N x 2

            # compute predicted keypoint locations using grids and deviations
            grid_scores, predict_grids = torch.max(predict_gridsO, dim=2)  # B2 x N, 0 ~ grid_length * grid_length - 1
            predict_gridxy = torch.FloatTensor(B2, N, 2).cuda()
            predict_gridxy[:, :, 0] = predict_grids % grid_length  # grid x
            predict_gridxy[:, :, 1] = predict_grids // grid_length # grid y

            # Method 1, deviation range: -1~1
            predictions = (((predict_gridxy + 0.5) + predict_deviations / 2.0) / grid_length - 0.5) * 2  # deviation -1~1
            # Method 2, deviation range: 0~1
            # predictions = ((predict_gridxy + predict_deviations) / grid_length - 0.5) * 2
            mean_predictions += predictions
            # mean_predictions_grid += ((predict_gridxy + 0.5) / grid_length - 0.5) * 2  # only used for testing grid classification
            multi_scale_predictions.append(predictions)
            multi_scale_grid_scores.append(grid_scores)
            ms_o_grids.append(predict_gridsO)
            ms_o_offsets.append(offsetsO)
            ms_o_rhos.append(rhoO)

        # t0 = time.time()
        # print('4: ', t0 - t4)

        # 0) expand pseudo label set (select top 2 or 3 grids per keypoint + sample z kps around top-k kp)
        # B2 x N x k x 2; B2 x N x k; B2 x N x k
        curr_preds, scores, curr_ucs, curr_covar = process_preds(ms_o_grids, ms_o_offsets, ms_o_rhos, topk=top_k_grids, samplez=sample_z_kps)
        k = scores.shape[-1]  # each kp expanded k kps (note k may not be same to top_k_grids), which depends sampling/selecting
        candidate_kp_mask = valid_kp_mask.unsqueeze(-1).repeat(1, 1, k)  # B2 x N x k
        scores = scores * candidate_kp_mask  # masking before sorting candidate kps
        curr_ucs = curr_ucs * candidate_kp_mask  # masking before sorting candidate kps
        curr_preds = curr_preds * candidate_kp_mask.unsqueeze(-1)

        # predictions = mean_predictions * (valid_kp_mask.unsqueeze(dim=2))
        # query_labels = query_labels * (valid_kp_mask.unsqueeze(dim=2))

        # curr_preds = mean_predictions * (valid_kp_mask.unsqueeze(dim=2))  # B2 x N x 2, our kp's range is -1~1
        # scores = 0
        # for i in range(scale_num):
        #     scores += multi_scale_grid_scores[i].exp()  # use exp(.) because original value is from logsoftmax
        # scores = (scores / scale_num)  * valid_kp_mask  # B2 x N

    return curr_preds[:, :, 0, :]  # top-1 kp predictions


def copy_original_params(model):
    # save a copy of the original model parameters; it should note that default keep_vars is false, which will cause require_grads=False.
    original_params_dict = {'E': copy.deepcopy(model.encoder.state_dict(keep_vars=True)), 'D': copy.deepcopy(model.descriptor_net.state_dict(keep_vars=True)),
                            'R': copy.deepcopy(model.regressor.state_dict(keep_vars=True)), 'MCovar': copy.deepcopy(model.covar_branch.state_dict(keep_vars=True)),
                            'SalT': copy.deepcopy(model.cascade_salT.state_dict(keep_vars=True))}
    return original_params_dict

def finetuning_via_gradient_steps(model, finetuning_steps, original_params_dict, supports, support_labels, support_kp_mask, support_saliency_t):
    if model.opts['set_eval']:
        model.set_train_status()
    torch.set_grad_enabled(True)  # enable grad computation
    current_params_dict = copy.deepcopy(original_params_dict)
    # print(current_params_dict)
    # assign the current_params_dict to model
    model.encoder.load_state_dict(current_params_dict['E'])
    model.descriptor_net.load_state_dict(current_params_dict['D'])
    model.regressor.load_state_dict(current_params_dict['R'])
    model.covar_branch.load_state_dict(current_params_dict['MCovar'])
    model.cascade_salT.load_state_dict(current_params_dict['SalT'])
    # model.encoder, model.descriptor_net, model.regressor, model.covar_branch = model.encoder.cuda(), model.descriptor_net.cuda(), model.regressor.cuda(), model.covar_branch.cuda()
    # e = nn.ParameterList(model.encoder.parameters())
    model.optimizer_init(lr=0.0001, lr_auxiliary = 0.0001, weight_decay=0, optimization_algorithm='Adam')
    # update_lr = 0.0001
    union_support_kp_mask = torch.sum(support_kp_mask, dim=0) > 0  # N
    for k in range(finetuning_steps):
        # print_weights(model.regressor[0].linear_grid_class[0].weight.data)
        support_features, support_lateral_out = model.encoder(x=supports, x2=None, enable_lateral_output=True)   # B1 x C x H x W, B1 x 1 x H x W
        if model.opts['use_pum']:
            # computing_inv_w = True
            w_computing_mode = 2
            p_support_lateral_out = model.numerical_transformation(support_lateral_out, w_computing_mode=w_computing_mode)  # B1 x 1 x H' x W', transform into positive number
            # regarding main training kps
            inv_w_patches, w_patches = model.get_distinctiveness_for_parts(p_support_lateral_out, p_support_lateral_out, support_labels, support_kp_mask, support_labels)
        else:
            inv_w_patches = None

        # finetuning phase
        salT_att_param_config = {
            'mask_mode':  model.opts['salT_cfg']['mask_mode'],
            'func_mode':  model.opts['salT_cfg']['func_mode'],
            'sal_beta' :  model.opts['salT_cfg']['sal_beta'],
            'sal_eps'  :  model.opts['salT_cfg']['sal_eps'],
            'thresh'   :  model.opts['salT_cfg']['thresh'],
            'use_max'  :  model.opts['salT_cfg']['use_max'],
            'rbf_l2norm': model.opts['salT_cfg']['rbf_l2norm'],
            'rownorm'   : model.opts['salT_cfg']['rownorm'],
        }

        # patched_sal_s = get_patches(support_saliency_t.cpu(), patch_size=(32, 32), save=True, prefix='s')
        # patched_sal_q = get_patches(query_saliency_t.cpu(),   patch_size=(32, 32), save=True, prefix='q')


        extract_kp_repres_func = partial(extract_representations, context_mode=model.context_mode, sigma=model.opts['sigma'], \
                                         downsize_factor=model.opts['downsize_factor'], image_length=model.opts['square_image_length'], together_trans_features=None)
        s_feats_select, vit_s_attn_probs, s_vis_pow_masks, s_morph_reg_loss, s_morph_reg_loss_o, _ = \
        model.cascade_salT(
            support_features,  # B1 x C x h x w (feature space)
            support_saliency_t,# B1 x 1 x H x W (image space)
            supports,          # B1 x 3 x H x W (image space)
            salT_att_param_config
            )
        q_feats_select = s_feats_select
        q_morph_reg_loss = s_morph_reg_loss
        q_morph_reg_loss_o = s_morph_reg_loss_o
        if (model.opts['sal_mask_cfg']['sal_mask_type'] in ['S', 'IS', 'II']) and model.opts['sal_mask_cfg']['impose_reg'] > 0:
            morph_reg_N = s_morph_reg_loss.shape[0] + q_morph_reg_loss.shape[0]  # B1 + B2
            morph_reg_loss = (s_morph_reg_loss.sum() + q_morph_reg_loss.sum()) / morph_reg_N
            morph_reg_loss_o = (s_morph_reg_loss_o.sum() + q_morph_reg_loss_o.sum()) / morph_reg_N
        else:
            morph_reg_loss = 0
            morph_reg_loss_o = 0
        attentive_feats, attention_maps_l2, attentive_feats_aux, attention_maps_l2_aux = \
        model.cascade_salT.support_query_modulation(
            s_feats_select,
            q_feats_select,
            support_labels,
            support_kp_mask,
            None,
            None,
            extract_kp_repres_func,
            fused_attention= model.opts['use_fused_attention'],
            output_attention_maps=False,
            )

        # process attentive_feats
        if model.cascade_map_process == 'catch':
            attentive_feats = torch.cat(attentive_feats, dim=2)  # B2 x N x C' x H x W (C': catched feature channels)
        elif model.cascade_map_process == 'mean':
            attentive_feats = (sum(attentive_feats) / len(attentive_feats))
        else:
            print('Error for model.cascade_map_process!')
            exit(0)

        keypoint_descriptors = model.descriptor_net(attentive_feats)  # B2 x N x D  or B2 x N x c x h x w
        valid_kp_mask_for_support = (support_kp_mask * union_support_kp_mask.reshape(1, -1))  # B1 x N
        loss_grid_class, loss_deviation, _ = model.multiscale_regression(keypoint_descriptors, support_labels, valid_kp_mask_for_support, model.grid_length, weight=None, inv_w_patches=inv_w_patches, w_patches=None)
        loss = loss_grid_class + loss_deviation
        loss = model.opts['loss_weight'][0] * loss
        if (model.opts['sal_mask_cfg']['sal_mask_type'] in ['S', 'IS', 'II']) and model.opts['sal_mask_cfg']['impose_reg'] > 0:
                reg_loss_weight = model.opts['sal_mask_cfg']['reg_loss_weight']
                reg_alpha = reg_loss_weight
                # reg_x = float(episode_i) / model.opts['num_episodes']
                # reg_p = 0.01
                # reg_a = np.log(2 * reg_loss_weight / reg_p -1)
                # reg_alpha = 2 * reg_loss_weight / (np.exp(reg_a * reg_x) + 1)  # ranges from 1 to reg_p (when 0<=x<=1) for annealing
                morph_reg_loss_t = reg_alpha * morph_reg_loss
                loss = loss + morph_reg_loss_t

        # params_list = nn.ParameterList()
        # model_list = [model.descriptor_net, model.regressor]
        # len_temp = 0
        # # combine the Parameters
        # for each_model in model_list:
        #     each_param_list = nn.ParameterList(each_model.parameters())
        #     params_list.extend(each_param_list)
        # # compute gradients; note that the Parameter in params_list should all be differentiable (namely requires_grad==True)
        # grad = torch.autograd.grad(loss, params_list)
        # # update weights
        # fast_weights = list(map(lambda p: p[1] - update_lr * p[0], zip(grad, params_list)))
        # # assign the adapted weights to the model
        # for i, each_model in enumerate(model_list):
        #     each_param_list = nn.ParameterList(each_model.parameters())
        #     for j in range(len(each_param_list)):  # Parameter has two
        #         each_param_list[j].data = fast_weights[len_temp + j]
        #     len_temp += len(each_param_list)

        model.optimizer_step(loss)
        # print(loss.item())
    torch.set_grad_enabled(False)  # disable grad computation
    if model.opts['set_eval']:
        model.set_eval_status()

