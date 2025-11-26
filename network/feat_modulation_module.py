# -*- coding: utf-8 -*-
import time

import torch
from torch import nn

from network.models_comparisons import LinearDiag, LinearMat
import copy
import numpy as np
from network.models_gridms2 import average_representations2

class AttentionBasedModulator(nn.Module):
    def __init__(self, dim, support_fc_type='diag', init_value=1.0, num_layers=1):
        '''
        dim: feature dim
        support_fc_type: 'diag', 'mat', None
        '''

        super(AttentionBasedModulator, self).__init__()

        self.support_fc_type = support_fc_type
        if support_fc_type == 'diag':
            self.support_fc = LinearDiag(dim, init_value=init_value, bias=False)
        elif support_fc_type == 'mat':
            self.support_fc = LinearMat(dim, init_value=init_value, bias=False, num_layers=num_layers)
        else:
            pass

    def forward(
        self,
        support_fibers,
        query_features,
        support_kp_mask=None,
        fused_attention = None,
        output_attention_maps = False,
        **nargs,
        # compute_similarity=False
    ):
        '''
        support_fibers: C x N (keypoint protos) or B x C x N (each kp repres)
        query_features: B x C x H x W
        fused_attention: None, 'cosine', 'softmax'
        '''
        is_kp_protos = True
        if len(support_fibers.size()) == 2:  # C x N
            N_categories = support_fibers.shape[1]
            support_kp_repres = support_fibers.transpose(1, 0)  # N x C
            N = N_categories
        elif len(support_fibers.size()) == 3:  # B x C x N
            assert support_kp_mask is not None, "Error for support_kp_mask, AttentionBasedModulator."
            is_kp_protos = False
            B1, C, N_categories = support_fibers.shape
            support_kp_repres = support_fibers.transpose(2, 1).reshape(B1 * N_categories, C)  # (B1 * N) x C
            N = B1 * N_categories

        if self.support_fc_type != None:
            support_kp_repres = self.support_fc(support_kp_repres)  # support keypoint representations

        B2, C, H, W = query_features.shape

        if fused_attention == None:
            attentive_features = query_features.reshape(B2, C, H*W).unsqueeze(dim=1) * support_kp_repres.reshape(1, N, C, 1)  # B2 x N x C x (H*W)
            attentive_features = attentive_features.reshape(B2, N, C, H, W)  # modulated attentive features, B2 x N x C x H x W
            if output_attention_maps == True:
                q_norm = torch.norm(query_features, p=2, dim=1, keepdim=True)
                s_norm = torch.norm(support_kp_repres, p=2, dim=1, keepdim=True)
                attention_map_l2 = attentive_features.sum(dim=2) / (q_norm * s_norm.view(1, N, 1, 1) + 1e-6)
                attention_map_l2 = torch.clamp(attention_map_l2, min=0, max=1.0)  # B2 x N x H x W
        elif ('channel' in fused_attention):
            str_splits = fused_attention.split('-')
            sample_type = int(str_splits[1])  # 0, no sampling; 1, independent noise (1shot-->B1); 2, independent noise (<B1 --> B1); 3, covariant noise (<B1 --> B1)

            s_repres = nargs['s_repres']  # B1 x C x N, only used here
            union_s_kp_mask = support_kp_mask.sum(0)  # N
            B1 = s_repres.shape[0]

            if sample_type == 0:  # no sampling noise
                pass
            else:
                if B1 == 1:  # 1shot---> augemnted to 5 shots
                    # s_repres = s_repres.repeat(5, 1, 1)  # 5 x C x N
                    # s_repres[1:, :, :] = s_repres[1:, :, :] + torch.randn(4, C, N).cuda() * 0.001
                    s_repres = support_fibers.unsqueeze(0).repeat(5, 1, 1)  # 5 x C x N (use previously computed avg proto for augmentation)
                    s_repres[1:, :, :] = s_repres[1:, :, :] + torch.randn(4, C, N).cuda() * 0.001

                    B1 = 5  # update num supports
                    support_kp_mask = support_kp_mask.repeat(5, 1)  # 5 x N
                    support_fibers = s_repres.mean(0)  # C x N, update kp protos
                else:
                    if sample_type == 1:  # feature aug., independent noise (only 1shot-->B1)
                        for i in range(N):
                            if union_s_kp_mask[i] == 1:
                                # ind = (support_kp_mask[:, i]).tolist().index(1)
                                # repres_tmp = (s_repres[ind, :, i]).unsqueeze(0).repeat(B1, 1)
                                # s_repres[:, :, i] = repres_tmp
                                # for j in range(B1):
                                #     if j != ind:
                                #         s_repres[j, :, i] = s_repres[j, :, i] + torch.randn(C).cuda() * 0.001
                                repres_tmp = support_fibers[:, i].unsqueeze(0).repeat(B1, 1)  # B1 x C (use previously computed avg proto for augmentation)
                                s_repres[:, :, i] = repres_tmp
                                s_repres[1:, :, i] = s_repres[1:, :, i] + torch.randn(B1-1, C).cuda() * 0.001
                                support_kp_mask[:, i] = 1
                        support_fibers = average_representations2(s_repres, support_kp_mask)  # C x N, update kp protos
                    elif (sample_type) == 2:  # feature aug., independent noise (<B1 --> B1)
                        for i in range(N):
                            if (union_s_kp_mask[i] < B1) and (union_s_kp_mask[i] > 0):
                                for j in range(B1):
                                    if support_kp_mask[j, i] == 0:
                                        s_repres[j, :, i] = support_fibers[:, i] + torch.randn(C).cuda() * 0.001
                                support_kp_mask[:, i] = 1
                        support_fibers = s_repres.mean(0)  # C x N, update kp protos
                    elif (sample_type) == 3:  # feature aug., covariant noise (<B1 --> B1)
                        s_repres_dev = s_repres - support_fibers.unsqueeze(0)  # B1 x C x N, (phi - uc)
                        s_repres_dev = s_repres_dev * support_kp_mask.unsqueeze(1)  # B1 x C x N
                        s_repres_dev = s_repres_dev.permute(2, 1, 0)  # N x C x B1, (phi - uc)
                        s_repres_dev = s_repres_dev.detach().to(torch.float64)
                        U, S, V = torch.svd(s_repres_dev, some=True)  # batch SVD decomposition
                        union_s_kp_mask_tmp = union_s_kp_mask + (union_s_kp_mask == 0).long()  # N, avoid 0 valid samples
                        S_tmp = (S) / union_s_kp_mask_tmp.sqrt().unsqueeze(1)  # N x k, divide num_valid support samples per class
                        # sample noise
                        noise = torch.randn(N, S.shape[1], B1).cuda() * S_tmp.unsqueeze(-1)  # N(0, S^2 / num_valid), N x k x B1
                        # project the noise to high dimension
                        noise_project_back = torch.bmm(U, noise).to(torch.float32)  # N x C x B1
                        noise_project_back = noise_project_back.permute(2, 1, 0)    # B1 x C x N
                        aug_fibers = noise_project_back + support_fibers.unsqueeze(0)  # B1 x C x N
                        mask_tmp = support_kp_mask.unsqueeze(1)
                        s_repres = mask_tmp * s_repres + (1-mask_tmp) * aug_fibers  # B1 x C x N
                        support_fibers = s_repres.mean(0)  # C x N, update kp protos
                        support_kp_mask = ((union_s_kp_mask > 0).long()).unsqueeze(0).repeat(B1, 1)  # B1 x N
                    else:
                        raise NotImplementedError

            support_kp_repres = support_fibers.transpose(1, 0)  # N x C
            attentive_features = query_features.reshape(B2, C, H*W).unsqueeze(dim=1) * support_kp_repres.reshape(1, N, C, 1)  # B2 x N x C x (H*W)
            attentive_features = attentive_features.reshape(B2, N, C, H, W)  # modulated attentive features, B2 x N x C x H x W
            if output_attention_maps == True:
                q_norm = torch.norm(query_features, p=2, dim=1, keepdim=True)
                s_norm = torch.norm(support_kp_repres, p=2, dim=1, keepdim=True)
                attention_map_l2 = attentive_features.sum(dim=2) / (q_norm * s_norm.view(1, N, 1, 1) + 1e-6)
                attention_map_l2 = torch.clamp(attention_map_l2, min=0, max=1.0)  # B2 x N x H x W

        elif fused_attention in ['spatial-att', 'spatial-att1.5', 'spatial-att2.0', 'spatial-att2.5', 'spatial-att3.0', 'spatial-att0.5', 'spatial-att0.8']:
            if fused_attention == 'spatial-att':
                scale_f = 1.0
            elif fused_attention == 'spatial-att1.5':
                scale_f = 1.5
            elif fused_attention == 'spatial-att2.0':
                scale_f = 2.0
            elif fused_attention == 'spatial-att2.5':
                scale_f = 2.5
            elif fused_attention == 'spatial-att3.0':
                scale_f = 3.0
            elif fused_attention == 'spatial-att0.5':
                scale_f = 0.5
            elif fused_attention == 'spatial-att0.8':
                scale_f = 0.8

            # normalize
            support_kp_repres2 = torch.nn.functional.normalize(support_kp_repres, p=2, dim=1, eps=1e-12)  # N x C
            query_features2 = torch.nn.functional.normalize(query_features, p=2, dim=1, eps=1e-12)  # B2 x C x H x W
            # dot-product
            attention_map_l2 = query_features2.reshape(B2, C, H*W).unsqueeze(dim=1) * support_kp_repres2.reshape(1, N, C, 1)  # B2 x N x C x (H*W)
            attention_map_l2 = attention_map_l2.sum(dim=2)  # B2 x N x (H*W)
            attentive_features = query_features.reshape(B2, 1, C, H, W)*attention_map_l2.reshape(B2, N, 1, H, W) * scale_f  # B2 x N x C x H x W
            if output_attention_maps == True:
                # B2 x N x H x W
                attention_map_l2 = torch.clamp(attention_map_l2.reshape(B2, N, H, W), min=0, max=1.0)
        elif ('spatial-suc' in fused_attention):
            str_splits = fused_attention.split('-')
            scale_f = float(str_splits[2]) # 1.0
            alpha = float(str_splits[3]) # 0.5
            k_subspace = int(str_splits[4]) # 5
            Temporature = float(str_splits[5])  # 0.5
            sample_type = int(str_splits[6])  # 0, no sampling; 1, independent noise (1shot-->B1); 2, independent noise (<B1 --> B1);

            is_normalized = nargs['is_norm']
            s_repres = nargs['s_repres']  # B1 x C x N, only used here
            B1 = s_repres.shape[0]
            # print(support_kp_mask)
            back_propagate_svd = False

            s_repres, support_kp_mask = fibers_augmentation(s_repres, support_fibers, support_kp_mask, sample_type)
            s_repres_dev = fibers_centering(s_repres, support_fibers)
            # -----------------------------------------------
            # # compute covariance matrix
            # covar = torch.zeros(N, C, C).cuda()  # N x C x C
            # for i in range(N):
            #     if union_s_kp_mask[i] > 0:
            #         for j in range(B1):
            #             if support_kp_mask[j, i] == 1:
            #                 repres_tmp = (s_repres_dev[j, :, i]).reshape(-1, 1)
            #                 covar[i] += torch.matmul(repres_tmp, repres_tmp.transpose(1, 0))
            #         covar[i] /= union_s_kp_mask[i]
            #     else:
            #         covar[i] = covar[i].fill_diagonal_(1.0)
            #
            # # SVD decomposition and grab dominant subspaces
            # # U, S, Vh = torch.linalg.svd(covar)  # batch SVD decomposition
            # covar = covar.detach().to(torch.float64)
            # U, S, V = torch.svd(covar)  # batch SVD decomposition
            # U2 = U[:, :, :k_subspace]  # N x C x k_subspace
            # U2h = U2.transpose(2, 1)   # N x k_subspace x C
            # U2h = U2h.to(torch.float32)
            # -----------------------------------------------
            # U0, S0 = get_subspaces_by_svd(s_repres_dev, support_kp_mask)

            s_repres_dev = s_repres_dev * support_kp_mask.unsqueeze(1)  # B1 x C x N
            s_repres_dev = s_repres_dev.permute(2, 1, 0)  # N x C x B1, (phi - uc)
            union_s_kp_mask = support_kp_mask.sum(0)  # N
            union_s_kp_mask_tmp = union_s_kp_mask + (union_s_kp_mask == 0).long()  # N, avoid 0 valid samples
            s_repres_dev = s_repres_dev / union_s_kp_mask_tmp.float().sqrt().reshape(N, 1, 1)  # divide num of valid samples
            if back_propagate_svd == False:
                s_repres_dev = s_repres_dev.detach().to(torch.float64)
            else:
                s_repres_dev = s_repres_dev + 1e-12
                # s_repres_dev = s_repres_dev + (torch.eye(C, B1).cuda() * 1e-12).unsqueeze(0)
                s_repres_dev = s_repres_dev.to(torch.float64)
            if (nargs.get('use_pre_U') is not None) and (nargs.get('use_pre_U') == True):
                U = self.previous_U  # use previous U
            else:  # nargs.get('use_pre_U') is None or == False
                # SVD decomposition and grab dominant subspaces
                U, S, V = torch.svd(s_repres_dev, some=True)  # batch SVD decomposition
                self.previous_U = U  # record U

            if (str_splits[0]+'-'+str_splits[1]) == 'spatial-suc':  # exp( -||(I-UU^T)(x-uc)||^2 / T ), using subspace
                U2 = U[:, :, :k_subspace]  # N x C x k_subspace
                # U2h = U2.transpose(2, 1)   # N x k_subspace x C
                # U2U2h = torch.bmm(U2, U2h)  # N x C x C
                # identity_matrix = torch.eye(C).unsqueeze(0).repeat(N, 1, 1).cuda()
                # proj_matrix = identity_matrix - U2U2h  # N x C x C, (I - U2U2h)
                # proj_matrix = proj_matrix.to(torch.float32)

                # U2 = U2.to(torch.float32)
                COMPUTE_MODE = 1
            elif (str_splits[0]+'-'+str_splits[1]) == 'spatial-suc2':  # exp( -(x-uc)^T * Sigma^{-1} * (x-uc) / T ) = ||inv_S*U^T*(x-uc)||^2
                # aa = S.detach().cpu().numpy()
                Uh = U.transpose(2, 1)  # N x k_subspace x C
                S = S + 1
                inv_S = 1 / S
                proj_matrix = inv_S.unsqueeze(-1) * Uh  # N x k_subspace x C, faster to compute than above bmm

                # proj_matrix = proj_matrix.to(torch.float32)
                COMPUTE_MODE = 2
            elif (str_splits[0]+'-'+str_splits[1]) == 'spatial-suc3':  # exp( -(x-uc)^T * Sigma^{-1} * (x-uc) / T ) = ||inv_S*U^T*(x-uc)||^2
                # aa = S.detach().cpu().numpy()
                Uh = U.transpose(2, 1)  # N x k_subspace x C
                S = S + 0.1
                inv_S = 1 / S
                proj_matrix = inv_S.unsqueeze(-1) * Uh  # N x k_subspace x C, faster to compute than above bmm

                # proj_matrix = proj_matrix.to(torch.float32)
                COMPUTE_MODE = 2
            elif (str_splits[0]+'-'+str_splits[1]) == 'spatial-suc4':  # exp( -(x-uc)^T * Sigma^{-1} * (x-uc) / T ) = ||inv_S*U^T*(x-uc)||^2
                # use pseudo inverse here
                # aa = S.detach().cpu().numpy()
                Uh = U.transpose(2, 1)  # N x k_subspace x C
                S_mask = (S >= 1e-4).long()  # we use pseudo inverse, thus for 0 entries we ignore them
                S = S_mask * S + (1-S_mask) * 0.1  # add 0.1 to avoid nan
                inv_S = S_mask * (1 / S)  # N x k_subspace
                proj_matrix = inv_S.unsqueeze(-1) * Uh  # N x k_subspace x C, faster to compute than above bmm

                # proj_matrix = proj_matrix.to(torch.float32)
                COMPUTE_MODE = 2
            else:
                raise NotImplementedError

            # compute similarity using dominant subspaces
            attention_map = torch.zeros(B2, N, H, W).cuda()
            if COMPUTE_MODE == 1:
                # query_features2 = query_features2.to(torch.float64)
                U2 = U2.to(torch.float32)
            elif COMPUTE_MODE == 2:
                # query_features2 = query_features2.to(torch.float64)
                proj_matrix = proj_matrix.to(torch.float32)

            # normalize
            support_kp_repres = support_fibers.transpose(1, 0)  # N x C
            query_features_norm = torch.nn.functional.normalize(query_features, p=2, dim=1, eps=1e-12)  # B2 x C x H x W
            if is_normalized == True:  # already normalized support_kp_repres outside this function
                query_features_tmp = query_features_norm  # used to compute distance by subspaces
                support_kp_repres_norm = support_kp_repres  # N x C, used to compute dot-product similarity
            else:
                query_features_tmp = query_features  # used to compute distance by subspaces
                support_kp_repres_norm = torch.nn.functional.normalize(support_kp_repres, p=2, dim=1, eps=1e-12)  # N x C, used to compute dot-product similarity

            for i in range(B2):
                each_query_feat = query_features_tmp[i]  # C x H x W
                each_query_feat_dev = each_query_feat.view(1, C, H*W) - support_kp_repres.view(N, C, 1)  # N x C x (H*W)
                if COMPUTE_MODE == 1:  # exp( -||(I-UU^T)(x-uc)||^2 / T ), using subspace
                    vecs_tmp = torch.bmm(U2.transpose(2, 1), each_query_feat_dev)
                    vecs_tmp = torch.bmm(U2, vecs_tmp)
                    projected_vecs = each_query_feat_dev - vecs_tmp  # N x C x (H*W)
                    projected_vecs = (projected_vecs ** 2).sum(1)  # N x (H*W)
                    projected_vecs = projected_vecs.reshape(N, H, W)
                elif COMPUTE_MODE == 2:  # exp( -(x-uc)^T * Sigma^{-1} * (x-uc) / T ) = ||inv_S*U^T*(x-uc)||^2
                    projected_vecs = torch.bmm(proj_matrix, each_query_feat_dev)  # N x k_subspace x (H*W)
                    projected_vecs = (projected_vecs ** 2).sum(1)  # N x (H*W)
                    projected_vecs = projected_vecs.reshape(N, H, W)
                    # projected_vecsnpp = projected_vecs.detach().cpu().numpy()
                else:
                    raise NotImplementedError
                attention_map[i, :, :, :] = projected_vecs[:, :, :]

            # attention_mapnp = attention_map.detach().cpu().numpy()
            attention_map = torch.exp(-attention_map/Temporature)  # exp(-dist/T)
            # attention_mapnp2 = attention_map.detach().cpu().numpy()
            # attention_map = attention_map.detach()


            # dot-product based similarity
            query_features_norm = query_features_norm.reshape(B2, C, H * W)
            attention_map_l2 = query_features_norm.unsqueeze(dim=1) * support_kp_repres_norm.reshape(1, N, C, 1)  # B2 x N x C x (H*W)
            attention_map_l2 = attention_map_l2.sum(dim=2)  # B2 x N x (H*W)
            attention_map_l2 = attention_map_l2.reshape(B2, N, H, W)
            # attention_map_l2np = attention_map_l2.detach().cpu().numpy()


            attention_map_final = alpha * attention_map_l2 + (1-alpha) * attention_map

            attentive_features = query_features.reshape(B2, 1, C, H, W)*attention_map_final.reshape(B2, N, 1, H, W) * scale_f  # B2 x N x C x H x W
            if output_attention_maps == True:
                # B2 x N x H x W
                attention_map_l2 = torch.clamp(attention_map_final, min=0, max=1.0)

        elif ('dual' in fused_attention):  # both channel att. + subspace based spatial att. (exp( -||(I-UU^T)x-uc)||^2 / T ))
            str_splits = fused_attention.split('-')
            scale_f = float(str_splits[1]) # 1.0
            alpha = float(str_splits[2]) # 0.5
            k_subspace = int(str_splits[3]) # 5
            Temporature = float(str_splits[4])  # 0.5
            sample_type = int(str_splits[5])  # 0, no sampling; 1, independent noise (1shot-->B1); 2, independent noise (<B1 --> B1);

            s_repres = nargs['s_repres']  # B1 x C x N, only used here
            B1 = s_repres.shape[0]
            # print(support_kp_mask)
            back_propagate_svd = False

            # B1 x C x N, B1 x N
            s_repres, support_kp_mask = fibers_augmentation(s_repres, support_fibers, support_kp_mask, sample_type)
            s_repres_tmp = torch.nn.functional.normalize(s_repres, p=2, dim=1, eps=1e-12)

            if (nargs.get('use_pre_U') is not None) and (nargs.get('use_pre_U') == True):
                U = self.previous_U  # use previous U
            else:  # nargs.get('use_pre_U') is None or == False
                # SVD decomposition and grab dominant subspaces
                s_repres_tmp = s_repres_tmp.to(torch.float64)
                U, S = get_subspaces_by_svd(s_repres_tmp, support_kp_mask, back_propagate_svd=back_propagate_svd)
                U = U.to(torch.float32)
                self.previous_U = U  # record U

            U2 = U[:, :, :k_subspace]  # N x C x k_subspace
            # compute similarity using dominant subspaces
            query_features_tmp = torch.nn.functional.normalize(query_features, p=2, dim=1, eps=1e-12)
            distance_map = get_distance_by_subspaces_wo_centering(query_features_tmp, U2)  # B2 x N x H x W
            # distance_map = (distance_map + 1e-12).sqrt()
            attention_map = torch.exp(-distance_map / Temporature)  # exp(-dist/T)
            # attention_mapnp = attention_map.cpu().detach().numpy()

            # modulated attentive features, B2 x N x C x H x W
            query_features = query_features.reshape(B2, C, H, W).unsqueeze(dim=1)  # B2 x 1 x C x H x W
            # attentive_features = alpha * query_features * support_kp_repres.reshape(1, N, C, 1, 1) + \
            #                      (1-alpha) * scale_f * query_features * attention_map.unsqueeze(dim=2)
            # below is faster than above
            attentive_features = (alpha * support_kp_repres.reshape(1, N, C, 1, 1) + \
                                 (1-alpha) * scale_f * attention_map.unsqueeze(dim=2)) * query_features
            if output_attention_maps == True:
                q_norm = torch.norm(query_features.reshape(B2, C, H, W), p=2, dim=1, keepdim=True)  # B2 x 1 x H x W
                s_norm = torch.norm(support_fibers.transpose(1, 0), p=2, dim=1, keepdim=True)  # N x 1
                # B2 x N x H x W
                attention_map_l2 = (query_features * support_kp_repres.reshape(1, N, C, 1, 1)).sum(dim=2) / (q_norm * s_norm.view(1, N, 1, 1) + 1e-6)
                # attention_map_l2np = attention_map_l2.cpu().detach().numpy()
                attention_map_final = alpha * attention_map_l2 + (1-alpha) * attention_map
                attention_map_l2 = torch.clamp(attention_map_final, min=0, max=1.0)

        elif fused_attention == 'cosine':
            # normalize
            support_kp_repres = torch.nn.functional.normalize(support_kp_repres, p=2, dim=1, eps=1e-12)  # N x C
            query_features = torch.nn.functional.normalize(query_features, p=2, dim=1, eps=1e-12)  # B2 x C x H x W
            # dot-product
            attentive_features = query_features.reshape(B2, C, H*W).unsqueeze(dim=1) * support_kp_repres.reshape(1, N, C, 1)  # B2 x N x C x (H*W)
            attentive_features = attentive_features.sum(dim=2)  # B2 x N x (H*W)
            attentive_features = attentive_features.reshape(B2, N, 1, H, W)  # B2 x N x 1 x H x W
            if output_attention_maps == True:
                # B2 x N x H x W
                attention_map_l2 = torch.clamp(attentive_features.reshape(B2, N, H, W), min=0, max=1.0)

        elif fused_attention == 'softmax':
            # dot-product
            attentive_features = query_features.reshape(B2, C, H*W).unsqueeze(dim=1) * support_kp_repres.reshape(1, N, C, 1)  # B2 x N x C x (H*W)
            scale = C ** -0.5
            attentive_features = attentive_features.sum(dim=2) * scale  # B2 x N x (H*W)
            # softmax
            attentive_features = torch.softmax(attentive_features, dim=2)
            attentive_features = attentive_features.reshape(B2, N, 1, H, W)  # B2 x N x 1 x H x W
            if output_attention_maps == True:
                # B2 x N x H x W
                if output_attention_maps == True:
                    # B2 x N x H x W
                    attention_map_l2 = torch.clamp(attentive_features.reshape(B2, N, H, W), min=0, max=1.0)
        else:
            print('Error for fused_attention in AttentionBasedModulator.')
            exit(0)

        # using keypoint protos, the attentive features size in fact is B2 x N x C x H x W or B2 x N x 1 x H x W
        if is_kp_protos == True:
            pass
        elif is_kp_protos == False:  # in fact attentive features size is B2 x (B1*N) x C x H x W or B2 x (B1*N) x 1 x H x W
            # attentive_features = attentive_features.reshape(B2, B1, N_categories, -1, H, W).mean(dim=2)
            attentive_features = attentive_features.reshape(B2, B1, N_categories, -1, H, W)
            kp_mask_temp = torch.clone(support_kp_mask)
            kp_weight = torch.sum(support_kp_mask, dim=0)  # N
            for i in range(N_categories):
                if kp_weight[i] > 0:
                    kp_weight[i] = kp_weight[i].reciprocal()  # 1.0 / B
                else:  # equal to 0
                    kp_weight[i] = 1.0 / B1
                    kp_mask_temp[:, i] = 1.0  # in order to avoid zero for final summation
            mask = kp_mask_temp.reshape(1, B1, N_categories, 1, 1, 1)
            # B2 x N x C x H x W or B2 x N x 1 x H x W
            attentive_features = (attentive_features * mask).sum(dim=1) * kp_weight.view(1, N_categories, 1, 1, 1)

            if output_attention_maps == True:
                attention_map_l2 = attention_map_l2.reshape(B2, B1, N_categories, H, W) \
                                   * support_kp_mask.view(1, B1, N_categories, 1, 1)
                attention_map_l2 = attention_map_l2.sum(dim=1) / (torch.sum(support_kp_mask, dim=0) + 1e-6).view(1, N_categories, 1, 1)

        if output_attention_maps == False:
            attention_maps_l2_cpu = None
        else:
            # copy attention maps, tensor, B2 x N x H x W
            attention_maps_l2_cpu = copy.deepcopy(attention_map_l2.cpu().detach())

        return attentive_features, attention_maps_l2_cpu

def fibers_augmentation(s_repres, support_fibers, support_kp_mask, sample_type=1):
    '''
    s_repres: B x C x N
    support_fibers: C x N (kp prototype)
    support_kp_mask: B x N
    '''
    B1, C, N = s_repres.shape
    union_s_kp_mask = support_kp_mask.sum(0)  # N
    if (sample_type) == 0:  # no sampling noise
        pass
    else:
        if B1 == 1:  # 1shot---> augemnted to 5 shots
            # s_repres = s_repres.repeat(5, 1, 1)  # 5 x C x N
            # s_repres[1:, :, :] = s_repres[1:, :, :] + torch.randn(4, C, N).cuda() * 0.001
            s_repres = support_fibers.unsqueeze(0).repeat(5, 1, 1)  # 5 x C x N (use previously computed avg proto for augmentation)
            s_repres[1:, :, :] = s_repres[1:, :, :] + torch.randn(4, C, N).cuda() * 0.001

            B1 = 5  # update num supports
            support_kp_mask = support_kp_mask.repeat(5, 1)  # 5 x N
            # support_fibers = s_repres.mean(0)  # C x N, update kp protos
        else:
            if sample_type == 1:  # feature aug., independent noise (only 1shot-->B1)
                for i in range(N):
                    if union_s_kp_mask[i] == 1:
                        # ind = (support_kp_mask[:, i]).tolist().index(1)
                        # repres_tmp = (s_repres[ind, :, i]).unsqueeze(0).repeat(B1, 1)
                        # s_repres[:, :, i] = repres_tmp
                        # for j in range(B1):
                        #     if j != ind:
                        #         s_repres[j, :, i] = s_repres[j, :, i] + torch.randn(C).cuda() * 0.001
                        ind = (support_kp_mask[:, i]).tolist().index(1)
                        repres_tmp = support_fibers[:, i]  # C (use previously computed avg proto for augmentation)
                        for j in range(B1):
                            if j != ind:
                                s_repres[j, :, i] = repres_tmp + torch.randn(C).cuda() * 0.001
                        support_kp_mask[:, i] = 1
                # support_fibers = average_representations2(s_repres, support_kp_mask)  # C x N, update kp protos
            elif (sample_type) == 2:  # feature aug., independent noise (<B1 --> B1)
                for i in range(N):
                    if (union_s_kp_mask[i] < B1) and (union_s_kp_mask[i] > 0):
                        for j in range(B1):
                            if support_kp_mask[j, i] == 0:
                                s_repres[j, :, i] = support_fibers[:, i] + torch.randn(C).cuda() * 0.001
                        support_kp_mask[:, i] = 1
                # support_fibers = s_repres.mean(0)  # C x N, update kp protos
            elif (sample_type) == 3:  # feature aug., covariant noise (<B1 --> B1)
                s_repres_dev = s_repres - support_fibers.unsqueeze(0)  # B1 x C x N, (phi - uc)
                s_repres_dev = s_repres_dev * support_kp_mask.unsqueeze(1)  # B1 x C x N
                s_repres_dev = s_repres_dev.permute(2, 1, 0)  # N x C x B1, (phi - uc)
                s_repres_dev = s_repres_dev.detach().to(torch.float64)
                U, S, V = torch.svd(s_repres_dev, some=True)  # batch SVD decomposition
                union_s_kp_mask_tmp = union_s_kp_mask + (union_s_kp_mask == 0).long()  # N, avoid 0 valid samples
                S_tmp = (S) / union_s_kp_mask_tmp.sqrt().unsqueeze(1)  # N x k, divide num_valid support samples per class
                # sample noise
                noise = torch.randn(N, S.shape[1], B1).cuda() * S_tmp.unsqueeze(-1)  # N(0, S^2 / num_valid), N x k x B1
                # project the noise to high dimension
                noise_project_back = torch.bmm(U, noise).to(torch.float32)  # N x C x B1
                noise_project_back = noise_project_back.permute(2, 1, 0)    # B1 x C x N
                aug_fibers = noise_project_back + support_fibers.unsqueeze(0)  # B1 x C x N
                mask_tmp = support_kp_mask.unsqueeze(1)
                s_repres = mask_tmp * s_repres + (1-mask_tmp) * aug_fibers  # B1 x C x N
                # support_fibers = s_repres.mean(0)  # C x N, update kp protos
                support_kp_mask = ((union_s_kp_mask > 0).long()).unsqueeze(0).repeat(B1, 1)  # B1 x N
            else:
                raise NotImplementedError

    return s_repres, support_kp_mask

def fibers_centering(s_repres, support_fibers):
    '''
    s_repres: B x C x N
    support_fibers: C x N (kp prototype)
    '''
    s_repres_dev = s_repres - support_fibers.unsqueeze(0)  # B1 x C x N, (phi - uc)
    return s_repres_dev


def get_subspaces_by_svd(centered_points, mask, back_propagate_svd=False):
    '''
    Get class-specific subspaces.
    centered_points: B x C x N (N classes, each class has B samples, each sample with C dim)
    mask: B x N (mask or normalized probability for each point)
    '''
    B, N = mask.shape
    centered_points = centered_points * mask.unsqueeze(1)  # B x C x N
    centered_points = centered_points.permute(2, 1, 0)  # N x C x B, (phi - uc)
    union_mask = mask.sum(0)  # N
    union_mask_tmp = union_mask + (union_mask == 0).long()  # N, avoid 0 valid samples by adding 1
    # centered_points = centered_points / union_mask_tmp.float().sqrt().reshape(N, 1, 1)  # divide num of valid samples
    union_mask_tmp2 = mask / union_mask_tmp.unsqueeze(0)   # B x N, each sample has a probability
    union_mask_tmp2 = union_mask_tmp2.float().sqrt()        # B x N
    centered_points = centered_points * union_mask_tmp2.permute(1, 0).reshape(N, 1, B)

    if back_propagate_svd == False:
        centered_points = centered_points.detach().to(torch.float64)  # N x C x B
    else:
        centered_points = (centered_points+1e-12).to(torch.float64)
    U, S, V = torch.svd(centered_points, some=True)  # batch SVD decomposition

    # U: N x C x k; S: N x k
    return U, S


def get_distance_by_subspaces(mean, query_features, subspaces):
    '''
    mean: C x N (N classes, each has C dim)
    query_features: B x C x H x W
    subspaces: N x C x k (class-specific subspaces, N classes and each with k base vectors)
    '''
    B2, C, H, W = query_features.shape
    _, N = mean.shape

    distance_map = torch.zeros(B2, N, H, W).cuda()
    mean = mean.transpose(1, 0)  # N x C
    for i in range(B2):
        each_query_feat = query_features[i]  # C x H x W
        each_query_feat_dev = each_query_feat.view(1, C, H*W) - mean.view(N, C, 1)  # N x C x (H*W)
        # exp( -||(I-UU^T)(x-uc)||^2 / T ), using subspace
        vecs_tmp = torch.bmm(subspaces.transpose(2, 1), each_query_feat_dev)
        vecs_tmp = torch.bmm(subspaces, vecs_tmp)
        projected_vecs = each_query_feat_dev - vecs_tmp  # N x C x (H*W)
        projected_vecs = (projected_vecs ** 2).sum(1)  # N x (H*W)
        projected_vecs = projected_vecs.reshape(N, H, W)

        distance_map[i, :, :, :] = projected_vecs[:, :, :]

    return distance_map  # B2 x N x H x W

def get_distance_by_subspaces_wo_centering(query_features, subspaces):
    '''
    query_features: B x C x H x W
    subspaces: N x C x k (class-specific subspaces, N classes and each with k base vectors)
    '''
    B2, C, H, W = query_features.shape
    N = subspaces.shape[0]

    distance_map = torch.zeros(B2, N, H, W).cuda()
    for i in range(B2):
        each_query_feat = query_features[i]  # C x H x W
        each_query_feat_dev = each_query_feat.reshape(1, C, H*W).expand(N, C, H*W)  # N x C x (H*W)
        # exp( -||(I-UU^T)x||^2 / T ), using subspace
        vecs_tmp = torch.bmm(subspaces.transpose(2, 1), each_query_feat_dev)
        vecs_tmp = torch.bmm(subspaces, vecs_tmp)
        projected_vecs = each_query_feat_dev - vecs_tmp  # N x C x (H*W)
        projected_vecs = (projected_vecs ** 2).sum(1)  # N x (H*W)
        projected_vecs = projected_vecs.reshape(N, H, W)

        distance_map[i, :, :, :] = projected_vecs[:, :, :]

    return distance_map  # B2 x N x H x W

def get_dot_product_similarity(mean_norm, query_features_norm):
    '''
    mean: C x N (N classes, each has C dim)
    query_features: B x C x H x W
    '''
    C, N = mean_norm.shape
    mean_tmp = mean_norm.permute(1, 0).unsqueeze(0).unsqueeze(-1)  # 1 x N x C x 1
    B2, _, H, W = query_features_norm.shape
    query_features = query_features_norm.reshape(B2, C, H*W)
    attention_map_l2 = query_features.unsqueeze(dim=1) * mean_tmp  # B2 x N x C x (H*W)
    attention_map_l2 = attention_map_l2.sum(dim=2)  # B2 x N x (H*W)
    attention_map_l2 = attention_map_l2.reshape(B2, N, H, W)
    # attention_map_l2np = attention_map_l2.detach().cpu().numpy()

    return attention_map_l2  # B2 x N x H x W