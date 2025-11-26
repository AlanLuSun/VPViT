import os
import time

import PIL.Image
import numpy as np
import scipy.stats
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import argparse
import json
import random
from einops import rearrange
from heatmap import putGaussianMaps
import torch
import torch.nn as nn
from collections import namedtuple

def mkdir(path):
    if os.path.exists(path):
        print("---  the folder already exists  ---")
    else:
        os.makedirs(path)

# the function for printing neural network's weights
print_weight_cnt = 0
def print_weights(data : torch.Tensor, mode='a'):
    global print_weight_cnt
    if print_weight_cnt == 0:
        fout = open('weight.txt', 'w')  # just open-close to clear previous content
        fout.close()
    print_weight_cnt += 1
    buffer_str = str(data.cpu().detach().numpy().copy())
    with open('weight.txt', mode) as fout:
        fout.write('=============times {}=============\n'.format(print_weight_cnt))
        fout.write(str(buffer_str))
        fout.write('\n')
        fout.close()

def image_normalize(image, denormalize=False, copy=False):
    '''
    image: H x W x C, image pixel's value range should be 0~1
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if copy == True:
        image = image.clone() if isinstance(image, torch.Tensor) else image.copy()
    if denormalize == False:  # noralize
        for channel in range(3):
            image[:, :, channel] = (image[:, :, channel] - mean[channel]) / std[channel]
    else:  # de-normalize (after de-normalize, range is 0~1)
        for channel in range(3):
            image[:, :, channel] = image[:, :, channel] * std[channel] + mean[channel]
    return image

def batch_images_normalize(images, denormalize=False, copy=False):
    '''
    image: B x C x H x W, pixel value's range should be 0~1
    '''
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if copy == True:
        images = images.clone() if isinstance(images, torch.Tensor) else images.copy()
    if denormalize == False:  # noralize
        for channel in range(3):
            images[:, channel, :, :] = (images[:, channel, :, :] - mean[channel]) / std[channel]
    else:  # de-normalize
        for channel in range(3):
            images[:, channel, :, :] = images[:, channel, :, :] * std[channel] + mean[channel]
    return images

def make_grid_images(tensor_image, denormalize=True, save_path=None):
    '''
    :param tensor: B x C x H x W, pixel value should be ranges from 0~1 !!!
    :param denormalize:
    :param save_path:
    :return: grid_image, H' x W' x 3
    '''
    # vmax, vmin = torch.max(tensor_image), torch.min(tensor_image)
    image_temp = torch.clone(tensor_image)
    # de-normalize
    if denormalize == True:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        for channel in range(3):
            image_temp[:, channel, :, :] = image_temp[:, channel, :, :] * std[channel] + mean[channel]
    grid_image = torchvision.utils.make_grid(image_temp, scale_each=0.2)  # 3 x H x W, make_grid will output image with 3 channels
    grid_image = grid_image.permute(1, 2, 0)  # H x W x 3
    if save_path != None:
        grid_image = grid_image.cpu().detach().numpy()[:, :, ::-1]  # convert RGB image to BGR image
        cv2.imwrite(save_path, grid_image * 255)  # convert 0~1 to be 0~255

    return grid_image

def save_batch_images(tensor_images, save_path, prefix='', suffix=''):
    '''
    tensor_images (RGB or single-channel format, 0~255): B x C x H x W
    '''
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    B, C, H, W = tensor_images.shape
    ims_temp = tensor_images.permute(0, 2, 3, 1)  # B x H x W x C
    ims_temp = ims_temp.numpy()
    for i in range(B):
        each_im = ims_temp[i]
        p = os.path.join(save_path, '%s_%s_%s.png'%(prefix, i, suffix))
        if C == 3:  # RGB
            cv2.imwrite(p, each_im[:, :, [2, 1, 0]])
        elif C == 1:  # grayscale
            cv2.imwrite(p, each_im[:, :, 0])

def make_uncertainty_map(sigmas_np, B):
    # sigmas: (B*N) * (L*L), numpy
    # we want to make B images and each image shows N keypoint hotspots.
    W, H = 368, 368
    sigmas = sigmas_np  # sigmas_tensor.cpu().detach().numpy()
    N = int(sigmas.shape[0] / B)
    L = int(np.sqrt(sigmas.shape[1]))
    im_combined = np.zeros((B, H, W))
    for im_j in range(B):
        for i in range(im_j * N, (im_j + 1) * N):
            im = sigmas[i, :].reshape(L, L)
            im_resize = cv2.resize(im, (W, H), interpolation=cv2.INTER_CUBIC)
            vmin, vmax = np.min(im_resize), np.max(im_resize)
            # print(vmin, vmax)
            if vmax > vmin:
                # im_resize = (im_resize - vmin) / (vmax-vmin)  # normalize to 0~1
                im_resize = (im_resize - vmin) / (vmax)
            # merging using max operation
            ind = im_combined[im_j] < im_resize
            im_combined[im_j][ind] = im_resize[ind]
    vmin, vmax = np.min(im_combined), np.max(im_combined)
    # plt.imshow(im_combined[0])
    # plt.show()

    return im_combined  # B x H x W


def save_plot_image(im, save_path, does_show=False):
    '''
    :param im: H x W x 3 or H x W , numpy
    :return:
    '''
    H, W = im.shape[0], im.shape[1]
    fig, ax = plt.subplots()
    # fig = plt.figure()
    # ax = fig.gca()
    fig.tight_layout()
    fig.patch.set_alpha(0.)  # set the figure face to be transparent
    # im = Image.open("/home/changsheng/LabDatasets/AnimalPoseDataset-2019WS-CDA/Animal_Dataset_Combined/images/dog/do86.jpeg").convert('RGB')
    # im = im.resize((square_image_length, square_image_length), PIL.Image.BILINEAR)
    # plt.imshow(im, cmap=plt.cm.jet)
    plt.imshow(im, cmap=plt.cm.viridis)  # default
    # plt.imshow(im)
    # plt.show()

    # remove ticks but the frame still exists
    plt.xticks([])
    plt.yticks([])
    ax.invert_yaxis()

    # Remove the white margin around image
    fig.set_size_inches(W / 100.0, H / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)

    plt.savefig(save_path)
    if does_show == True:
        plt.show()

def compute_eigenvalues(covar):
    '''
    :param covar: 2 x 2
    :return: eigenvalues, eigenvectors, orientation
    '''
    # eigenvalues e: 2 x 2, each row is an eigenvalue e[i,0]+j*e[i, 1],
    # eigenvectors v: 2 x 2, each column is a corresponding eigenvector v[:, i]
    e, v = torch.eig(covar, eigenvectors=True)
    _, indices = torch.sort(e[:, 0], descending=True, dim=0)
    e2, v2 = e[indices, :], v[:, indices]
    radian = torch.atan2(v2[1, 0], v2[0, 0])  # atan2(vy, vx)
    angle = radian / 3.1415926 * 180  # orientation of the eigenvector for major axis
    return e2, v2, angle

def mean_confidence_interval(accs, confidence=0.95):
    '''
    compute mean and standard error of mean for a sequence of observations
    using t-test
    '''
    if isinstance(accs, np.ndarray) == False:
        accs = np.array(accs)

    n = accs.shape[0]
    if n == 1:
        return accs[0], 0
    m, se = np.mean(accs), scipy.stats.sem(accs)  # sem = standard error of mean = sigma / sqrt(n)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)  # ppf here is the inverse of cdf (cumulative distributin function)
    return m, h

def mean_confidence_interval_multiple(accs_multiple, confidence=0.95):
    '''
    accs_multiple: K x N, K rows, each row will compute mean_confidence_interval
    '''
    K = len(accs_multiple)
    mean, interval = np.zeros(K), np.zeros(K)
    for i in range(K):
        mean[i], interval[i] = mean_confidence_interval(np.array(accs_multiple[i]), confidence=confidence)

    return mean, interval

def load_samples(ann_json_files, local_json_root):
    '''
    ann_json_files: a list
    local_json_root: a path
    return: a list of samples
    '''
    samples = []
    for p in ann_json_files:
        annotation_path = os.path.join(local_json_root, p)
        with open(annotation_path, 'r') as fin:
            # self.samples = json.load(fin)
            samples_temp = json.load(fin)
            # self.samples = dataset['anns']
            fin.close()
        samples += samples_temp

    return samples

def power_norm1(x, SIGMA):
    out = 2/(1 + torch.exp(-SIGMA*x)) - 1
    return out

def power_norm2(x, SIGMA):
    out = torch.sign(x) * torch.abs(x).pow(SIGMA)
    return out





def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)



# count model learnable parameters
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000



def get_patches(ims: torch.Tensor, patch_size=(32, 32), save=False, prefix='s', saveroot='./episode_images/patched_ims'):
    '''
    ims: cpu Tensor, B x C x H x W
    return patched_ims: B x (grid_h*grid_w) x C x p1 x p2
    '''
    if save:
        if os.path.exists(saveroot) == False:
            os.makedirs(saveroot)
    B, _, H, W = ims.shape
    patched_ims = rearrange(ims, 'B C (h p1) (w p2) -> B C (h w) p1 p2', p1=patch_size[0], p2=patch_size[1])
    patched_ims = patched_ims.permute(0, 2, 1, 3, 4)  # B x (grid_h*grid_w) x C x p1 x p2

    p = patch_size[0]
    h = H // p
    mask1 = (patched_ims).flatten(2).mean(dim=2)  # B x (grid_h*grid_w) x C
    mask1 = mask1.reshape(B, h, h, -1)

    # mask2 = torch.nn.functional.avg_pool2d(ims, kernel_size=p, stride=p, padding=0)
    # mask2 = mask2.permute(0, 2, 3, 1)  # B x h x h x C
    #
    # mask3 = torch.nn.functional.interpolate(ims, size=(h, h))
    # mask3 = mask3.permute(0, 2, 3, 1)  # B x h x h x C

    if save:
        for i in range(B):
            # grid_iamge: C x H' x W'
            grid_image = torchvision.utils.make_grid(patched_ims[i], nrow=W // patch_size[1], padding=2, normalize=False, pad_value=0.8)
            grid_image = grid_image.permute(1, 2, 0)
            grid_image = grid_image.numpy()[:, :, ::-1]
            cv2.imwrite(os.path.join(saveroot, prefix+'_'+str(i)+'.jpg'), grid_image * 255)

            t = mask1.numpy()[:, :, ::-1]
            cv2.imwrite(os.path.join(saveroot, prefix+'_'+str(i)+'_avg1.jpg'), t * 255)
            # t = mask2.numpy()[:, :, ::-1]
            # cv2.imwrite(os.path.join(saveroot, prefix + '_' + str(i) + '_avg2.jpg'), t * 255)
            # t = mask3.numpy()[:, :, ::-1]
            # cv2.imwrite(os.path.join(saveroot, prefix + '_' + str(i) + '_dsize.jpg'), t * 255)


    return patched_ims

def draw_contours(thresh, mask, im, color='pink', thickness=1):
    '''
    mask: H x W
    im  : H x W x 3
    '''
    # thresh = int(0.1 * 255)
    H, W = im.shape[0:2]
    ret, imbinary = cv2.threshold(mask, thresh, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(imbinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(type(contours))  # a list of coordinates

    # im_out = np.copy(im)
    # if len(im_out.shape) == 2 or (len(im_out.shape) == 3 and im_out.shape[2] == 1):  # single channel
    #     im_out = np.repeat(im_out[..., np.newaxis], repeats=3, axis=2)  # expand

    im_out = np.zeros((H, W, 3))
    if len(im.shape) == 2 or (len(im.shape) == 3 and im.shape[2] == 1):  # single channel
        im = im.reshape(H, W)
        im_out[:, :, 0] = im[:, :]
        im_out[:, :, 1] = im[:, :]
        im_out[:, :, 2] = im[:, :]
    else:
        im_out[:, :, :] = im[:, :, :]

    if color == 'pink':
        c = (255, 0, 255)
    elif color == 'red':
        c = (255, 0, 0)
    elif color == 'green':
        c = (0, 255, 0)
    elif color == 'blue':
        c = (0, 0, 255)
    elif color == 'white':
        c = (255, 255, 255)
    cv2.drawContours(im_out, contours, -1, c, thickness=thickness)

    return im_out

def rgb2gray(ims, copy=False):
    # ims: B x 3 x H x W in RGB format
    if copy == True:
        ims = ims.clone() if isinstance(ims, torch.Tensor) else ims.copy()
    ims = 0.299 * ims[:, 0] + 0.587 * ims[:, 1] + 0.114 * ims[:, 2]
    return ims  # B x H x W

def ele_max(a, b=0):
    # sign = (a >= b).detach().float()
    # return sign * a + (1 - sign) * b
    return torch.clamp(a, min=b)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def get_model_summary(model, inputs, item_length=26, verbose=True):
    """
    :param model: an object which inherits nn.Module
    :param inputs: a list which contains input parameters
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            # here we can also add other modules if we need to compute their params & FLOPs,
            # added by Changsheng Lu, 2023.01.27
            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                # Lienar layer B X (.) X C1 --> B X (.) X C2, complexity is B x (.) X C2 x C1,
                # modified by Changsheng Lu, 2023.01.27
                flops = (torch.prod(torch.LongTensor(list(output.size()))) * input[0].size(-1)).item()
            elif class_name.find('SalAttention') != -1:  # only used in my SalViT based FSKD, added by Changsheng Lu, 2023.01.27
                flops = (torch.prod(torch.LongTensor(list(input[0].size()))) * input[0].size(1) * 2).item()
                output = output[0]
            else:
                pass

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            if flops != 'Not Available':  # skip unknown modules, modified by Changsheng Lu, 2023.01.27
                summary.append(
                    ModuleDetails(
                        name=layer_name,
                        input_size=list(input[0].size()),
                        output_size=list(output.size()),
                        num_parameters=params,
                        multiply_adds=flops)
                )
            else:
                summary.append(
                    ModuleDetails(
                        name=layer_name,
                        input_size='unknown',
                        output_size='unknown',
                        num_parameters=0,
                        multiply_adds=0)
                )
                # pass

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*inputs)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep
    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,}".format(flops_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


def display_all_args(args):
    '''
    args: the parsed args, type(args) == argparse.Namespace
    '''
    print('==========================================')
    print('Display all the hyper-parameters in args:')
    print('------------------------------------------')
    for arg in vars(args):
        value = getattr(args, arg)
        # if value is not None:
        print('%s: %s' % (str(arg), str(value)))
    print('==========================================')

def display_dict(d):
    print('==========================================')
    print('Display key-value pairs in dict:')
    print('------------------------------------------')
    for key in d:
        print('%s: %s'%(key, d[key]))
    print('==========================================')

def list2str(l):
    return '-'.join([str(v) for v in l])

class DTModule(object):
    '''
    Distance Transform (DT).
    The whole routine runs cpu Tensor and depends on OPENCV2
    https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#ga8a0b7fdfcb7a13dde018988ba3a43042
    '''
    def __init__(self, dt_type=0, fg_thresh=0.4, im_size=(384, 384)):
        '''
        dt_type: 0, no DT applied
                 1~3, single-point diffusion map and variants
                 4~6, shape diffusion map (whole_edt) and vairants
        fg_thresh: 0.0 ~ 1.0
        '''
        self.dt_type = dt_type
        self.fg_thresh = int(fg_thresh * 255)

        H, W = im_size
        self.max_dist = np.sqrt(H ** 2 + W ** 2)
        if dt_type >= 1 and dt_type <= 3:  # used in single-point diffusion map
            coords_h = torch.arange(H).float()
            coords_w = torch.arange(W).float()
            self.coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2 x H x W
            self.coords = self.coords.reshape(1, 2, H, W)  # 1 x 2 x H x W
        print('DT type (%d) Used!' % self.dt_type)

    def single_point_diffusion_map(self, gray_ims, im_min, im_max):
        # gray_ims is a tensor with size of B x 1 x H x W, and pixel value of 0~1
        # im_min, im_max are also within the range of 0~1
        B, _, H, W = gray_ims.shape
        sal_sum = gray_ims.view(B, -1).sum(dim=1).view(B, 1)  # B x 1
        centroids = (self.coords * gray_ims).reshape(B, 2, H*W).sum(dim=2) / (sal_sum + 1e-6)  # B x 2
        centroids = centroids.view(B, 2, 1, 1)
        dist_field = ((self.coords - centroids) ** 2).sum(dim=1).sqrt()  # B x H x W
        dist_field /= self.max_dist
        rev_dist_field = 1 - dist_field  # B x H x W

        # contrast-preserving normalization
        rev_dist_field_t = rev_dist_field.view(B, -1)  # B x (H*W)
        rev_dist_min, rev_dist_max = torch.min(rev_dist_field_t, dim=1)[0], torch.max(rev_dist_field_t, dim=1)[0]
        rev_dist_min, rev_dist_max = rev_dist_min.view(B, 1, 1), rev_dist_max.view(B, 1, 1)
        rev_dist_field = (rev_dist_field - rev_dist_min) / (rev_dist_max - rev_dist_min + 1e-6) \
                         * (im_max - im_min).view(B, 1, 1) + im_min.view(B, 1, 1)
        rev_dist_field = rev_dist_field.reshape(B, 1, H, W)

        return rev_dist_field  # B x 1 x H x W

    def dt(self, gray_ims: torch.Tensor):
        # gray_im is a tensor with size of B x 1 x H x W, and value of 0~1
        # single-point diffusion map
        # return disance transformed map (tensor) with size of B x 1 x H x W
        if self.dt_type <= 0:
            return gray_ims

        B, _, H, W = gray_ims.shape
        gray_ims_t = gray_ims.reshape(B, H*W)
        im_min, im_max = gray_ims_t.min(dim=1)[0], gray_ims_t.max(dim=1)[0]  # size B, value range: 0~1
        if self.dt_type >= 1 and self.dt_type <= 3:  # used in single-point diffusion map
            rev_dist_field = self.single_point_diffusion_map(gray_ims, im_min, im_max)
            # torch.save(rev_dist_field, 'a2.pt')

        if self.dt_type == 1:
            return rev_dist_field  # B x 1 x H x W, tensor
        final_map = torch.zeros(gray_ims.shape)

        # compute fg_edt (foreground Euclidean Distance Transform)
        if self.dt_type >= 2 and self.dt_type <= 6: # 2,3; 4, 5, 6
            # numpy image with value range 0~255
            np_ims = (gray_ims *255).long().numpy().astype(np.uint8)
            np_ims = np_ims.reshape(B, H, W)
            # fg_thresh_f = self.fg_thresh / 255.0
            for i in range(B):
                # 1) compute binary image based on saliency map
                ret, imbinary = cv2.threshold(np_ims[i], thresh=self.fg_thresh, maxval=255, type=cv2.THRESH_BINARY)
                # 2) fg_edt
                fg_edt = cv2.distanceTransform(imbinary,distanceType=cv2.DIST_L2,maskSize=cv2.DIST_MASK_PRECISE)

                #----------------------------------------------------------------
                # 3) normalize fg_edt
                fg_edt_max_value, fg_edt_min_value = fg_edt.max(), fg_edt.min()
                fg_edt_normalize = (fg_edt - fg_edt_min_value) / (fg_edt_max_value - fg_edt_min_value + 1e-6)  # 0~1, fg
                fg_edt_normalize = torch.tensor(fg_edt_normalize)  # H x W
                # 4) bg_edt
                if self.dt_type >= 4 and self.dt_type <= 6:
                    bg_edt = cv2.distanceTransform(255-imbinary,distanceType=cv2.DIST_L2,maskSize=cv2.DIST_MASK_PRECISE)
                    bg_edt_max_value, bg_edt_min_value = bg_edt.max(), bg_edt.min()
                    bg_edt_normalize = (bg_edt_min_value - bg_edt) / (bg_edt_max_value - bg_edt_min_value + 1e-6)  # -1~0, bg
                    bg_edt_normalize = torch.tensor(bg_edt_normalize)  # H x W

                    # contrast-preserving normalization
                    each_im_min, each_im_max = im_min[i].numpy(), im_max[i].numpy()
                    whole_edt_normalize = (fg_edt_normalize + bg_edt_normalize) / 2 + 0.5
                    # whole_edt_normalize = fg_edt_normalize * (1-fg_thresh_f) + (bg_edt_normalize+1) * fg_thresh_f # fg ranges from fg_thresh_f~1; bg ranges from 0~fg_thresh_f
                    whole_edt2 = whole_edt_normalize * (each_im_max-each_im_min) + each_im_min

                # # 3) bg_edt
                # if self.dt_type >= 4 and self.dt_type <= 6:
                #     bg_edt = cv2.distanceTransform(255-imbinary,distanceType=cv2.DIST_L2,maskSize=cv2.DIST_MASK_PRECISE)
                #     whole_edt = (fg_edt - bg_edt)
                #     # contrast-preserving normalization
                #     edt_min, edt_max = np.min(whole_edt), np.max(whole_edt)
                #     each_im_min, each_im_max = im_min[i].numpy(), im_max[i].numpy()
                #     whole_edt2 = (whole_edt - edt_min)/(edt_max-edt_min+1e-6) * (each_im_max-each_im_min) + each_im_min
                #     whole_edt2 = torch.tensor(whole_edt2)
                # # 4) normalize fg_edt
                # fg_edt_max_value = np.max(fg_edt)
                # fg_edt_normalize = fg_edt / (fg_edt_max_value + 1e-6)
                # fg_edt_normalize = torch.tensor(fg_edt_normalize)  # H x W
                # ----------------------------------------------------------------

                if self.dt_type == 2:  # single-point diffusion map + fg_edt
                    final_map[i, 0] = (rev_dist_field[i,0] + fg_edt_normalize) / 2
                elif self.dt_type == 3:  # single-point diffusion map + fg_edt + saliency
                    final_map[i, 0] = (rev_dist_field[i,0] + fg_edt_normalize + gray_ims[i,0]) / 3
                elif self.dt_type == 4:  # shape diffusion map
                    final_map[i, 0] = whole_edt2
                elif self.dt_type == 5:  # shape diffusion map + fg_edt
                    final_map[i, 0] = (whole_edt2 * 1/2 + fg_edt_normalize * 1/2)
                elif self.dt_type == 6:  # shape diffusion map + fg_edt + saliency
                    final_map[i, 0] = (whole_edt2 + fg_edt_normalize + gray_ims[i, 0]) / 3
                else:
                    raise NotImplementedError

        # torch.save(final_map, 'b2.pt')
        return final_map  # B x 1 x H x W, tensor

class GaussianBlur(object):
    def __init__(self, sigma, use_cuda=False):
        '''
        Automatic determine filtering kernel (~+-3sigma) by sigma
        '''
        self.sigma = sigma
        if self.sigma > 0:  # Gaussian blur
            stride = 1
            start = stride / 2 - 0.5
            kernel_w_half = round(3 * sigma / stride)
            kernel_w = 2 * kernel_w_half + 1
            kernel_center = torch.Tensor([0.5, 0.5]).mul(kernel_w - 1)
            gaussian_kernel = torch.zeros(kernel_w, kernel_w)  # gaussian kernel
            if use_cuda and torch.cuda.is_available():
                kernel_center = kernel_center.cuda()
                gaussian_kernel = gaussian_kernel.cuda()
            gaussian_kernel[:, :], return_flag = putGaussianMaps(kernel_center*stride + start, sigma, kernel_w, kernel_w, stride, normalization=True)
            self.gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_w, kernel_w)  # 1 x 1 x kernel_w x kernel_w
            self.kernel_w_half = kernel_w_half
            # print(self.gaussian_kernel)
        print('Gaussian Blur (%f) Used!'%self.sigma)

    def gaussian_blur(self, inputs):
        '''
        inputs: B x 1 x H x W, cuda tensor
        '''
        if self.sigma <= 0:  # no blur applied
            return inputs
        out = torch.nn.functional.conv2d(inputs, self.gaussian_kernel, bias=None, stride=1, padding=self.kernel_w_half)
        return out

class GaussianBlurV2(object):
    def __init__(self, sigma, use_cuda=False):
        '''
        Automatic determine filtering kernel (~+-3sigma) by sigma
        '''
        self.sigma = sigma
        if self.sigma > 0:  # Gaussian blur
            stride = 1
            start = stride / 2 - 0.5
            kernel_w_half = round(3 * sigma / stride)
            kernel_w = 2 * kernel_w_half + 1
            kernel_center = torch.Tensor([0.5, 0.5]).mul(kernel_w - 1)
            gaussian_kernel = torch.zeros(kernel_w, kernel_w)  # gaussian kernel
            if use_cuda and torch.cuda.is_available():
                kernel_center = kernel_center.cuda()
                gaussian_kernel = gaussian_kernel.cuda()
            gaussian_kernel[:, :], return_flag = putGaussianMaps(kernel_center*stride + start, sigma, kernel_w, kernel_w, stride, normalization=True)
            self.gaussian_kernel = gaussian_kernel.reshape(1, 1, kernel_w, kernel_w)  # 1 x 1 x kernel_w x kernel_w
            self.kernel_w_half = kernel_w_half
            self.kernel_w = kernel_w
            # print(self.gaussian_kernel)
        print('Gaussian Blur (%f) Used!'%self.sigma)

    def __call__(self, inputs):
        '''
        inputs: B x C x H x W, torch tensor
        '''
        if self.sigma <= 0:  # no blur applied
            return inputs
        C = inputs.shape[1]
        gaussian_kernel = self.gaussian_kernel.expand(C, 1, self.kernel_w, self.kernel_w)
        out = torch.nn.functional.conv2d(inputs, gaussian_kernel, bias=None, stride=1, padding=self.kernel_w_half, groups=C)
        return out

def sample_even_points(sampling_freq=12, is_cuda=False):
    '''
    sampling (sampling_freq*sampling_freq) points with each point (x, y) in range -1~1
    '''
    W = sampling_freq
    yy, xx = torch.meshgrid(torch.arange(0, W), torch.arange(0, W))
    grids = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=-1)  # (H*W) x 2
    grids = ((grids + 0.5) / W - 0.5) * 2  # -1 < x,y < 1
    if is_cuda:
        grids = grids.cuda()
    return grids  # (H*W) x 2

def sample_keypoints_within_rect(nkps=20, im_h=384, im_w=384, bbx=None, sample_method='random'):
    '''
    nkps: the keypoints to sample
    im_h, im_w: image height and width
    bbx: if not None, sample nkps within bbx (xmin, ymin, w, h)
    sample_method: 'random' or 'regular_grid'

    return grids: (nkps) x 2, each row is a point (x, y)
    '''
    if bbx is None:
        bbx = (0, 0, im_w, im_h)  # (xmin, ymin, w, h)
    xmin, ymin = bbx[0], bbx[1]
    w, h = bbx[2], bbx[3]
    if sample_method == 'random':
        x = np.random.randint(xmin, xmin+w, nkps)
        y = np.random.randint(ymin, ymin+h, nkps)
    elif sample_method == 'regular_grid':
        # from skimage.util import regular_grid
        yx_slices = regular_grid((h, w), nkps)
        y = np.array(range(h)[yx_slices[0]])
        x = np.array(range(w)[yx_slices[1]])
        y = y + ymin
        x = x + xmin

    x, y = x.reshape(nkps, 1), y.reshape(nkps, 1)
    grids = np.concatenate((x, y), axis=1)

    return grids

def sample_points_within_rect_but_distant_to_anchors(nkps=20, bbx=(0, 0, 384, 384), sample_method='random', anchors=None, anchor_mask=None, dist_thresh=10):
    '''
    idea: firstly sample points within rect, secondly filter and maintain valid points
    bbx (xmin, ymin, w, h)
    anchors: M x 2 anchor points, np.array, each row is a point (x, y)
    anchor_mask: M
    return: success flag & sampled keypoints
    '''
    N = 100 if nkps <= 20 else 5*nkps  # our strategy is to sample N pts first and then fileter nkps valid.
    # long_edge = bbx[2] if bbx[2] >= bbx[3] else bbx[3]
    # T = float(long_edge) * dist_thresh
    T = dist_thresh ** 2
    sample_success = False
    sampled_kps = np.zeros((nkps, 2))
    invalid_anchor_mask = 1 - anchor_mask
    for i in range(30):  # try 30 times
        pts = sample_keypoints_within_rect(N, 0, 0, bbx, sample_method)
        dist_square = ((pts.reshape(-1, 1, 2) - anchors.reshape(1, -1, 2)) ** 2).sum(axis=2)  # N x M
        indicator = (dist_square >= T)  # N x M
        M = len(anchor_mask)
        indicator = (indicator + invalid_anchor_mask.reshape(1, -1)) > 0  # N x M
        indicator = indicator.sum(axis=1) == M  # N

        if sum(indicator) < nkps:
            continue

        pts_filtered = pts[indicator]
        sampled_kps = pts_filtered[0:nkps]
        sample_success = True
        break

    return sample_success, sampled_kps

def sample_points_via_interpolate_but_distant_to_anchors(nkps=20, anchors=None, anchor_mask=None, dist_thresh=10):
    '''
    idea: Firstly extract valid anchor points given anchor mask. Then construct body part paths given valid points.
    Finally sample body part path and interpolate keypoint.
    anchors: M x 2 anchor points, np.array, each row is a point (x, y)
    anchor_mask: M
    return: success flag & sampled keypoints
    '''
    T = dist_thresh ** 2
    sample_success = False
    sampled_kps = np.zeros((nkps, 2))
    N_valid = int(sum(anchor_mask))
    if N_valid <= 1:
        return sample_success, sampled_kps
    # copy valid kps
    valid_kps = np.zeros((N_valid, 2))
    count = 0
    for i in range(len(anchor_mask)):
        if anchor_mask[i] == 1:
            valid_kps[count] = anchors[i]
            count += 1
    # construct paths
    body_part_paths = []
    for i in range(0, N_valid - 1, 1):
        for j in range(i + 1, N_valid, 1):
            body_part_paths.append([i, j])
    # sample path and interpolate
    sample_success = True
    for i in range(nkps):
        sample_each_pt_success = False
        for j in range(30):  # try 30 times
            path_ind = np.random.randint(0, len(body_part_paths))
            ind1, ind2 = body_part_paths[path_ind]
            pt1, pt2 = valid_kps[ind1], valid_kps[ind2]
            t = 0.5
            pt = t * pt1 + (1-t) * pt2
            dist1 = sum((pt - pt1) ** 2)
            dist2 = sum((pt - pt2) ** 2)
            if dist1 >= T and dist2 >= T:
                sampled_kps[i] = pt
                sample_each_pt_success = True
                break
            else:
                continue
        if sample_each_pt_success == False:
            sample_success = False
            break

    return sample_success, sampled_kps

def get_random_occlusion_map(num_patch_l=1, num_patch_h=5, im_prob=0.5, patch_grids=(12, 12), saliency_map=None, fg_thresh=0.4, p_thresh=0.1):
    '''
    Generate occlusion map by randomly occluding a number of foreground patches. The foreground patch is determined by
    the saliency and patch threshold.

    num_patch_l, num_patch_h: low ~ high number of patches to be occluded
    saliency_map: torch tensor, B x C x H x W, e.g., 5 x 1 x 384 x 384
    im_prob: probability to apply occlusion on an image
    fg_thresh: saliency larger than this value is regarded as a foreground pixel
    p_thresh: (the number of FG pixel)/(number of pixels in a patch) larger than this ratio is regarded a FG patch
    '''

    # im_H, im_W = saliency_map.shape[2:]

    patched_ims = rearrange(saliency_map, 'B C (h p1) (w p2) -> B C (h w) p1 p2', h=patch_grids[0], w=patch_grids[1])
    B2, _, num_total_grids, p1, p2 = patched_ims.shape
    patched_ims = patched_ims.reshape(B2, num_total_grids, p1, p2)  # B x (grid_h*grid_w) x p1 x p2
    fg_pixel_indicator = (patched_ims.flatten(2) >= fg_thresh)  # B x (grid_h*grid_w) x (p1 x p2)
    valid_patches = fg_pixel_indicator.sum(-1) >= (p_thresh * p1 * p2)  # B x (grid_h*grid_w)
    num_valid_patches = valid_patches.sum(-1)  # B

    # it is "white" (value=1) by default if no occlusion
    occlusion_maps = torch.ones((B2, patch_grids[0]*patch_grids[1]))  # B x (grid_h x grid_w)
    inds = np.arange(0, num_total_grids, step=1)
    for im_i in range(B2):
        if num_patch_l >= num_patch_h:
            num_patch = num_patch_h
        else:
            num_patch = random.randint(num_patch_l, num_patch_h)
        image_prob = random.uniform(0, 1)
        if (image_prob > im_prob) or (num_patch <= 0) or (num_patch > num_valid_patches[im_i]):
            pass
        else:
            inds_filtered = inds[valid_patches[im_i]]
            random.shuffle(inds_filtered)
            inds_sampled = inds_filtered[0:num_patch]
            occlusion_maps[im_i, inds_sampled] = 0

    return occlusion_maps  # B x (grid_h x grid_w)

def get_random_occlusion_map_by_ratio(num_patch_ratio=0.1, im_prob=0.5, patch_grids=(12, 12), saliency_map=None, fg_thresh=0.4, p_thresh=0.1):
    '''
    Generate occlusion map by a ratio of foreground patches. The foreground patch is determined by
    the saliency and patch threshold.

    num_patch_ratio: a given ratio number of foreground patches to be occluded
    saliency_map: torch tensor, B x C x H x W, e.g., 5 x 1 x 384 x 384
    im_prob: probability to apply occlusion on an image
    fg_thresh: saliency larger than this value is regarded as a foreground pixel
    p_thresh: (the number of FG pixel)/(number of pixels in a patch) larger than this ratio is regarded a FG patch
    '''

    # im_H, im_W = saliency_map.shape[2:]

    patched_ims = rearrange(saliency_map, 'B C (h p1) (w p2) -> B C (h w) p1 p2', h=patch_grids[0], w=patch_grids[1])
    B2, _, num_total_grids, p1, p2 = patched_ims.shape
    patched_ims = patched_ims.reshape(B2, num_total_grids, p1, p2)  # B x (grid_h*grid_w) x p1 x p2
    fg_pixel_indicator = (patched_ims.flatten(2) >= fg_thresh)  # B x (grid_h*grid_w) x (p1 x p2)
    valid_patches = fg_pixel_indicator.sum(-1) >= (p_thresh * p1 * p2)  # B x (grid_h*grid_w)
    num_valid_patches = valid_patches.sum(-1)  # B

    # it is "white" (value=1) by default if no occlusion
    occlusion_maps = torch.ones((B2, patch_grids[0]*patch_grids[1]))  # B x (grid_h x grid_w)
    inds = np.arange(0, num_total_grids, step=1)
    for im_i in range(B2):
        num_patch = int(num_patch_ratio * num_valid_patches[im_i])
        if num_patch > num_valid_patches[im_i]:
            num_patch = num_valid_patches[im_i]
        image_prob = random.uniform(0, 1)
        if (image_prob > im_prob) or (num_patch <= 0) or (num_patch > num_valid_patches[im_i]):
            pass
        else:
            inds_filtered = inds[valid_patches[im_i]]
            random.shuffle(inds_filtered)
            inds_sampled = inds_filtered[0:num_patch]
            occlusion_maps[im_i, inds_sampled] = 0

    return occlusion_maps  # B x (grid_h x grid_w)

#==============================================================
# Below code is useless
def train_parser():
    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt", help="optimizer", choices=['adam', 'sgd'])
    parser.add_argument("--lr", help="initial learning rate", type=float)
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--epoch", help="number of epochs before lr is cut by gamma", type=int)
    parser.add_argument("--stage", help="number lr stages", type=int)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float)
    parser.add_argument("--gpu", help="gpu device", type=int, default=0)
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=20)
    parser.add_argument("--resnet", help="whether use resnet18 as backbone or not", action="store_true")

    ## PN model related hyper-parameters
    parser.add_argument("--alpha", help="scalar for pose loss", type=int)
    parser.add_argument("--num_part", help="number of parts", type=int)
    parser.add_argument("--percent", help="percent of base images with part annotation", type=float)

    ## shared optional
    parser.add_argument("--batch_size", help="batch size", type=int)
    parser.add_argument("--load_path", help="load path for dynamic/transfer models", type=str)

    args = parser.parse_args()

    if args.resnet:
        name = 'ResNet18'
    else:
        name = 'Conv4'

    return args, name


if __name__ == '__main1__':
    # below code is for developing distance transform

    p = '/home/changsheng/KeypointDetectionWithFSL/FSKD-ViT/episode_images/sal_examples/2007_000876.png'
    im = PIL.Image.open(p)
    im = im.resize((384, 384))
    im = np.array(im)
    im_min, im_max = np.min(im), np.max(im)
    H, W = im.shape
    max_dist = np.sqrt(H ** 2 + W ** 2)



    thresh = int(0.4*255)
    ret, imbinary = cv2.threshold(im, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY)
    plt.imshow(im, cmap='gray')
    plt.show()
    plt.imshow(imbinary, cmap='gray', vmin=0, vmax=255)
    plt.show()

    t1 = time.time()
    for i in range(1):
        fg_edt = cv2.distanceTransform(imbinary,distanceType=cv2.DIST_L2,maskSize=cv2.DIST_MASK_PRECISE)
        bg_edt = cv2.distanceTransform(255-imbinary,distanceType=cv2.DIST_L2,maskSize=cv2.DIST_MASK_PRECISE)
    t2 = time.time()
    print('time: ', (t2-t1))

    plt.imshow(fg_edt)
    plt.show()
    plt.imshow(bg_edt)
    plt.show()
    whole_edt = (fg_edt - bg_edt)
    plt.imshow(whole_edt)
    plt.show()
    # contrast-preserving normalization
    edt_min, edt_max = np.min(whole_edt), np.max(whole_edt)
    whole_edt2 = (whole_edt - edt_min)/(edt_max-edt_min+1e-6) * (im_max-im_min) + im_min
    whole_edt2 /= 255
    plt.imshow(whole_edt2, cmap='gray')
    plt.show()


    coords_h = torch.arange(H).float()
    coords_w = torch.arange(W).float()
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2 x H x W
    sal_normalize = im / 255.0  # H x W
    sal_normalize = torch.tensor(sal_normalize)
    sal_sum = torch.sum(sal_normalize)
    centroid = (coords * sal_normalize.reshape(1, H, W)).view(2, -1).sum(dim=1) / (sal_sum + 1e-6)  # 2
    centroid = centroid.view(2, 1, 1)
    dist_field = ((coords - centroid) ** 2).sum(dim=0).sqrt()  # H x W
    # dist_field = torch.clamp(dist_field, min=1)
    dist_field /= max_dist
    rev_dist_field = 1 - dist_field
    # contrast-preserving normalization
    rev_dist_min, rev_dist_max = torch.min(rev_dist_field), torch.max(rev_dist_field)
    rev_dist_field = (rev_dist_field - rev_dist_min) / (rev_dist_max - rev_dist_min + 1e-6) * (im_max - im_min) + im_min
    rev_dist_field /= 255
    plt.imshow(rev_dist_field.numpy(), cmap='gray'); plt.show()
    # torch.save(rev_dist_field, 'a.pt')

    fg_edt_max_value = np.max(fg_edt)
    fg_edt_normalize = fg_edt / (fg_edt_max_value+1e-6)
    fg_edt_normalize = torch.tensor(fg_edt_normalize)
    comb1 = (rev_dist_field + fg_edt_normalize) / 2
    plt.imshow(comb1.numpy(), cmap='gray'); plt.show()
    # torch.save(comb1, 'b.pt')

    comb2 = (rev_dist_field + fg_edt_normalize + sal_normalize) / 3
    plt.imshow(comb2.numpy(), cmap='gray'); plt.show()

    whole_edt2 = torch.tensor(whole_edt2)
    comb3 = (whole_edt2 + fg_edt_normalize) / 2
    plt.imshow(comb3.numpy(), cmap='gray'); plt.show()

    comb4 = (whole_edt2 + fg_edt_normalize + sal_normalize) / 3
    plt.imshow(comb4.numpy(), cmap='gray'); plt.show()
    print('ok')


if __name__=='__main__':
    # below code is for testing

    # p = '/home/changsheng/KeypointDetectionWithFSL/FSKD-ViT/episode_images/sal_examples/2007_000876.png'
    p = '/home/changsheng/KeypointDetectionWithFSL/FSKD-ViT/episode_images/sal_examples/Ring_Billed_Gull_0092_51521.png'
    # p = '/home/changsheng/KeypointDetectionWithFSL/FSKD-ViT/episode_images/sal_examples/Tropical_Kingbird_0038_69455.png'
    # p = '/home/changsheng/KeypointDetectionWithFSL/FSKD-ViT/episode_images/sal_examples/Tropical_Kingbird_0063_69589.png'
    # p = '/media/changsheng/1651-846C/FSKD/FSKD-ViT-Relatings/saliency_map_samples/SCRN/ca195.png'
    p = '/home/changsheng/KeypointDetectionWithFSL/FSKD-ViT/episode_images/sal_examples2/15_q2_1.jpg'
    p_save = '/home/changsheng/KeypointDetectionWithFSL/FSKD-ViT/episode_images/sal_examples2/15_q2_1_ip5-0.4-5'

    im = PIL.Image.open(p)
    w, h = im.size
    w2, h2 = w, h  # 384, 384
    # im = im.resize((384, 384))
    im = np.array(im) / 255.0
    plt.imshow(im, cmap='gray', vmin=0, vmax=1.0); plt.show()

    dt_module = DTModule(dt_type=5, fg_thresh=0.4, im_size=(h2, w2))
    blur_module = GaussianBlur(sigma=5, use_cuda=True)

    # distance transform
    im_input = torch.tensor(im.reshape(1, 1, h2, w2))
    out = dt_module.dt(im_input)
    plt.imshow(out[0,0].numpy(), cmap='gray', vmin=0, vmax=1.0); plt.show()

    # gaussian blur
    out2 = blur_module.gaussian_blur(out.cuda())
    out2 = out2[0, 0]
    plt.imshow(out2.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1.0)
    # plt.savefig('/media/changsheng/1651-846C/FSKD/FSKD-ViT-Relatings/saliency_map_samples/SCRN/ca57_ip5_0.4_5.png')
    out_saved = (out2.cpu().detach().numpy() * 255).astype(np.uint8)
    out_saved_color = cv2.applyColorMap(np.uint8(out_saved), cv2.COLORMAP_JET)
    cv2.imwrite(p_save + '.png', out_saved)
    cv2.imwrite(p_save + 'color.png', out_saved_color)

    outpn = torch.pow(out2, 0.2)
    out_saved = (outpn.cpu().detach().numpy() * 255).astype(np.uint8)
    out_saved_color = cv2.applyColorMap(np.uint8(out_saved), cv2.COLORMAP_JET)
    cv2.imwrite(p_save + '_pn0.2.png', out_saved)
    cv2.imwrite(p_save + '_pn0.2color.png', out_saved_color)

    outpn = torch.pow(out2, 0.5)
    out_saved = (outpn.cpu().detach().numpy() * 255).astype(np.uint8)
    out_saved_color = cv2.applyColorMap(np.uint8(out_saved), cv2.COLORMAP_JET)
    cv2.imwrite(p_save + '_pn0.5.png', out_saved)
    cv2.imwrite(p_save + '_pn0.5color.png', out_saved_color)

    outpn = torch.pow(out2, 0.6)
    out_saved = (outpn.cpu().detach().numpy() * 255).astype(np.uint8)
    out_saved_color = cv2.applyColorMap(np.uint8(out_saved), cv2.COLORMAP_JET)
    cv2.imwrite(p_save + '_pn0.6.png', out_saved)
    cv2.imwrite(p_save + '_pn0.6color.png', out_saved_color)

    outpn = torch.pow(out2, 0.7)
    out_saved = (outpn.cpu().detach().numpy() * 255).astype(np.uint8)
    out_saved_color = cv2.applyColorMap(np.uint8(out_saved), cv2.COLORMAP_JET)
    cv2.imwrite(p_save + '_pn0.7.png', out_saved)
    cv2.imwrite(p_save + '_pn0.7color.png', out_saved_color)

    outpn = torch.pow(out2, 2.0)
    out_saved = (outpn.cpu().detach().numpy() * 255).astype(np.uint8)
    out_saved_color = cv2.applyColorMap(np.uint8(out_saved), cv2.COLORMAP_JET)
    cv2.imwrite(p_save + '_pn2.0.png', out_saved)
    cv2.imwrite(p_save+'_pn2.0color.png', out_saved_color)

    outpn = torch.pow(out2, 5.0)
    out_saved = (outpn.cpu().detach().numpy() * 255).astype(np.uint8)
    out_saved_color = cv2.applyColorMap(np.uint8(out_saved), cv2.COLORMAP_JET)
    cv2.imwrite(p_save + '_pn5.0.png', out_saved)
    cv2.imwrite(p_save + '_pn5.0color.png', out_saved_color)


    plt.show()
    print('finished')


if __name__=='__main3__':
    a = torch.load('./a.pt')
    b = torch.load('./b.pt')
    a2 = torch.load('./a2.pt').squeeze()
    b2 = torch.load('./b2.pt').squeeze()
    print(torch.sum((a-a2)**2))
    print(torch.sum((b - b2) ** 2))
    print('ok')