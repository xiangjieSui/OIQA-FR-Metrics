import os
from Grid import genSamplingPattern
from torch import nn
import math
from torchvision import transforms
import numpy as np
import torch
from torch.nn import functional as F
pi = math.pi


'''
For [WS_PSNR', 'CPP_PSNR', 'S_PSNR', 'S_SSIM'], we pre-generate the weight/coordinate to speed up. 
The generated stuff will be stored in './weight_map'. 
'''


class FR_Metric(nn.Module):

    def __init__(self, print_models=True):
        super().__init__()
        self.model_list = ['MSE', 'PSNR',
                           'SSIM', 'WS_PSNR',
                           'CPP_PSNR', 'S_PSNR',
                           'S_SSIM']
        if print_models:
            print('\nSupport models:')
            print(self.model_list, '\n')

    def forward(self, model_list, img_pairs):
        assert all(item in self.model_list for item in model_list)

        score_dic = {}

        for model_i in model_list:
            function = getattr(self, model_i)
            score_dic[model_i] = function(img_pairs[0], img_pairs[1]).cpu()

        return score_dic

    ''' [Mean Squared Error]  '''

    def MSE(self, X, Y, weight=None):
        # X/Y.shape (B, C, H, W)
        assert(X.shape == Y.shape), 'X.shape and Y.shape are unmatch'

        diff = torch.abs(X - Y)
        diff_2 = diff * diff

        if weight is not None:
            # (B, H, W)
            diff_2 = torch.mean(diff_2, dim=1)
            weight = weight.repeat(diff_2.shape[0], 1, 1)
            assert weight.shape == diff_2.shape
            mse = torch.sum(diff_2 * weight, dim=[1, 2]) / torch.sum(weight)
        else:
            mse = torch.mean(diff_2, dim=[1, 2, 3])

        return mse

    ''' [Peak Signal-to-Noise Ratio] '''

    def PSNR(self, X, Y):
        # X/Y.shape (B, C, H, W)
        mse = self.MSE(X, Y)

        score = 10 * torch.log10(1 / mse)
        score[mse == 0] = 100

        return score

    '''
    [Structural Similarity]
    Z. Wang, A.C. Bovik, H.R. Sheikh, E.P. Simoncelli. Image quality assessment: 
    from error visibility to structural similarity. IEEE Transactions on Image Processing. 
    vol. 13, no. 4, pp.600–612
    Implemented by Keyan Ding
    part of this code is source from: https://github.com/dingkeyan93/IQA-optimization/blob/master/IQA_pytorch/SSIM.py
    '''

    def auto_downsampling(self, X):
        B, C, H, W = X.shape
        stride = H // 1024

        if stride > 0:
            if stride == 1:
                stride = 2
            target_H = H // stride
            target_W = W // stride
            X = F.interpolate(X, size=(target_H, target_W),
                              mode='bilinear', align_corners=False)
        return X

    def fspecial_gauss(self, size, sigma, channels):
        # Function to mimic the 'fspecial' gaussian MATLAB function
        x, y = np.mgrid[-size // 2 + 1:size //
                        2 + 1, -size // 2 + 1:size // 2 + 1]
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
        g = torch.from_numpy(g / g.sum()).float().unsqueeze(0).unsqueeze(0)
        return g.repeat(channels, 1, 1, 1)

    def gaussian_filter(self, input, win):
        out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
        return out

    def ssim_conv(self, win, X, Y, C1, C2, weight=None):
        mu1 = self.gaussian_filter(X, win)
        mu2 = self.gaussian_filter(Y, win)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = self.gaussian_filter(X * X, win) - mu1_sq
        sigma2_sq = self.gaussian_filter(Y * Y, win) - mu2_sq
        sigma12 = self.gaussian_filter(X * Y, win) - mu1_mu2

        cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
        # force the ssim response to be nonnegative to avoid negative results.
        cs_map = F.relu(cs_map)
        ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

        if weight is not None:
            weight = (torch.ones(1) * weight).to(device)
            ssim_map = ssim_map * weight

        ssim_val = torch.mean(ssim_map, dim=[1, 2, 3])

        return ssim_map, ssim_val

    def ssim_index(self, X, Y, win, get_ssim_map=False):

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        win = win.to(X.device)

        ssim_map, ssim_val = self.ssim_conv(win, X, Y, C1, C2)

        if get_ssim_map:
            return ssim_map

        return ssim_val

    def SSIM(self, X, Y, auto_ds=True):

        # X/Y.shape (B, C, H, W)
        assert(X.shape == Y.shape), 'X.shape and Y.shape are unmatch'

        if auto_ds:
            X = self.auto_downsampling(X)
            Y = self.auto_downsampling(Y)

        with torch.no_grad():
            win = self.fspecial_gauss(11, 1.5, X.shape[1])
            score = self.ssim_index(X, Y, win=win)

        return score

    '''OIQA metrics are pytorch implements of  https://github.com/Samsung/360tools'''

    '''
    [Weighted Spherical PSNR]
    Y. Sun, A. Lu, and L. Yu. Weighted-to-spherically-uniform quality evaluation 
    for omnidirectional video. IEEE Signal Processing Letters, vol. 24, no. 9, 
    pp. 1408-1412, 2017.
    '''

    def WS_PSNR_WM(self, h, w):
        weight_map = torch.zeros([h, w])
        # for ERP format
        for i in range(h):
            for j in range(w):
                weight_map[i, j] = math.cos((i - h / 2 + 0.5) * pi / h)
        return weight_map

    def WS_PSNR(self, X, Y):
        # X/Y.shape (B, C, H, W)
        assert(X.shape == Y.shape), 'X.shape and Y.shape are unmatch'
        device = X.device

        weight_map_name = '({}, {})_WS_PSNR.pth'.format(X.shape[2], X.shape[3])
        weight_map_path = os.path.join('./weight_map', weight_map_name)

        if os.path.exists(weight_map_path):
            weight_map = torch.load(weight_map_path).to(device)

        else:
            print('Generating WS_PSNR weight map (DO NOT use multi-GPUs when generation)')
            weight_map = self.WS_PSNR_WM(X.shape[2], X.shape[3])

            if not os.path.exists('./weight_map'):
                os.mkdir('./weight_map')
            torch.save(weight_map, weight_map_path)

            print('saving to {}'.format(weight_map_path))

            weight_map = weight_map.to(device)

        mse = self.MSE(X, Y, weight=weight_map)
        score = 10 * torch.log10(1 / mse)
        score[mse == 0] = 100

        return score

    '''
    [Craster Parabolic Projection PSNR]
    V. Zakharchenko, K.P. Choi, J.H. Park, Video quality metric for spherical
    panoramic video,  Optics and Photonics, SPIE, 9970, 2016.
    '''

    def CPP_PSNR_WM(self, h, w):
        weight_map = torch.zeros([h, w])
        for i in range(h):
            for j in range(w):
                x = j / w * 2 * pi - pi
                y = i / h * pi - pi / 2

                phi = 3 * math.asin(y / pi)
                theta = x / (2 * math.cos(2 * phi / 3) - 1)

                x = (theta + pi) / 2 / pi * w
                y = (phi + (pi / 2)) / pi * h

                idx_x = (x + 0.5, x - 0.5)[x < 0]
                idx_y = (y + 0.5, y - 0.5)[y < 0]

                if idx_y >= 0 and idx_x >= 0 and idx_x < w and idx_y < h:
                    weight_map[i, j] = 1

        return weight_map

    def CPP_PSNR(self, X, Y):

        # X/Y.shape (B, C, H, W)
        assert(X.shape == Y.shape), 'X.shape and Y.shape are unmatch'
        device = X.device

        weight_map_name = '({}, {})_CPP_PSNR.pth'.format(
            X.shape[2], X.shape[3])
        weight_map_path = os.path.join('./weight_map', weight_map_name)

        if os.path.exists(weight_map_path):
            weight_map = torch.load(weight_map_path).to(device)

        else:
            print('Generating CPP_PSNR weight map')
            weight_map = self.CPP_PSNR_WM(X.shape[2], X.shape[3])

            if not os.path.exists('./weight_map'):
                os.mkdir('./weight_map')
            torch.save(weight_map, weight_map_path)

            print('saving to {}'.format(weight_map_path))

            weight_map = weight_map.to(device)

        mse = self.MSE(X, Y, weight=weight_map)
        score = 10 * torch.log10(1 / mse)
        score[mse == 0] = 100

        return score

    '''
    [Spherical PSNR]
    M. Yu, H. Lakshman, and B. Girod. A frame-work to evaluate
    omnidirectional video coding schemes. in IEEE International Symposium on
    Mixed and Augmented Reality, 2015, pp. 31–36.
    '''

    def S_PSNR_WM(self, sphere_cord, height_width):
        plane_coords = sphere2plane(sphere_cord, height_width)
        weight_map = torch.zeros(height_width)
        for i in range(plane_coords.shape[0]):
            y, x = plane_coords[i, 0], plane_coords[i, 1]
            weight_map[int(y), int(x)] = 1
        return weight_map

    def S_PSNR(self, X, Y):
        # X/Y.shape (B, C, H, W)
        assert(X.shape == Y.shape), 'X.shape and Y.shape are unmatch'
        device = X.device

        weight_map_name = '({}, {})_S_PSNR.pth'.format(
            X.shape[2], X.shape[3])
        weight_map_path = os.path.join('./weight_map', weight_map_name)

        if os.path.exists(weight_map_path):
            weight_map = torch.load(weight_map_path).to(device)

        else:
            print('Generating coordinate for S_PSNR')
            coords = torch.from_numpy(np.loadtxt('sphere_655362.txt'))
            weight_map = self.S_PSNR_WM(coords, X.shape[2:])

            if not os.path.exists('./weight_map'):
                os.mkdir('./weight_map')
            torch.save(weight_map, weight_map_path)

            print('saving to {}'.format(weight_map_path))

            weight_map = weight_map.to(device)

        mse = self.MSE(X, Y, weight=weight_map)
        score = 10 * torch.log10(1 / mse)
        score[mse == 0] = 100

        return score

    '''
    [Spherical SSIM]
    S. Chen, Y. Zhang, Y. Li, Z. Chen and Z. Wang.
    Spherical structural similarity index for objective omnidirectional video quality assessment.
    IEEE International Conference on Multimedia and Expo. 2018, pp. 1-6.
    '''

    def S_SSIM(self, X, Y, auto_ds=True, stride=10):
        '''
        stride:int, stride of the grid sampling. Noting that larger value of stride 
        might reduce the accuracy of the model, but would be faster.
        '''

        # X/Y.shape (B, C, H, W)
        assert(X.shape == Y.shape), 'X.shape and Y.shape are unmatch'
        device = X.device

        if auto_ds:
            X = self.auto_downsampling(X)
            Y = self.auto_downsampling(Y)

        B, C, H, W = X.shape

        coord_name = '({}, {})_Stride-{}_S_SSIM.pth'.format(H, W, stride)
        coord_name_path = os.path.join('./weight_map', coord_name)

        if os.path.exists(coord_name_path):
            coords = torch.load(coord_name_path)

        else:
            print('Generating patch coordinate for S_SSIM')
            coords = genSamplingPattern(H, W, 11, 11, stride).long()

            if not os.path.exists('./weight_map'):
                os.mkdir('./weight_map')
            torch.save(coords, coord_name_path)

            print('saving to {}'.format(coord_name_path))

            coords = coords.to(device)

        H, W = coords.shape[:2]

        win = self.fspecial_gauss(11, 1.5, C).to(device)
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        sum_weight = 0
        sum_score = 0

        for i in range(H):
            for j in range(W):
                patch_coords = coords[i, j]

                X_patch = X[:, :, patch_coords[:, :, 0],
                            patch_coords[:, :, 1]]
                Y_patch = Y[:, :, patch_coords[:, :, 0],
                            patch_coords[:, :, 1]]

                weight = math.cos((i - H / 2 + 0.5) * pi / H)

                _, _score = self.ssim_conv(
                    win, X_patch, Y_patch, C1, C2, weight)

                sum_weight += weight
                sum_score += _score

        score = sum_score / sum_weight

        return score


if __name__ == '__main__':
    from PIL import Image

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.ToTensor()

    ref = torch.unsqueeze(transform(Image.open('ref.png').convert(
        "RGB")), dim=0).to(device)
    dis = torch.unsqueeze(transform(Image.open('dis.png').convert(
        "RGB")), dim=0).to(device)

    model_list = ['MSE', 'PSNR', 'SSIM',
                  'WS_PSNR', 'CPP_PSNR', 'S_PSNR', 'S_SSIM']

    model = FR_Metric()

    scores = model(model_list, (ref, dis))

    for key in scores:
        print(key, scores[key])
