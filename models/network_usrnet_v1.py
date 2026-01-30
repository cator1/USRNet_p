import torch
import torch.nn as nn
import models.basicblock as B
import numpy as np
from utils import utils_image as util
import torch.fft
import math


# for pytorch version >= 1.8.1


"""
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
"""


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""


def splits(a, sf):
    '''split a into sfxsf distinct blocks

    Args:
        a: NxCxWxH
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    '''
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b


def p2o(psf, shape):
    '''
    Convert point-spread function to optical transfer function.
    otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
    point-spread function (PSF) array and creates the optical transfer
    function (OTF) array that is not influenced by the PSF off-centering.

    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    '''
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[...,:psf.shape[2],:psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis+2)
    otf = torch.fft.fftn(otf, dim=(-2,-1))
    #n_ops = torch.sum(torch.tensor(psf.shape).type_as(psf) * torch.log2(torch.tensor(psf.shape).type_as(psf)))
    #otf[..., 1][torch.abs(otf[..., 1]) < n_ops*2.22e-16] = torch.tensor(0).type_as(psf)
    return otf


def upsample(x, sf=3):
    '''s-fold upsampler

    Upsampling the spatial size by filling the new entries with zeros

    x: tensor image, NxCxWxH
    '''
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z


def downsample(x, sf=3):
    '''s-fold downsampler

    Keeping the upper-left pixel for each distinct sfxsf patch and discarding the others

    x: tensor image, NxCxWxH
    '''
    st = 0
    return x[..., st::sf, st::sf]


def downsample_np(x, sf=3):
    st = 0
    return x[st::sf, st::sf, ...]


"""
# --------------------------------------------
# (1) Prior module; ResUNet: act as a non-blind denoiser
# x_k = P(z_k, beta_k)
# --------------------------------------------
"""


class ResUNet(nn.Module):
    def __init__(self, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(ResUNet, self).__init__()

        self.m_head = B.conv(in_nc, nc[0], bias=False, mode='C')

        # downsample
        if downsample_mode == 'avgpool':
            downsample_block = B.downsample_avgpool
        elif downsample_mode == 'maxpool':
            downsample_block = B.downsample_maxpool
        elif downsample_mode == 'strideconv':
            downsample_block = B.downsample_strideconv
        else:
            raise NotImplementedError('downsample mode [{:s}] is not found'.format(downsample_mode))

        self.m_down1 = B.sequential(*[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[0], nc[1], bias=False, mode='2'))
        self.m_down2 = B.sequential(*[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[1], nc[2], bias=False, mode='2'))
        self.m_down3 = B.sequential(*[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)], downsample_block(nc[2], nc[3], bias=False, mode='2'))

        self.m_body  = B.sequential(*[B.ResBlock(nc[3], nc[3], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        # upsample
        if upsample_mode == 'upconv':
            upsample_block = B.upsample_upconv
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.upsample_pixelshuffle
        elif upsample_mode == 'convtranspose':
            upsample_block = B.upsample_convtranspose
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))

        self.m_up3 = B.sequential(upsample_block(nc[3], nc[2], bias=False, mode='2'), *[B.ResBlock(nc[2], nc[2], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up2 = B.sequential(upsample_block(nc[2], nc[1], bias=False, mode='2'), *[B.ResBlock(nc[1], nc[1], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])
        self.m_up1 = B.sequential(upsample_block(nc[1], nc[0], bias=False, mode='2'), *[B.ResBlock(nc[0], nc[0], bias=False, mode='C'+act_mode+'C') for _ in range(nb)])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode='C')

    def forward(self, x):
        
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x+x4)
        x = self.m_up2(x+x3)
        x = self.m_up1(x+x2)
        x = self.m_tail(x+x1)

        x = x[..., :h, :w]

        return x


"""
# --------------------------------------------
# (2) Data module, closed-form solution
# It is a trainable-parameter-free module  ^_^
# z_k = D(x_{k-1}, s, k, y, alpha_k)
# some can be pre-calculated
# --------------------------------------------
"""


class DataNet(nn.Module):
    def __init__(self):
        super(DataNet, self).__init__()

    def forward(self, x, FB, FBC, F2B, FBFy, alpha, sf):

        FR = FBFy + torch.fft.fftn(alpha*x, dim=(-2,-1))
        x1 = FB.mul(FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + alpha)
        FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
        FX = (FR-FCBinvWBR)/alpha
        Xest = torch.real(torch.fft.ifftn(FX, dim=(-2,-1)))

        return Xest


"""
# --------------------------------------------
# (3) Hyper-parameter module
# --------------------------------------------
"""


class HyPaNet(nn.Module):
    def __init__(self, in_nc_global=2, in_nc_local=2, out_nc=2, channel=64):
        super(HyPaNet, self).__init__()

        # --- Net A: 全局上下文网络 (Controller) ---
        # 输入: sigma, sf
        # 输出: FiLM 调制参数 gamma (scale) 和 delta (shift)
        # 结构: Conv 1x1 -> ReLU -> Conv 1x1 -> ReLU -> Conv 1x1
        self.global_net = nn.Sequential(
            nn.Conv2d(in_nc_global, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel * 2, 1, padding=0, bias=True) # 输出 2*channel
        )

        # --- Net B: 时序动态网络 (Backbone) ---
        # 第一部分: 特征提取
        self.step_net_1 = nn.Sequential(
            nn.Conv2d(in_nc_local, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True)
        )
        # 第二部分: 输出映射 (在 FiLM 调制之后)
        self.step_net_2 = nn.Sequential(
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus() # 确保 alpha, beta 为正数
        )

    def forward(self, sigma, sf, t, n_inv):
        # 计算全局参数
        global_input = torch.cat((sigma, sf), dim=1)  # (N, 2, 1, 1)
        film_params = self.global_net(global_input)  # (N, 2*channel, 1, 1)
        gamma, delta = torch.chunk(film_params, 2, dim=1)

        # 计算局部参数
        local_input = torch.cat((t, n_inv), dim=1)  # (N, 2, 1, 1)
        features = self.step_net_1(local_input)  # (N, channel, 1, 1)

        # FiLM 调制
        features = features * (1 + gamma) + delta  # (N, channel, 1, 1)

        # 输出映射
        out = self.step_net_2(features) + 1e-6 # (N, out_nc, 1, 1)

        return out

"""
# --------------------------------------------
# main USRNet
# deep unfolding super-resolution network
# --------------------------------------------
"""


class USRNet(nn.Module):
    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=2, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose'):
        super(USRNet, self).__init__()

        self.d = DataNet()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb, act_mode=act_mode, downsample_mode=downsample_mode, upsample_mode=upsample_mode)
   
        self.h = HyPaNet(in_nc_global=2, in_nc_local=2, out_nc=2, channel=h_nc)
        
        self.n_iter = n_iter

    def forward(self, x, k, sf, sigma, n_iter=None):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''

        # initialization & pre-calculation
        w, h = x.shape[-2:]
        FB = p2o(k, (w*sf, h*sf))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        STy = upsample(x, sf=sf)
        FBFy = FBC*torch.fft.fftn(STy, dim=(-2,-1))
        x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')

        # 确定本次前向传播的迭代次数
        # 训练时通常使用 self.n_iter (可能被外部随机化)
        # 测试时可以通过 n_iter 参数手动指定
        current_iter = n_iter if n_iter is not None else self.n_iter

        #数据预处理
        sf_tensor = torch.tensor(sf).type_as(sigma).view(1, 1, 1, 1).expand_as(sigma)
        n_inv_val = np.float32(1.0) / np.float32(current_iter)
        n_inv_tensor = torch.tensor(n_inv_val).type_as(sigma).expand_as(sigma)
        

        # unfolding
        for i in range(current_iter):

            t_val = np.float32(i/current_iter)
            t_tensor = torch.tensor(t_val).type_as(sigma).expand_as(sigma)          
            # 调用超参数预测
            ab = self.h(sigma, sf_tensor, t_tensor, n_inv_tensor)
            alpha = ab[:, 0:1, :, :]
            beta  = ab[:, 1:2, :, :]

            # 1. 求解数据子问题 (D Module)
            x = self.d(x, FB, FBC, F2B, FBFy, alpha, sf)
            
            # 2. 求解先验子问题 (P Module)
            # 将 beta 作为噪声水平图拼接到输入中
            beta_map = beta.repeat(1, 1, x.size(2), x.size(3))
            x = self.p(torch.cat((x, beta_map), dim=1))
            
        return x