"""
Masked Autoregressive Flow for Density Estimation
arXiv:1705.07057v4


选择流模型架构主要取决于你的具体应用场景和需求。以下是一些常见的流模型架构及其适用场景：

1. **MADE (Masked Autoencoder for Distribution Estimation)**:
   - **适用场景**: 适用于高维数据的分布估计。
   - **优点**: 结构简单，易于实现，计算效率高。
   - **缺点**: 可能无法很好地捕捉复杂的数据分布。

2. **MAF (Masked Autoregressive Flow)**:
   - **适用场景**: 适用于高维数据的分布估计，特别是需要捕捉复杂依赖关系的场景。
   - **优点**: 可以捕捉复杂的依赖关系，适用于高维数据。
   - **缺点**: 计算复杂度较高，训练时间较长。

3. **RealNVP (Real NVP)**:
   - **适用场景**: 适用于高维数据的分布估计，特别是需要保持输入输出的连续性的场景。
   - **优点**: 可以保持输入输出的连续性，适用于需要平滑采样的任务。
   - **缺点**: 计算复杂度较高，训练时间较长。

4. **MADE-MOG (Masked Autoencoder for Distribution Estimation with Mixture of Gaussians)**:
   - **适用场景**: 适用于高维数据的混合高斯分布估计。
   - **优点**: 可以估计混合高斯分布，适用于需要捕捉多模态数据的场景。
   - **缺点**: 计算复杂度较高，训练时间较长。

5. **MAF-MOG (Masked Autoregressive Flow with Mixture of Gaussians)**:
   - **适用场景**: 适用于高维数据的混合高斯分布估计，特别是需要捕捉复杂依赖关系的场景。
   - **优点**: 可以捕捉复杂的依赖关系，适用于高维数据。
   - **缺点**: 计算复杂度较高，训练时间较长。

在选择流模型架构时，需要考虑以下几个因素：

- **数据维度**: 高维数据通常需要选择能够处理高维数据的模型架构。
- **数据分布**: 如果数据分布复杂，需要选择能够捕捉复杂依赖关系的模型架构。
- **计算资源**: 计算复杂度较高的模型架构需要更多的计算资源，需要根据实际情况进行权衡。
- **应用场景**: 根据具体的应用场景，选择能够满足需求的模型架构。

在选择流模型架构时，还可以参考相关的文献和实验结果，以确定哪种模型架构最适合的任务。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torchvision.transforms as T
from torchvision.utils import save_image

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import math
import argparse
import pprint
import copy


# --------------------
# Model layers and helpers
# --------------------

def create_masks(input_size, hidden_size, n_hidden, input_order='sequential', input_degrees=None):
    # MADE paper sec 4:
    # degrees of connections between layers -- ensure at most in_degree - 1 connections
    degrees = []

    # set input degrees to what is provided in args (the flipped order of the previous layer in a stack of mades);
    # else init input degrees based on strategy in input_order (sequential or random)
    if input_order == 'sequential':
        degrees += [torch.arange(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            degrees += [torch.arange(hidden_size) % (input_size - 1)]
        degrees += [torch.arange(input_size) % input_size - 1] if input_degrees is None else [
            input_degrees % input_size - 1]

    elif input_order == 'random':
        degrees += [torch.randperm(input_size)] if input_degrees is None else [input_degrees]
        for _ in range(n_hidden + 1):
            min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
            degrees += [torch.randint(min_prev_degree, input_size, (hidden_size,))]
        min_prev_degree = min(degrees[-1].min().item(), input_size - 1)
        degrees += [torch.randint(min_prev_degree, input_size, (input_size,)) - 1] if input_degrees is None else [
            input_degrees - 1]

    # construct masks
    masks = []
    for (d0, d1) in zip(degrees[:-1], degrees[1:]):
        masks += [(d1.unsqueeze(-1) >= d0.unsqueeze(0)).float()]

    return masks, degrees[0]


class MaskedLinear(nn.Linear):
    """ MADE building block layer """

    def __init__(self, input_size, n_outputs, mask, cond_label_size=None):
        super().__init__(input_size, n_outputs)

        self.register_buffer('mask', mask)

        self.cond_label_size = cond_label_size
        if cond_label_size is not None:
            self.cond_weight = nn.Parameter(torch.rand(n_outputs, cond_label_size) / math.sqrt(cond_label_size))

    def forward(self, x, y=None):
        out = F.linear(x, self.weight * self.mask, self.bias)
        if y is not None:
            out = out + F.linear(y, self.cond_weight)
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + (self.cond_label_size != None) * ', cond_features={}'.format(self.cond_label_size)


class LinearMaskedCoupling(nn.Module):
    """ Modified RealNVP Coupling Layers per the MAF paper """

    def __init__(self, input_size, hidden_size, n_hidden, mask, cond_label_size=None):
        super().__init__()

        self.register_buffer('mask', mask)

        # scale function
        s_net = [nn.Linear(input_size + (cond_label_size if cond_label_size is not None else 0), hidden_size)]
        for _ in range(n_hidden):
            s_net += [nn.Tanh(), nn.Linear(hidden_size, hidden_size)]
        s_net += [nn.Tanh(), nn.Linear(hidden_size, input_size)]
        self.s_net = nn.Sequential(*s_net)

        # translation function
        self.t_net = copy.deepcopy(self.s_net)
        # replace Tanh with ReLU's per MAF paper
        for i in range(len(self.t_net)):
            if not isinstance(self.t_net[i], nn.Linear): self.t_net[i] = nn.ReLU()

    def forward(self, x, y=None):
        # apply mask
        mx = x * self.mask

        # run through model
        s = self.s_net(mx if y is None else torch.cat([y, mx], dim=1))
        t = self.t_net(mx if y is None else torch.cat([y, mx], dim=1))
        u = mx + (1 - self.mask) * (x - t) * torch.exp(
            -s)  # cf RealNVP eq 8 where u corresponds to x (here we're modeling u)

        log_abs_det_jacobian = - (
                    1 - self.mask) * s  # log det du/dx; cf RealNVP 8 and 6; note, sum over input_size done at model log_prob

        return u, log_abs_det_jacobian

    def inverse(self, u, y=None):
        # apply mask
        mu = u * self.mask

        # run through model
        s = self.s_net(mu if y is None else torch.cat([y, mu], dim=1))
        t = self.t_net(mu if y is None else torch.cat([y, mu], dim=1))
        x = mu + (1 - self.mask) * (u * s.exp() + t)  # cf RealNVP eq 7

        log_abs_det_jacobian = (1 - self.mask) * s  # log det dx/du

        return x, log_abs_det_jacobian


class BatchNorm(nn.Module):
    """ RealNVP BatchNorm layer """

    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.ones(input_size))

    def forward(self, x, cond_y=None):
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = x.var(0)  # note MAF paper uses biased variance estimate; ie x.var(0, unbiased=False)

            # update running mean
            self.running_mean.mul_(self.momentum).add_(self.batch_mean.data * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(self.batch_var.data * (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        # compute normalized input (cf original batch norm paper algo 1)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        y = self.log_gamma.exp() * x_hat + self.beta

        # compute log_abs_det_jacobian (cf RealNVP paper)
        log_abs_det_jacobian = self.log_gamma - 0.5 * torch.log(var + self.eps)
        #        print('in sum log var {:6.3f} ; out sum log var {:6.3f}; sum log det {:8.3f}; mean log_gamma {:5.3f}; mean beta {:5.3f}'.format(
        #            (var + self.eps).log().sum().data.numpy(), y.var(0).log().sum().data.numpy(), log_abs_det_jacobian.mean(0).item(), self.log_gamma.mean(), self.beta.mean()))
        return y, log_abs_det_jacobian.expand_as(x)

    def inverse(self, y, cond_y=None):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (y - self.beta) * torch.exp(-self.log_gamma)
        x = x_hat * torch.sqrt(var + self.eps) + mean

        log_abs_det_jacobian = 0.5 * torch.log(var + self.eps) - self.log_gamma

        return x, log_abs_det_jacobian.expand_as(x)


class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """

    def forward(self, x, y):
        sum_log_abs_det_jacobians = 0
        for module in self:
            x, log_abs_det_jacobian = module(x, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return x, sum_log_abs_det_jacobians

    def inverse(self, u, y):
        sum_log_abs_det_jacobians = 0
        for module in reversed(self):
            u, log_abs_det_jacobian = module.inverse(u, y)
            sum_log_abs_det_jacobians = sum_log_abs_det_jacobians + log_abs_det_jacobian
        return u, sum_log_abs_det_jacobians


# --------------------
# Models
# --------------------

class MADE(nn.Module):
    def __init__(self, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu',
                 input_order='sequential', input_degrees=None):
        """
        Args:
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn, MaskedLinear(hidden_size, 2 * input_size, masks[-1].repeat(2, 1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # MAF eq 4 -- return mean and log std
        m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
        u = (x - m) * torch.exp(-loga)
        # MAF eq 5
        log_abs_det_jacobian = - loga
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # MAF eq 3
        D = u.shape[1]
        x = torch.zeros_like(u)
        # run through reverse model
        for i in self.input_degrees:
            m, loga = self.net(self.net_input(x, y)).chunk(chunks=2, dim=1)
            x[:, i] = u[:, i] * torch.exp(loga[:, i]) + m[:, i]
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + log_abs_det_jacobian, dim=1)


class MADEMOG(nn.Module):
    """ Mixture of Gaussians MADE """

    def __init__(self, n_components, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu',
                 input_order='sequential', input_degrees=None):
        """
        Args:
            n_components -- scalar; number of gauassian components in the mixture
            input_size -- scalar; dim of inputs
            hidden_size -- scalar; dim of hidden layers
            n_hidden -- scalar; number of hidden layers
            activation -- str; activation function to use
            input_order -- str or tensor; variable order for creating the autoregressive masks (sequential|random)
                            or the order flipped from the previous layer in a stack of mades
            conditional -- bool; whether model is conditional
        """
        super().__init__()
        self.n_components = n_components

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # create masks
        masks, self.input_degrees = create_masks(input_size, hidden_size, n_hidden, input_order, input_degrees)

        # setup activation
        if activation == 'relu':
            activation_fn = nn.ReLU()
        elif activation == 'tanh':
            activation_fn = nn.Tanh()
        else:
            raise ValueError('Check activation function.')

        # construct model
        self.net_input = MaskedLinear(input_size, hidden_size, masks[0], cond_label_size)
        self.net = []
        for m in masks[1:-1]:
            self.net += [activation_fn, MaskedLinear(hidden_size, hidden_size, m)]
        self.net += [activation_fn,
                     MaskedLinear(hidden_size, n_components * 3 * input_size, masks[-1].repeat(n_components * 3, 1))]
        self.net = nn.Sequential(*self.net)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        # shapes
        N, L = x.shape
        C = self.n_components
        # MAF eq 2 -- parameters of Gaussians - mean, logsigma, log unnormalized cluster probabilities
        m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3, dim=-1)  # out 3 x (N, C, L)
        # MAF eq 4
        x = x.repeat(1, C).view(N, C, L)  # out (N, C, L)
        u = (x - m) * torch.exp(-loga)  # out (N, C, L)
        # MAF eq 5
        log_abs_det_jacobian = - loga  # out (N, C, L)
        # normalize cluster responsibilities
        self.logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
        return u, log_abs_det_jacobian

    def inverse(self, u, y=None, sum_log_abs_det_jacobians=None):
        # shapes
        N, C, L = u.shape
        # init output
        x = torch.zeros(N, L).to(u.device)
        # MAF eq 3
        # run through reverse model along each L
        for i in self.input_degrees:
            m, loga, logr = self.net(self.net_input(x, y)).view(N, C, 3 * L).chunk(chunks=3,
                                                                                   dim=-1)  # out 3 x (N, C, L)
            # normalize cluster responsibilities and sample cluster assignments from a categorical dist
            logr = logr - logr.logsumexp(1, keepdim=True)  # out (N, C, L)
            z = D.Categorical(logits=logr[:, :, i]).sample().unsqueeze(-1)  # out (N, 1)
            u_z = torch.gather(u[:, :, i], 1, z).squeeze()  # out (N, 1)
            m_z = torch.gather(m[:, :, i], 1, z).squeeze()  # out (N, 1)
            loga_z = torch.gather(loga[:, :, i], 1, z).squeeze()
            x[:, i] = u_z * torch.exp(loga_z) + m_z
        log_abs_det_jacobian = loga
        return x, log_abs_det_jacobian

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)  # u = (N,C,L); log_abs_det_jacobian = (N,C,L)
        # marginalize cluster probs
        log_probs = torch.logsumexp(self.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian,
                                    dim=1)  # sum over C; out (N, L)
        return log_probs.sum(1)  # sum over L; out (N,)


class MAF(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, activation='relu',
                 input_order='sequential', batch_norm=True):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))
        print("MAF", n_blocks, input_size, hidden_size, n_hidden, cond_label_size)
        # construct model
        modules = []
        self.input_degrees = None
        for i in range(n_blocks):
            modules += [
                MADE(input_size, hidden_size, n_hidden, cond_label_size, activation, input_order, self.input_degrees)]
            self.input_degrees = modules[-1].input_degrees.flip(0)
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        # y = y.squeeze()
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)


class MAFMOG(nn.Module):
    """ MAF on mixture of gaussian MADE """

    def __init__(self, n_blocks, n_components, input_size, hidden_size, n_hidden, cond_label_size=None,
                 activation='relu',
                 input_order='sequential', batch_norm=True):
        super().__init__()
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        self.maf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size, activation, input_order,
                       batch_norm)
        # get reversed input order from the last layer (note in maf model, input_degrees are already flipped in for-loop model constructor
        input_degrees = self.maf.input_degrees  # .flip(0)
        self.mademog = MADEMOG(n_components, input_size, hidden_size, n_hidden, cond_label_size, activation,
                               input_order, input_degrees)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        u, maf_log_abs_dets = self.maf(x, y)
        u, made_log_abs_dets = self.mademog(u, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return u, sum_log_abs_det_jacobians

    def inverse(self, u, y=None):
        x, made_log_abs_dets = self.mademog.inverse(u, y)
        x, maf_log_abs_dets = self.maf.inverse(x, y)
        sum_log_abs_det_jacobians = maf_log_abs_dets.unsqueeze(1) + made_log_abs_dets
        return x, sum_log_abs_det_jacobians

    def log_prob(self, x, y=None):
        u, log_abs_det_jacobian = self.forward(x, y)  # u = (N,C,L); log_abs_det_jacobian = (N,C,L)
        # marginalize cluster probs
        log_probs = torch.logsumexp(self.mademog.logr + self.base_dist.log_prob(u) + log_abs_det_jacobian,
                                    dim=1)  # out (N, L)
        return log_probs.sum(1)  # out (N,)


class RealNVP(nn.Module):
    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, cond_label_size=None, batch_norm=True):
        super().__init__()

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(input_size))
        self.register_buffer('base_dist_var', torch.ones(input_size))

        # construct model
        modules = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            modules += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask, cond_label_size)]
            mask = 1 - mask
            modules += batch_norm * [BatchNorm(input_size)]

        self.net = FlowSequential(*modules)

    @property
    def base_dist(self):
        return D.Normal(self.base_dist_mean, self.base_dist_var)

    def forward(self, x, y=None):
        return self.net(x, y)

    def inverse(self, u, y=None):
        return self.net.inverse(u, y)

    def log_prob(self, x, y=None):
        u, sum_log_abs_det_jacobians = self.forward(x, y)
        return torch.sum(self.base_dist.log_prob(u) + sum_log_abs_det_jacobians, dim=1)
