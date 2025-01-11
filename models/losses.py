import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils
from torch.autograd import grad
import itertools
import math


def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        # 如果二者都有，就将他们沿第二维拼接起来生成一个综合的梯度张量all_grads
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)

        # 如果只有一个存在，则直接将它赋值给all_grads
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

        # 根据Eikonal_type的值选择损失类型
    if eikonal_type == 'abs':       # abs计算的是绝对误差
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()        # 计算的是梯度在3维空间中的L2范数（欧几里得距离），然后-1
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()     # 否则计算平方误差

        # 这个损失通过penalizing梯度的偏离，帮助模型生成一个更平滑的表面或形状
    return eikonal_term         # 返回损失，这是梯度和单位梯度之间偏离的平均值


def latent_rg_loss(latent_reg, device):
    # compute the VAE latent representation regularization loss
    if latent_reg is not None:
        reg_loss = latent_reg.mean()    # 平均值
    else:
        reg_loss = torch.tensor([0.0], device=device)   # 创建一个值为0.0的张量并分配到指定设备上以保证兼容性

    return reg_loss


def DT(t):
    pi = math.pi
    return (
        ((64 * pi - 80) / pi**4) * t**4
        - ((64 * pi - 88) / pi**3) * t**3
        + ((16 * pi - 29) / pi**2) * t**2
        + (3 / pi) * t
    )


def gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad):
    device = morse_nonmnfld_grad.device

    # 扩展 Hessian 矩阵
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, morse_nonmnfld_grad[:, :, :, None]), dim=-1)
    zero_grad = torch.zeros(
        (morse_nonmnfld_grad.shape[0], morse_nonmnfld_grad.shape[1], 1, 1),
        device=device)
    zero_grad = torch.cat((morse_nonmnfld_grad[:, :, None, :], zero_grad), dim=-1)
    nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)

    # 计算曲率
    morse_nonmnfld = (-1. / (morse_nonmnfld_grad.norm(dim=-1) ** 2 + 1e-12)) * torch.det(nonmnfld_hessian_term)
    morse_nonmnfld = morse_nonmnfld.abs()

    # 应用 DT(t)
    morse_nonmnfld = DT(morse_nonmnfld)

    # 计算均值作为损失
    morse_loss = morse_nonmnfld.mean()

    return morse_loss


# def gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad):
#     device = morse_nonmnfld_grad.device
#     nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, morse_nonmnfld_grad[:, :, :, None]), dim=-1)
#     zero_grad = torch.zeros(
#         (morse_nonmnfld_grad.shape[0], morse_nonmnfld_grad.shape[1], 1, 1),
#         device=device)
#     zero_grad = torch.cat((morse_nonmnfld_grad[:, :, None, :], zero_grad), dim=-1)
#     nonmnfld_hessian_term = torch.cat((nonmnfld_hessian_term, zero_grad), dim=-2)
#     morse_nonmnfld = (-1. / (morse_nonmnfld_grad.norm(dim=-1) ** 2 + 1e-12)) * torch.det(
#         nonmnfld_hessian_term)

#     morse_nonmnfld = morse_nonmnfld.abs()

#     morse_loss = morse_nonmnfld.mean()

#     return morse_loss
class MorseLoss(nn.Module):
    def __init__(self, weights=None, loss_type='siren_wo_n_w_morse', div_decay='none',
                 div_type='l1', bidirectional_morse=True, udf=False):
        super().__init__()
        if weights is None:
            weights = [3e3, 1e2, 1e2, 5e1, 1e2, 1e1]
        self.weights = weights  # sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_morse = True if 'morse' in self.loss_type else False
        self.bidirectional_morse = bidirectional_morse
        self.udf = udf

    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None, near_points=None):
        # output_pred：模型输出的预测结果，包含非流形和流形点的预测、潜在向量的正则项等
        # mnfld_points：流形上的点，用于计算流形点相关的梯度、曲率等信息
        # nonmnfld_points：非流形上的点，用于计算非流形点的梯度、曲率等信息
        # mnfld_n_gt和near_points：其他辅助点信息
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        #########################################
        # Compute required terms
        #########################################

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]    # 从output_pred中提取
        manifold_pred = output_pred["manifold_pnts_pred"]
        latent_reg = output_pred["latent_reg"]

        div_loss = torch.tensor([0.0], device=mnfld_points.device)  # 初始化各种损失项
        morse_loss = torch.tensor([0.0], device=mnfld_points.device)
        curv_term = torch.tensor([0.0], device=mnfld_points.device)
        normal_term = torch.tensor([0.0], device=mnfld_points.device)
        min_surf_loss = torch.tensor([0.0], device=mnfld_points.device)

        # compute gradients for div (divergence), curl and curv (curvature)
        if manifold_pred is not None:
            mnfld_grad = utils.gradient(mnfld_points, manifold_pred)    # 计算流形和非流形点的梯度
        else:
            mnfld_grad = None

        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        morse_nonmnfld_points = None
        morse_nonmnfld_grad = None

        if self.use_morse and near_points is not None:                  
            morse_nonmnfld_points = near_points
            morse_nonmnfld_grad = utils.gradient(near_points, output_pred['near_points_pred'])

            
        elif self.use_morse and near_points is None:
            morse_nonmnfld_points = nonmnfld_points
            morse_nonmnfld_grad = nonmnfld_grad

        if self.use_morse:      # 如果启用了 Morse 函数 (self.use_morse 为 True)，则基于 near_points 或 nonmnfld_points 计算所需梯度
            nonmnfld_dx = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 0])
            nonmnfld_dy = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 1])

            mnfld_dx = utils.gradient(mnfld_points, mnfld_grad[:, :, 0])
            mnfld_dy = utils.gradient(mnfld_points, mnfld_grad[:, :, 1])
            if dims == 3:       # 如果维度为 3，则分别计算沿 x、y、z 轴的梯度并构建 3x3 Hessian 矩阵，否则构建 2x2 Hessian 矩阵
                nonmnfld_dz = utils.gradient(morse_nonmnfld_points, morse_nonmnfld_grad[:, :, 2])
                nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy, nonmnfld_dz), dim=-1)

                mnfld_dz = utils.gradient(mnfld_points, mnfld_grad[:, :, 2])
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy, mnfld_dz), dim=-1)
            else:
                nonmnfld_hessian_term = torch.stack((nonmnfld_dx, nonmnfld_dy), dim=-1)
                mnfld_hessian_term = torch.stack((mnfld_dx, mnfld_dy), dim=-1)

            morse_mnfld = torch.tensor([0.0], device=mnfld_points.device)
            if self.div_type == 'l1':       # 使用 gaussian_curvature 函数计算流形和非流形点上的高斯曲率损失，并根据是否双向计算 (self.bidirectional_morse) 取其均值
                morse_loss = gaussian_curvature(nonmnfld_hessian_term, morse_nonmnfld_grad)

                if self.bidirectional_morse:
                    morse_mnfld = gaussian_curvature(mnfld_hessian_term, mnfld_grad)

                morse_loss = 0.5 * (morse_loss + morse_mnfld)
        # latent regulariation for multiple shape learning
        latent_reg_term = latent_rg_loss(latent_reg, device)        # 计算潜在空间正则化损失。

        # signed distance function term
        sdf_term = torch.abs(manifold_pred).mean()      # 用于调整流形预测的绝对值

        # eikonal term
        eikonal_term = eikonal_loss(morse_nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

        # losses used in the paper
        if self.loss_type == 'siren_wo_n_w_morse':
            self.weights[2] = 0     # siren类型的损失的话就生成组合损失
            loss = self.weights[0] * sdf_term + self.weights[1] * inter_term + self.weights[3] * eikonal_term + \
                   self.weights[5] * morse_loss
        else:
            print(self.loss_type)
            raise Warning("unrecognized loss type")

        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term, 'latent_reg_term': latent_reg_term,
                'eikonal_term': eikonal_term, 'normals_loss': normal_term, 'div_loss': div_loss,
                'curv_loss': curv_term.mean(), 'morse_term': morse_loss, 'min_surf_loss': min_surf_loss}, mnfld_grad

    def update_morse_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.7, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            # 如果尚未初始化self.decay_params_list，则创建一个按百分比配对的权重变化表
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]], [0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        # we, e 表示找到的下一个训练阶段权重及其位置
        we, e = min([tup for tup in self.decay_params_list if tup[1] >= curr], key=lambda tup: tup[1])
        # w0, s 表示当前阶段的起始权重和位置
        w0, s = max([tup for tup in self.decay_params_list if tup[1] <= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions 退火函数
        if self.div_decay == 'linear':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0

                # 线性衰减 linear：在当前阶段的开始（s）和结束（e）之间，权重以线性插值的方式从 w0 逐渐变化到 we
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (current_iteration / n_iterations - s) / (e - s)
            else:
                self.weights[5] = we

                # 五次方衰减 quintic：使用五次方插值，以更平滑的方式逐渐调整权重值，使其从 w0 过渡到 we
        elif self.div_decay == 'quintic':  # linearly decrease weight from iter s to iter e
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            elif current_iteration >= s * n_iterations and current_iteration < e * n_iterations:
                self.weights[5] = w0 + (we - w0) * (1 - (1 - (current_iteration / n_iterations - s) / (e - s)) ** 5)
            else:
                self.weights[5] = we

                # 阶梯模式 step：在 s 点直接改变权重值，无需插值
        elif self.div_decay == 'step':  # change weight at s
            if current_iteration < s * n_iterations:
                self.weights[5] = w0
            else:
                self.weights[5] = we

                # 无衰减 none：保持原始权重不变
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")
