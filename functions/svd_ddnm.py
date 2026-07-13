import torch
from tqdm import tqdm
import torchvision.utils as tvu
import torchvision
import os
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
class_num = 951


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def inverse_data_transform(x):
    x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


import torch
from tqdm import tqdm


def compute_alpha(beta, t):
    """
    beta: [T]
    t: [B]
    返回 alpha_t = prod(1 - beta)
    支持 t = -1，此时 alpha = 1
    """
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1)
    return a.view(-1, 1, 1, 1)





def ddnm_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')
                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())

                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

def ddnm_EquS_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))
        Tg = transforms.RandomHorizontalFlip(p=1)
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')

                if ((i//skip) % 2) != 1:
                    xt = Tg(xt)

                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                if ((i//skip) % 2) != 1:
                    x0_t = Tg(x0_t)

                

                # using guidance
                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())
                
                # ddim sampler
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

def ddnm_mDSAR_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():
        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat)
        time_pairs = list(zip(times[:-1], times[1:]))
        Tg = transforms.RandomHorizontalFlip(p=1)
        
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i * skip, j * skip
            if j < 0: j = -1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')

                if ((i // skip) % 2) != 1:
                    xt = Tg(xt)

                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda")) * class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                if ((i // skip) % 2) != 1:
                    x0_t = Tg(x0_t)

                # ==========================================
                # 🚀 NOVELTY 1: 双统计自适应特征掩码提取 (DSAR)
                # ==========================================
                with torch.enable_grad():
                    x0_t_opt = x0_t.detach().clone().requires_grad_(True)

                    # 1. 一阶语义掩码 (保护舰船主干)
                    intensity = (x0_t_opt.detach().mean(dim=1, keepdim=True) + 1.0) / 2.0
                    mask_ship = torch.sigmoid(15.0 * (intensity - 0.20))
                    mask_bg = 1.0 - mask_ship

                    # 2. 二阶局部标准差权重 (保护高频纹理)
                    def compute_sdgw(x, window_size=7, eps=1e-8):
                        B, C, H, W = x.shape
                        unfold = torch.nn.functional.unfold(
                            x, kernel_size=window_size, padding=window_size // 2
                        )
                        mean = unfold.mean(dim=1, keepdim=True)
                        var = (unfold - mean).pow(2).mean(dim=1, keepdim=True)
                        std = var.sqrt() + eps
                        std_map = torch.nn.functional.fold(std, output_size=(H, W), kernel_size=1)
                        std_map = (std_map - std_map.min()) / (std_map.max() - std_map.min() + eps)
                        return std_map

                    sdgw_weight = compute_sdgw(x0_t_opt.detach(), window_size=7)
                    sdgw_weight = 1.0 - sdgw_weight  # 反转：海面权重趋近1，舰船边缘趋近0

                    # 💡 核心：融合得到终极的 DSAR 空间掩码
                    M_DSAR = mask_bg * sdgw_weight

                    # ==========================================
                    # 🚀 NOVELTY 2: 空间自适应 SAR-TV 优化
                    # ==========================================
                    diff_i_mat = (x0_t_opt[:, :, :, 1:] - x0_t_opt[:, :, :, :-1]) ** 2
                    diff_j_mat = (x0_t_opt[:, :, 1:, :] - x0_t_opt[:, :, :-1, :]) ** 2

                    # 分配 TV 权重 (仅在 M_DSAR 大的平坦海面区域施加强平滑)
                    adaptive_weight_i = M_DSAR[:, :, :, :-1]
                    adaptive_weight_j = M_DSAR[:, :, :-1, :]

                    tv_loss = torch.sum(diff_i_mat * adaptive_weight_i) + torch.sum(diff_j_mat * adaptive_weight_j)
                    grad_tv = torch.autograd.grad(outputs=tv_loss, inputs=x0_t_opt)[0]

                    lambda_tv = 0.050 
                    x0_t_reg = x0_t.detach() - lambda_tv * grad_tv
                    x0_t = torch.clamp(x0_t_reg, -1.0, 1.0)
                # ==========================================

                # using guidance (数据一致性约束)
                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())
                
                # ==========================================
                # 🚀 NOVELTY 3: RDBM (CVPR 2026) 残差调制扩散采样
                # 抛弃传统 DDIM 全局加噪，利用 DSAR 物理掩码进行零样本残差调制！
                # ==========================================
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                
                # 标准高斯噪声
                standard_noise = torch.randn_like(x0_t)
                
                # 💡 核心魔法：物理空间掩码调制 (Residual Modulation)
                # 海面区域 (M_DSAR~1) -> 注入完整噪声，加速平滑重建
                # 舰船区域 (M_DSAR~0) -> 阻断噪声注入，完美“冻结”高频物理边缘
                modulated_noise = M_DSAR * standard_noise

                xt_next = at_next.sqrt() * x0_t_hat + c1 * modulated_noise + c2 * et
                # ==========================================

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
                
            else: 
                # time-travel back (时空穿梭重采样)
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                # 同理，回退时的随机漫步也必须加入 RDBM 空间调制约束
                standard_noise_travel = torch.randn_like(x0_t)
                modulated_noise_travel = M_DSAR * standard_noise_travel
                
                xt_next = at_next.sqrt() * x0_t + modulated_noise_travel * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]

def ddnm_DSAR_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():

        # setup iteration variables
        skip = config.diffusion.num_diffusion_timesteps//config.time_travel.T_sampling
        n = x.size(0)
        x0_preds = []
        xs = [x]

        # generate time schedule
        times = get_schedule_jump(config.time_travel.T_sampling, 
                               config.time_travel.travel_length, 
                               config.time_travel.travel_repeat,
                              )
        time_pairs = list(zip(times[:-1], times[1:]))
        Tg = transforms.RandomHorizontalFlip(p=1)
        # reverse diffusion sampling
        for i, j in tqdm(time_pairs):
            i, j = i*skip, j*skip
            if j<0: j=-1 

            if j < i: # normal sampling 
                t = (torch.ones(n) * i).to(x.device)
                next_t = (torch.ones(n) * j).to(x.device)
                at = compute_alpha(b, t.long())
                at_next = compute_alpha(b, next_t.long())
                xt = xs[-1].to('cuda')

                if ((i//skip) % 2) != 1:
                    xt = Tg(xt)

                if cls_fn == None:
                    et = model(xt, t)
                else:
                    classes = torch.ones(xt.size(0), dtype=torch.long, device=torch.device("cuda"))*class_num
                    et = model(xt, t, classes)
                    et = et[:, :3]
                    et = et - (1 - at).sqrt()[0, 0, 0, 0] * cls_fn(x, t, classes)

                if et.size(1) == 6:
                    et = et[:, :3]

                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

                if ((i//skip) % 2) != 1:
                    x0_t = Tg(x0_t)

                # ==========================================
                # 🚀 NOVELTY 1 & 2: Spatially Adaptive SAR-TV + SCGN SDGW 融合版
                # ==========================================
                with torch.enable_grad():
                    x0_t_opt = x0_t.detach().clone().requires_grad_(True)

                    # 1. 你原来的差分计算
                    diff_i_mat = (x0_t_opt[:, :, :, 1:] - x0_t_opt[:, :, :, :-1]) ** 2
                    diff_j_mat = (x0_t_opt[:, :, 1:, :] - x0_t_opt[:, :, :-1, :]) ** 2

                    # 2. 你原来的语义掩码（保护舰船）
                    intensity = (x0_t_opt.detach().mean(dim=1, keepdim=True) + 1.0) / 2.0
                    mask_ship = torch.sigmoid(15.0 * (intensity - 0.20))
                    mask_bg = 1.0 - mask_ship
                    mask_bg_i = mask_bg[:, :, :, :-1]
                    mask_bg_j = mask_bg[:, :, :-1, :]

                    # 3. 🚀 新增：SCGN SDGW（局部标准差权重）
                    # 参数建议：SAR船舰用 window_size=5~7（根据船大小调）
                    def compute_sdgw(x, window_size=7, eps=1e-8):
                        # 使用 unfold 高效计算局部 std（即插即用，无需额外网络）
                        B, C, H, W = x.shape
                        unfold = torch.nn.functional.unfold(
                            x,
                            kernel_size=window_size,
                            padding=window_size // 2,
                        )
                        # unfold 形状: [B, C*window^2, H*W]
                        mean = unfold.mean(dim=1, keepdim=True)
                        var = (unfold - mean).pow(2).mean(dim=1, keepdim=True)
                        std = var.sqrt() + eps
                        # 折回图像尺寸
                        std_map = torch.nn.functional.fold(std, output_size=(H, W), kernel_size=1)
                        # 归一化到 [0,1]（std越大 → 结构越强 → 权重越小（保护））
                        std_map = (std_map - std_map.min()) / (std_map.max() - std_map.min() + eps)
                        return std_map  # 值越大 = 结构区（应弱平滑）

                    sdgw_weight = compute_sdgw(x0_t_opt.detach(), window_size=7)
                    # 反转：std大（结构）→ 权重小（少平滑），std小（海面）→ 权重大（强平滑）
                    sdgw_weight = 1.0 - sdgw_weight

                    # 4. 融合：mask_bg * sdgw_weight 做最终权重
                    adaptive_weight_i = mask_bg_i * sdgw_weight[:, :, :, :-1]
                    adaptive_weight_j = mask_bg_j * sdgw_weight[:, :, :-1, :]

                    # 5. 加权 TV loss（只在“海面+低方差”区域强平滑）
                    tv_loss = torch.sum(diff_i_mat * adaptive_weight_i) + torch.sum(diff_j_mat * adaptive_weight_j)

                    grad_tv = torch.autograd.grad(outputs=tv_loss, inputs=x0_t_opt)[0]

                # 6. 最终调节（可把 lambda_tv 调大一点，SDGW 让它更安全）
                lambda_tv = 0.050  # 原来 0.015，现在可略微提高
                x0_t_reg = x0_t.detach() - lambda_tv * grad_tv
                x0_t = torch.clamp(x0_t_reg, -1.0, 1.0)
                # ==========================================

                # using guidance
                x0_t_hat = x0_t - A_funcs.A_pinv(
                    A_funcs.A(x0_t.reshape(x0_t.size(0), -1)) - y.reshape(y.size(0), -1)
                ).reshape(*x0_t.size())
                
                # ddim sampler
                c1 = (1 - at_next).sqrt() * eta
                c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t) + c2 * et

                x0_preds.append(x0_t.to('cpu'))
                xs.append(xt_next.to('cpu'))
            else: # time-travel back
                next_t = (torch.ones(n) * j).to(x.device)
                at_next = compute_alpha(b, next_t.long())
                x0_t = x0_preds[-1].to('cuda')
                
                xt_next = at_next.sqrt() * x0_t + torch.randn_like(x0_t) * (1 - at_next).sqrt()

                xs.append(xt_next.to('cpu'))

    return [xs[-1]], [x0_preds[-1]]



def get_schedule(timesteps, T_sampling, skip_type):

    if skip_type == "uniform":
            skip = timesteps // T_sampling
            seq = range(timesteps, 0, skip)
    elif skip_type == "quad":
        seq = (
            np.linspace(
                np.sqrt(timesteps-1), 0, T_sampling
            )
            ** 2
        )
        seq = [int(s) for s in list(seq)]

    seq.append(-1)
    return seq

# form RePaint
def get_schedule_jump(T_sampling, travel_length, travel_repeat):

    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)

    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)
