import torch
from tqdm import tqdm

class_num = 951

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t.clamp(min=-1) + 1).view(-1, 1, 1, 1)
    return a

def dps_diffusion(x, model, b, eta, A_funcs, y, cls_fn=None, classes=None, config=None):
    with torch.no_grad():
        skip = config.diffusion.num_diffusion_timesteps // config.time_travel.T_sampling
        seq = list(range(0, config.diffusion.num_diffusion_timesteps, skip))
        seq_next = [-1] + seq[:-1]
        time_pairs = list(zip(reversed(seq), reversed(seq_next)))
        
        n = x.size(0)
        x0_preds = []
        xs = [x]
        
        for i, j in tqdm(time_pairs):
            t = (torch.ones(n, device=x.device) * i).long()
            next_t = (torch.ones(n, device=x.device) * j).long()
            
            at = compute_alpha(b, t)
            at_next = compute_alpha(b, next_t)
            
            xt = xs[-1].to('cuda')
            
            with torch.enable_grad():
                xt_in = xt.detach().requires_grad_(True)
                
                # ==========================================
                # 🚀 1:1 完美复刻你的原版判定逻辑
                # ==========================================
                if cls_fn == None:
                    et = model(xt_in, t)
                else:
                    dps_classes = torch.ones(xt_in.size(0), dtype=torch.long, device=x.device) * class_num
                    et = model(xt_in, t, y=dps_classes)
                    et = et[:, :3]
                
                if et.size(1) == 6:
                    et = et[:, :3]
                # ==========================================
                
                x0_t = (xt_in - et * (1 - at).sqrt()) / at.sqrt()
                
                # 计算 L2 误差并求导
                loss = torch.norm(y - A_funcs.A(x0_t)) ** 2
                grad = torch.autograd.grad(outputs=loss, inputs=xt_in)[0]
            
            # --- DDIM 推断 ---
            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
            
            noise = torch.randn_like(xt) if j >= 0 else torch.zeros_like(xt)
            xt_next_uncond = at_next.sqrt() * x0_t.detach() + c1 * noise + c2 * et.detach()
            
            step_size = 1.0 / (torch.norm(grad) + 1e-8)
            xt_next = xt_next_uncond - step_size * grad
            
            x0_preds.append(x0_t.detach().to('cpu'))
            xs.append(xt_next.to('cpu'))
            
    return [xs[-1]], [x0_preds[-1]]