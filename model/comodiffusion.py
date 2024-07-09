import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import copy
import numpy as np
from einops import rearrange
from model.baseblock import (
    ResnetBlock,
    Residual,
    Block,
    Mish,
    Upsample,
    Downsample,
    Rezero,
    LinearAttention,
)


class PosEmbedding(pl.LightningModule):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator2d(pl.LightningModule):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        groups=8,
        multi_spks=1,
        spk_emb_dim=512,
        n_feats=80,
        pe_scale=1000,
    ):
        super().__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.multi_spks = multi_spks
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if self.multi_spks:
            self.spk_mlp = nn.Sequential(
                nn.Linear(spk_emb_dim, spk_emb_dim * 4),
                Mish(),
                nn.Linear(spk_emb_dim * 4, n_feats),
            )

        self.time_pos_emb = PosEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )

        dims = [2 + (1 if self.multi_spks else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )

        self.final_block = Block(dim, dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if not self.multi_spks:
            x = torch.stack([mu, x], 1)

        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.size(-1))
            x = torch.stack([mu, x, s], 1)

        mask = mask.unsqueeze(1)
        hiddens = []
        masks = [mask]

        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)

   

class ComoDiffusion(pl.LightningModule):
    def __init__(
        self,
        n_feats,
        dim,
        multi_spks=1,
        spk_emb_dim=512,
        beta_min=0.05,
        beta_max=20,
        pe_scale=1000,
        config=dict(),
        teacher=False,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.multi_spks = multi_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.config = config
        self.teacher=teacher
        self.estimator = GradLogPEstimator2d(
            dim,
            multi_spks=multi_spks,
            spk_emb_dim=spk_emb_dim,
            pe_scale=pe_scale,
            n_feats=self.n_feats,
        )
        self.config["perceptual_loss"]=True
        
        self.P_mean =-1.2 # P_mean
        self.P_std =1.2# P_std
        self.sigma_data =0.5# sigma_data
        self.sigma_min= 0.002
        self.sigma_max= 80
        self.rho=7
        self.N = 50         #100   
 
        # Time step discretization
        step_indices = torch.arange(self.N )   
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (self.N - 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho
        self.t_steps = torch.cat([torch.zeros_like(t_steps[:1]), self.round_sigma(t_steps)])  
 
        if not teacher:
            self.estimator_ema = copy.deepcopy(self.estimator)
            self.estimator_pretrained = copy.deepcopy(self.estimator)

    
    
    def EDMLoss(self, x_start,   cond,nonpadding ,spk=None):
 
        rnd_normal = torch.randn([x_start.shape[0], 1,  1], device=x_start.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

 
        n = (torch.randn_like(x_start)+cond ) * sigma
        D_yn = self.EDMPrecond(x_start + n, sigma ,cond,self.estimator,nonpadding,spk) #(8,128,128)
        loss = (weight * ((D_yn - x_start) ** 2))#(8,128,128)
        loss=loss*nonpadding 
        loss=loss.mean()
        xt_hat= D_yn*nonpadding
        return loss,xt_hat
    def EDMPrecond(self, x, sigma ,cond,estimator,mask,spk=None):
 
        sigma = sigma.reshape(-1, 1, 1 )
        #논문의 9번 식
        c_skip = self.sigma_data ** 2 / ((sigma-self.sigma_min) ** 2 + self.sigma_data ** 2)
        c_out = (sigma-self.sigma_min) * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
 
        F_x =  estimator((c_in * x), mask, cond,c_noise.flatten(),spk) 
        D_x = c_skip * x + c_out * (F_x  )
        return D_x
    
    def edm_sampler(self,
        latents,  cond,nonpadding,spk,
        num_steps=50, sigma_min=0.002, sigma_max=80, rho=7, 
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
        
    ):
        # Time step discretization.
        num_steps=num_steps+1
        step_indices = torch.arange(num_steps,   device=latents.device)

        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([self.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  

        # Main sampling loop.
        x_next = latents * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  
            x_cur = x_next
            # print('step',i+1)
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = self.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            denoised = self.EDMPrecond(x_hat, t_hat , cond,self.estimator,nonpadding,spk) 
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
        return x_next
    
    def forward(self, x,nonpadding,cond, spk=None, t_steps=1, infer=False):
        #x : ground truth(16,80,172), nonpadding : y_mask(16,1,172), cond : mu_y(16,80,172)
        if self.teacher: #teacher model  
            if not infer:
                loss,xt_hat = self.EDMLoss(x, cond,nonpadding,spk)
                return loss,xt_hat
            else:
                shape = (cond.shape[0],   cond.shape[1], cond.shape[2])
                x = torch.randn(shape, device=x.device)+cond
                x=self.edm_sampler(x, cond,nonpadding,spk,t_steps)
            return x
        else:  #Consistency distillation
            if not  infer:
                loss = self.CTLoss_D(x, cond,nonpadding)
                return loss
            else:
                shape = (cond.shape[0],   80, cond.shape[2])
                x = torch.randn(shape, device=x.device)+cond    
                x=self.CT_sampler(x, cond,nonpadding,t_steps)
            return x

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(
            x0.size(0), dtype=x0.dtype, device=self.device, requires_grad=False
        )
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk)
     
    
    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)
    
    def CTLoss_D(self,y, cond,nonpadding): #k 
        #y,x(groundtruth), cond(mu_y, prediction),nonpadding(y_mask)
        with torch.no_grad(): #새로운 EMA 파라미터는 이전 EMA 파라미터와 현재 모델 파라미터의 가중합으로 업데이트
            mu = 0.95  
            for p, ema_p in zip(self.denoise_fn.parameters(), self.denoise_fn_ema.parameters()):
                ema_p.mul_(mu).add_(p, alpha=1 - mu)

        #n:tensor([6,32,45 ... 16개])
        n = torch.randint(1, self.N, (y.shape[0],)) #batch size만큼 (16)
        z = torch.randn_like(y)+ cond #(16,80,172)
        #tn_1 (16,1,1), 0-1 사이의 값
        tn_1 = self.c_t_d(n + 1 ).reshape(-1, 1,   1).to(y.device) #tn_1은 특정 시점에서의 시그마 값
        f_theta = self.EDMPrecond(y + tn_1 * z, tn_1, cond, self.denoise_fn,nonpadding) #EDMPrecond가 D_theta
        #y + tn_1 * z는 diffusion 모델에서 데이터에 노이즈를 추가하는 과정, 즉, 논문 11번 식의 x_(i+1)를 만들어냈다.
        with torch.no_grad():
            tn = self.c_t_d(n ).reshape(-1, 1,   1).to(y.device)

            #euler step
            x_hat = y + tn_1 * z
            denoised = self.EDMPrecond(x_hat, tn_1 , cond,self.denoise_fn_pretrained,nonpadding) 
            d_cur = (x_hat - denoised) / tn_1
            y_tn = x_hat + (tn - tn_1) * d_cur
 
            f_theta_ema = self.EDMPrecond( y_tn, tn,cond, self.denoise_fn_ema,nonpadding) #11번 식의 2번째 D_theta

        loss =   (f_theta - f_theta_ema.detach()) ** 2  #논문 11번식
        loss=loss*nonpadding 
        loss=loss.mean() 

        return loss
    def c_t_d(self, i ):


        return self.t_steps[i]

    def get_t_steps(self,N):
        N=N+1
        step_indices = torch.arange( N ) #, device=latents.device)
        t_steps = (self.sigma_min ** (1 / self.rho) + step_indices / (N- 1) * (self.sigma_max ** (1 / self.rho) - self.sigma_min ** (1 / self.rho))) ** self.rho

        return  t_steps.flip(0)
