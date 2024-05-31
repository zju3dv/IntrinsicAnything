import math
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image
from torchvision.ops import masks_to_boxes
from torchvision.transforms import Resize 
from diffusers import DDIMScheduler, DDPMScheduler
from einops import rearrange, repeat
from tqdm import tqdm
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append("./models/")
from loguru import logger

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import extract_into_tensor

# load model
def load_model_from_config(config, ckpt, device, vram_O=False, verbose=True):

    pl_sd = torch.load(ckpt, map_location='cpu')

    sd = pl_sd['state_dict']

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0:
        logger.warning('missing keys: \n', m)
    if len(u) > 0:
        logger.warning('unexpected keys: \n', u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        logger.debug('loading EMA...')
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)
    # model.first_stage_model.train = True
    # model.first_stage_model.train()
    for param in model.first_stage_model.parameters():
        param.requires_grad = True

    return model

class MateralDiffusion(nn.Module):
    def __init__(self, device, fp16,
                 config=None,
                 ckpt=None, vram_O=False, t_range=[0.02, 0.98], opt=None, use_ddim=True):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.vram_O = vram_O
        self.t_range = t_range
        self.opt = opt

        self.config = OmegaConf.load(config)
        # TODO: seems it cannot load into fp16...
        self.model = load_model_from_config(self.config, ckpt, device=self.device, vram_O=vram_O, verbose=True)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps

        self.use_ddim = use_ddim

        if self.use_ddim: 
            self.scheduler = DDIMScheduler(
                self.num_train_timesteps,
                self.config.model.params.linear_start,
                self.config.model.params.linear_end,
                beta_schedule='scaled_linear',
                clip_sample=False,
                set_alpha_to_one=False,
                steps_offset=1,
            )
            print("Using DDIM...")
        else:
            self.scheduler = DDPMScheduler(
                self.num_train_timesteps,
                self.config.model.params.linear_start,
                self.config.model.params.linear_end,
                beta_schedule='scaled_linear',
                clip_sample=False,
            )
            print("Using DDPM...")


        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

    def get_input(self, x):
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def _image2diffusion(self, embeddings, pred_rgb, mask, image_size=256):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]
        # assert pred_rgb.w
        assert len(pred_rgb.shape) == 4, f"except 4 dim tensor, got: {pred_rgb.shape}"

        cond_img = embeddings["cond_img"]

        xc = self.get_input(cond_img)
        pred_rgb = self.get_input(pred_rgb)

        return pred_rgb, xc
    
    def _get_condition(self, xc, with_uncondition=False):
        # To support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        # z.shape: [8, 4, 64, 64]; c.shape: [8, 1, 768]
        # print('=========== xc shape ===========', xc.shape)

        # print(xc.shape, xc.min(), xc.max(), self.model.use_clip_embdding)
        xc = xc * 2 - 1
        cond = {}
        clip_emb = self.model.get_learned_conditioning(xc if self.model.use_clip_embdding else [""]).detach()
        c_concat = self.model.encode_first_stage((xc.to(self.device))).mode().detach()
        # print(clip_emb.shape, clip_emb.min(), clip_emb.max(), self.model.use_clip_embdding)
        if with_uncondition:
            cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
            cond['c_concat'] = [torch.cat([torch.zeros_like(c_concat).to(self.device), c_concat], dim=0)]
        else:
            cond['c_crossattn'] = [clip_emb]
            cond['c_concat'] = [c_concat]
        return cond

    @torch.no_grad()
    def __call__(self, embeddings, pred_rgb, mask, guidance_scale=3, dps_scale=0.2, as_latent=False, grad_scale=1, save_guidance_path:Path=None,
        ddim_steps=200, ddim_eta=1, operator=None):
        # todo: The upsacle is currectly hard-coded
        upscale = 1
        
        # with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_rgb_256, xc = self._image2diffusion(embeddings, pred_rgb, mask, image_size=256*upscale)
        cond = self._get_condition(xc, with_uncondition=True)
        assert pred_rgb_256.shape[-1] == pred_rgb_256.shape[-2], f"Expect image of square size, get {pred_rgb.shape}"

        latents = torch.randn_like(self.encode_imgs(pred_rgb_256))

        if self.use_ddim:
            self.scheduler.set_timesteps(ddim_steps)
        else:
            self.scheduler.set_timesteps(self.num_train_timesteps)

        intermidates = []

        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.view(1).expand(latents.shape[0])] * 2).to(self.device)

            noise_pred = self.model.apply_model(x_in, t_in, cond)

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # dps
            if dps_scale > 0:
                with torch.enable_grad():
                    t_batch = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device) * 0 + t
                    x_hat_latents = self.model.predict_start_from_noise(latents.requires_grad_(True), t_batch, noise_pred)
                    x_hat = self.decode_latents(x_hat_latents)
                    x_hat = operator.forward(x_hat)
                    norm = torch.linalg.norm((pred_rgb_256-x_hat).reshape(pred_rgb_256.shape[0], -1), dim=-1)
                    guidance_score = torch.autograd.grad(norm.sum(), latents, retain_graph=True)[0]

                if (not save_guidance_path is None) and i % (len(self.scheduler.timesteps)//20) == 0:
                    x_t = self.decode_latents(latents)
                    intermidates.append(torch.cat([x_hat, x_t, pred_rgb_256, pred_rgb_256-x_hat], dim=-2).detach().cpu())
                
                # print("before", noise_pred[0, 2, 10, 16:22], noise_pred.shape, dps_scale)
                logger.debug(f"Guidance loss: {norm}")
                noise_pred = noise_pred + dps_scale * guidance_score


            if self.use_ddim:
                latents = self.scheduler.step(noise_pred, t, latents, eta=ddim_eta)['prev_sample']
            else:
                latents = self.scheduler.step(noise_pred.clone().detach(), t, latents)['prev_sample']
            if dps_scale > 0:
                del x_hat
                del guidance_score
                del noise_pred
                del x_hat_latents
                del norm

        imgs = self.decode_latents(latents)
        viz_images = torch.cat([pred_rgb_256, imgs],dim=-1)[:1]
        # save_image(viz_images, "vis.jpg")

        if not save_guidance_path is None and len(intermidates) > 0:
            save_image(viz_images, save_guidance_path)

            viz_images = torch.cat(intermidates,dim=-1)[:1]
            save_image(viz_images, save_guidance_path+"all.jpg")

        if not save_guidance_path is None:
            save_image(imgs[:1], save_guidance_path+"_out.jpg")
        return rearrange(imgs, 'b c h w -> b h w c')

    def decode_latents(self, latents):
        # zs: [B, 4, 32, 32] Latent space image
        # with self.model.ema_scope():
        imgs = self.model.decode_first_stage(latents)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs # [B, 3, 256, 256] RGB space image

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256] RGB space image
        # with self.model.ema_scope():
        imgs = imgs * 2 - 1
        # latents = torch.cat([self.model.get_first_stage_encoding(self.model.encode_first_stage(img.unsqueeze(0))) for img in imgs], dim=0)
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))

        return latents # [B, 4, 32, 32] Latent space image