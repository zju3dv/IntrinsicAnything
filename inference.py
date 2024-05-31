import os
import imageio
import numpy as np

import glob
import sys
from typing import Any
sys.path.insert(1, '.')

import argparse
from pytorch_lightning import seed_everything
from PIL import Image
import torch
from models.operators import GaussialBlurOperator
from models.utils import get_rank, synchronize, get_world_size
from torchvision.ops import masks_to_boxes
from models.matfusion import MateralDiffusion
from loguru import logger
from torchvision.transforms import Resize 
seed_everything(0)

def set_loggers(level):
    logger.remove()
    logger.add(sys.stderr, level=level)

def init_model(ckpt_path, ddim, gpu_id):
    # find config
    configs = os.listdir(f'{ckpt_path}/configs')
    model_config = [config for config in configs if "project.yaml" in config][0]
    sds_loss_class = MateralDiffusion(device=gpu_id, fp16=True,
                        config=f'{ckpt_path}/configs/{model_config}',
                        ckpt=f'{ckpt_path}/checkpoints/last.ckpt', vram_O=False, 
                        t_range=[0.001, 0.02], opt=None, use_ddim=ddim)
    return sds_loss_class

def pad_to_divisible(num, divsor):
    return (divsor - num%(divsor)) * (num%(divsor)!=0)

def images_spliter(image_input, seg_h, seg_w, padding_pixel, padding_val, overlaps=1, processor=None):
    # split the input images along height and weidth by 
    # return a list of images
    h, w, c = image_input.shape

    # image padding to devisible
    h_pad = pad_to_divisible(h, seg_h*(overlaps+1))
    w_pad = pad_to_divisible(w, seg_w*(overlaps+1))
    h = h + h_pad
    w = w + w_pad
    image = torch.ones(h,w,c).to(image_input.device)
    image[:h-h_pad, :w-w_pad, :] = image_input


    h_crop = h // seg_h
    w_crop = w // seg_w
    images = []
    positions = []
    img_padded = torch.zeros(h+padding_pixel*2, w+padding_pixel*2, 3, device=image.device) + padding_val
    img_padded[padding_pixel:h+padding_pixel, padding_pixel:w+padding_pixel, :] = image[:h, :w]

    # overlapped sampling
    seg_h = np.round((h - h_crop) / h_crop * (overlaps+1)).astype(int) + 1
    seg_w = np.round((w - w_crop) / w_crop * (overlaps+1)).astype(int) + 1

    h_step = np.round(h_crop / (overlaps+1)).astype(int)
    w_step = np.round(w_crop / (overlaps+1)).astype(int)
    # print(f"h_step: {h_step}, seg_h: {seg_h}, w_step: {w_step}, seg_w: {seg_w}, img_padded: {img_padded.shape}, image[:h, :w]: {image[:h, :w].shape}")

    for ind_i in range(0,seg_h):
        i = ind_i * h_step
        for ind_j in range(0,seg_w):
            j = ind_j * w_step
            img_ = img_padded[i:i+h_crop+padding_pixel*2, j:j+w_crop+padding_pixel*2, :]
            img_processed = processor(img_)
            images.append(img_processed)
            positions.append(torch.FloatTensor([i-padding_pixel, j-padding_pixel, img_.shape[0], img_.shape[1]]).reshape(4))
    return torch.stack(images, dim=0), torch.stack(positions, dim=0), seg_h, seg_w

def read_img(img_path, read_alpha=False):
    img = imageio.imread(img_path)
    img = np.array(img)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    if read_alpha:
        if img.shape[-1] <= 3:
            logger.info(f"No alpha found for: {img_path}")
            img = (img[:, :, :1]) * 0 + 1.0
        else:
            img = (img[:, :, 3:] / 255.0) + 0.0
    else:
        img = img[:, :, :3] / 255.0

    img = torch.from_numpy(img).to(get_rank()).float()
    return img

class InferenceModel():
    def __init__(self, input_dir, mask_dir, output_dir, guidance_dir, 
                 split_overlap, padding, split_hw, inference_padding,
                 ckpt_path, use_ddim, gpu_id=0):
        self.gpu_id = gpu_id
        self.split_overlap = split_overlap
        self.split_hw = split_hw

        self.padding = padding

        # cropped image to diffusion input
        self.diffusion_size = 256
        self.inference_padding = inference_padding

        self.results_list = None
        self.results_output_list = []
        self.image_sizes_list = []

        # load scene
        self.input_dir = input_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.guidance_dir = guidance_dir

        self.all_inputs, self.all_outputs, self.all_guides = self.load_scene(self.input_dir)

        if len(self.all_inputs) == 0:
            return
        
        self.world_size = get_world_size()
        if self.world_size > 1:
            logger.info(f"Total Images: {len(self.all_inputs)}")
            self.all_inputs, self.all_outputs, self.all_guides = map(lambda x:x[self.gpu_id::self.world_size], 
                                                                     (self.all_inputs, self.all_outputs, self.all_guides))
            logger.info(f"Assigned Images: {len(self.all_inputs)} for GPU :{self.gpu_id}")

        self.model = init_model(ckpt_path, use_ddim, gpu_id=gpu_id)

    def _get_padding_idx(self, image_size):
        crop_length = self.diffusion_size - self.inference_padding * 2
        resize_ratio = crop_length / image_size.max()
        crop_hw = torch.round(image_size * resize_ratio).long()
        start_uv = torch.round((self.diffusion_size - crop_hw) / 2).long()
        return start_uv, crop_hw

    def _prepare_image(self, image):
        hw_ori = torch.tensor(image.shape[:2]).long()
        start_uv, crop_hw = self._get_padding_idx(hw_ori)
        out_img = torch.ones(self.diffusion_size, self.diffusion_size, 3)
        img_resizes = Resize((crop_hw[0].item(), crop_hw[1].item()))(image.permute(2,0,1)).permute(1,2,0)
        out_img[start_uv[0]:start_uv[0]+crop_hw[0], start_uv[1]:start_uv[1]+crop_hw[1]] = img_resizes
        return out_img
    
    def _restore_image(self, image, image_size):
        start_uv, crop_hw = self._get_padding_idx(image_size)
        img_cropped = image[start_uv[0]:start_uv[0]+crop_hw[0], start_uv[1]:start_uv[1]+crop_hw[1]]
        img_resizes = Resize((image_size[0].item(), image_size[1].item()))(img_cropped.permute(2,0,1)).permute(1,2,0)
        return img_resizes

    def get_item(self, idx):
        input_name = self.all_inputs[idx]
        output_name = self.all_outputs[idx]
        guid_name = self.all_guides[idx]

        # load images
        input_img = read_img(input_name)
        if self.mask_dir is None:
            img_mask = read_img(input_name, read_alpha=True)
        else:
            img_mask = read_img(input_name.replace(self.input_dir, self.mask_dir))

        if not self.guidance_dir is None:
            guid_images = read_img(guid_name)
        else:
            guid_images = None
        return self.parse_item(input_img, img_mask, guid_images, output_name)
    
    def __len__(self):
        return len(self.all_inputs)
    
    def parse_item(self, img_ori, mask_img_ori, guid_images, output_name):
        # if mask_img_ori is None:
        #     mask_img_ori = read_img(input_name, read_alpha=True)
        #     # ensure background is white, same as training data
        #     img_ori[~(mask_img_ori[..., 0] > 0.5)] = 1
        img_ori[~(mask_img_ori[..., 0] > 0.5)] = 1
        use_true_mask = (self.split_hw[0] * self.split_hw[1]) <= 1 # Flag for mask visulization only

        # mask cropping
        min_max_uv = masks_to_boxes(mask_img_ori[None, ..., -1] > 0.5).long()
        min_uv, max_uv = min_max_uv[0, ..., [1,0]], min_max_uv[0, ..., [3,2]]+1
        # print(self.min_uv, max_uv)

        mask_img = mask_img_ori[min_uv[0]:max_uv[0], min_uv[1]:max_uv[1]]
        img = img_ori[min_uv[0]:max_uv[0], min_uv[1]:max_uv[1]]

        image_size = list(img.shape) + list(img_ori.shape) + [min_uv, max_uv]


        if not use_true_mask:
            mask_img = torch.ones_like(mask_img)
        mask_img, _ = images_spliter(mask_img[..., [0, 0, 0]], self.split_hw[0], self.split_hw[1], self.padding, not use_true_mask, self.split_overlap, processor=self._prepare_image)[:2]

        img, position_indexes, seg_h, seg_w = images_spliter(img, self.split_hw[0], self.split_hw[1], self.padding, 1, self.split_overlap, processor=self._prepare_image)
        self.split_hw_overlapped = [seg_h, seg_w]

        logger.info(f"Spliting Size: {image_size[:2]}, splits: {self.split_hw}, Overlapped: {self.split_hw_overlapped}")

        if guid_images is None:
            guid_images = torch.zeros_like(img)
        else:
            guid_images = guid_images[min_uv[0]:max_uv[0], min_uv[1]:max_uv[1]]
            guid_images, _ = images_spliter(guid_images, self.split_hw[0], self.split_hw[1], self.padding, 1, self.split_overlap, processor=self._prepare_image)[:2]

        return guid_images, img, mask_img[..., :1], image_size, position_indexes, output_name

    def load_scene(self, scene_name):
        logger.info(f"Processing Scene {scene_name}")
        scene_dir = self.input_dir
        output_dir = self.output_dir
        guid_dir = self.guidance_dir if not self.guidance_dir is None else ""
        os.makedirs(output_dir, exist_ok=True)

        rgb_names = os.listdir(scene_dir)
        image_indices_all = [_name.rsplit("/", 1)[-1].rsplit(".")[0] for _name in rgb_names if ".DS_Store" not in _name]
        exr_ori = rgb_names[0].rsplit("/", 1)[-1].rsplit(".")[-1]
        # print(image_indices_all, os.path.join(output_dir, f"{image_indices_all[0]}.{exr}"), output_dir, self.output_dir)

        # filter exisiting images
        image_indices = [image_index for image_index in image_indices_all if not os.path.exists(os.path.join(output_dir, f"{image_index}.{exr_ori}"))]
        
        input_images = [os.path.join(scene_dir, f"{image_index}.{exr_ori}") for image_index in image_indices]
        output_images = [os.path.join(output_dir, f"{image_index}.{exr_ori}") for image_index in image_indices]
    
        guid_images = [os.path.join(guid_dir, f"{image_index}.{exr_ori}") for image_index in image_indices]
        logger.info(f"Total Images to generate {len(guid_images)}")

        return input_images, output_images, guid_images

    def prepare_batch(self, idx, batch_size):
        guid_img = []
        cond_img = []
        mask_img = []
        image_size = []
        position_indexes = []
        output_names = []

        for i in range(batch_size):
            if i+idx >= len(self):
                continue
    
            _guid_ing, _cond_img, _mask_img, _image_size, _position_indexes, _output_name = \
                self.get_item(idx+i)
            guid_img.append(_guid_ing)
            cond_img.append(_cond_img)
            mask_img.append(_mask_img)
            position_indexes.append(_position_indexes)

            image_size += [_image_size] * _guid_ing.shape[0]
            output_names += [_output_name] * _guid_ing.shape[0]

        guid_img, cond_img, mask_img, position_indexes = map(lambda x:torch.cat(x, dim=0).to(self.gpu_id), 
                                           (guid_img, cond_img, mask_img, position_indexes))

        return guid_img, cond_img, mask_img, image_size, position_indexes, output_names

    
    def assemble_results(self, img_out, img_hw=None, position_index=None, default_val=1):
        results_img = np.zeros((img_hw[0], img_hw[1], 3))
        weight_img = np.zeros((img_hw[0], img_hw[1], 3)) + 1e-5

        for i in range(position_index.shape[0]):
            # restore to original crop
            img_restored = self._restore_image(torch.from_numpy(img_out[i]), torch.from_numpy(position_index[i, 2:4]).long()).numpy()
            # crop out boarder
            crop_h, crop_w = img_restored.shape[:2]
            pathed_img = img_restored[self.padding:crop_h-self.padding, self.padding:crop_w-self.padding]
            position_index[i] += self.padding
            crop_h, crop_w = pathed_img.shape[:2]
            crop_x, crop_y = max(position_index[i][0], 0), max(position_index[i][1], 0)
            shape_max = results_img[crop_x:crop_x+crop_h, crop_y:crop_y+crop_w].shape[:2]
            start_crop_x, start_crop_y = abs(min(position_index[i][0], 0)), abs(min(position_index[i][1], 0))
            results_img[crop_x:crop_x+shape_max[0]-start_crop_x, crop_y:crop_y+shape_max[1]-start_crop_y] += pathed_img[start_crop_x:shape_max[0], start_crop_y:shape_max[1]]
            weight_img[crop_x:crop_x+crop_h-start_crop_x, crop_y:crop_y+shape_max[1]-start_crop_y] += 1
        img_out = results_img / weight_img
        img_out[weight_img[:,:,0] < 1] = 255
        img_out_ = (np.zeros((img_hw[3], img_hw[4], 3)) + default_val) * 255
        img_out_[img_hw[6][0]:img_hw[7][0], img_hw[6][1]:img_hw[7][1]] = img_out
        img_out = img_out_
        return img_out

    def write_batch_img(self, imgs, output_paths, image_sizes, position_indexes):
        cropped_batch = self.split_hw_overlapped[0] * self.split_hw_overlapped[1]
        if self.results_list is None or self.results_list.shape[0] == 0:
            self.results_list = imgs
            self.position_indexes = position_indexes
        else:
            self.results_list = torch.cat([self.results_list, imgs], dim=0)
            self.position_indexes = torch.cat([self.position_indexes, position_indexes], dim=0)
        self.image_sizes_list += image_sizes
        self.results_output_list += output_paths

        valid_len = self.results_list.shape[0] - (self.results_list.shape[0] % cropped_batch)
        out_images = []
        for ind in range(0, valid_len, cropped_batch):
            # assemble results
            img_out = (self.results_list[ind:ind+cropped_batch].detach().cpu().numpy() * 255).astype(np.uint8)
            img_out = self.assemble_results(img_out, self.image_sizes_list[ind], self.position_indexes[ind:ind+cropped_batch].detach().cpu().numpy().astype(int))
            Image.fromarray(img_out.astype(np.uint8)).save(self.results_output_list[ind])
            out_images.append(img_out.astype(np.uint8))
        self.results_list = self.results_list[valid_len:]
        self.results_output_list = self.results_output_list[valid_len:]

        self.position_indexes = self.position_indexes[valid_len:]
        self.image_sizes_list = self.image_sizes_list[valid_len:]

        return out_images

    def write_batch_input(self, imgs, image_sizes, position_indexes, default_val=1):
        cropped_batch = self.split_hw_overlapped[0] * self.split_hw_overlapped[1]

        images = []
        valid_len = imgs.shape[0]
        for ind in range(0, valid_len, cropped_batch):
            # assemble results
            img_out = (imgs[ind:ind+cropped_batch].detach().cpu().numpy() * 255).astype(np.uint8)
            img_out = self.assemble_results(img_out, image_sizes[ind], position_indexes.detach().cpu().numpy().astype(int), default_val).astype(np.uint8)
            images.append(img_out)
        return images
            
    def generation(self, dps_scale, uc_score, ddim_steps, batch_size=32, n_samples=1):
        operator = GaussialBlurOperator(61, 3.0, self.gpu_id)
        
        # get img hw
        for src_img_id in range(0, len(self), batch_size):
            guid_img, cond_img, mask_img, image_sizes, position_indexes, output_names = self.prepare_batch(src_img_id, batch_size)

            # Debugging: visualize mask and foreground
            # input_masked = self.write_batch_input(cond_img, image_sizes, position_indexes)
            # input_maskes = self.write_batch_input(mask_img, image_sizes, position_indexes, 0)
            # input_guid = self.write_batch_input(guid_img, image_sizes, position_indexes, 0)

            # Image.fromarray(input_masked[0].astype(np.uint8)).save(output_names[0]+".input.jpg")
            # Image.fromarray(input_maskes[0].astype(np.uint8)).save(output_names[0]+".mask.jpg")
            # Image.fromarray(input_guid[0].astype(np.uint8)).save(output_names[0]+".guid.jpg")
            
            assert n_samples == 1
            for _ in range(n_samples):
                for batch_id in range(0, guid_img.shape[0], batch_size):
                    embeddings = {}
                    embeddings["cond_img"] = cond_img[batch_id:batch_id+batch_size]
                    # results = embeddings["cond_img"]
                    if (mask_img[batch_id:batch_id+batch_size] > 0.5).sum() == 0:
                        results = torch.ones_like(cond_img[batch_id:batch_id+batch_size])
                    else:
                        results = self.model(embeddings, guid_img[batch_id:batch_id+batch_size], mask_img[batch_id:batch_id+batch_size], ddim_steps=ddim_steps,
                                            guidance_scale=uc_score, dps_scale=dps_scale, as_latent=False, grad_scale=1, operator=operator)
                    self.write_batch_img(results, output_names[batch_id:batch_id+batch_size], image_sizes[batch_id:batch_id+batch_size], position_indexes[batch_id:batch_id+batch_size])


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate samples from pre-trained diffusion models')
    parser.add_argument('--input_dir', type=str, default='data/objects', 
                        help="Input image directory")
    parser.add_argument('--mask_dir', type=str, default=None,
                        help="Optional mask directory")
    parser.add_argument('--output_dir', type=str, default='data/object_mask')

    parser.add_argument('--model_dir', type=str, default='None',
                        help="Trained diffusion model directory")
    
    parser.add_argument('--ddim', type=int, default=200,
                        help="DDIM steps")
    parser.add_argument('--batch_size', type=int, default=1,
                    help="Inference batch size")
    
    # Parameters for image padding
    parser.add_argument('--padding', type=int, default=10,
                        help="Boarder padding for patches. Helps to improve the border consistency.")
    parser.add_argument('--inference_padding', type=int, default=0,
                        help="Image padding for diffusion input")
    
    # Parameters for high resolution sample generation
    parser.add_argument('--guidance_dir', type=str, default=None,
                        help="High resolution guidance images directory")
    parser.add_argument('--guidance', type=float, default=0,
                        help="High resolution guidance scale")
    parser.add_argument('--splits_vertical', type=int, default=1)
    parser.add_argument('--splits_horizontal', type=int, default=1)
    parser.add_argument('--splits_overlap', type=int, default=0)

    parser.add_argument('--local-rank', type=int, default=None)

    args = parser.parse_args()
    set_loggers("INFO")

    # For DDP inference
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()
        logger.info(f"Rank {args.local_rank} Initialized")



    model = InferenceModel(input_dir=args.input_dir, mask_dir=args.mask_dir, 
                           output_dir=args.output_dir, guidance_dir=args.guidance_dir,
                           split_overlap=args.splits_overlap, padding=args.padding,
                           inference_padding=args.inference_padding,
                           split_hw=[args.splits_vertical, args.splits_horizontal],
                           ckpt_path=args.model_dir, 
                           use_ddim=True, gpu_id=get_rank())
    
    model.generation(dps_scale=args.guidance, uc_score=1, 
                     ddim_steps=args.ddim, batch_size=args.batch_size, n_samples=1)