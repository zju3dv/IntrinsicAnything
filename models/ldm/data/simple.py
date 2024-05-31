
import sys
sys.path.insert(1, '.')
from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import glob
import pickle
from ldm.data.objaverse_rendered import get_rendered_objaverse_list_v0
from ldm.data.decoder import ObjaverseDataDecoder, ObjaverseDecoerWDS, nodesplitter

from loguru import logger
from torch import distributed as dist
from tqdm import tqdm
from multiprocessing.pool import ThreadPool


# Some hacky things to make experimentation easier
def make_transform_multi_folder_data(paths, caption_files=None, **kwargs):
    ds = make_multi_folder_data(paths, caption_files, **kwargs)
    return TransformDataset(ds)

def make_nfp_data(base_path):
    dirs = list(Path(base_path).glob("*/"))
    print(f"Found {len(dirs)} folders")
    print(dirs)
    tforms = [transforms.Resize(512), transforms.CenterCrop(512)]
    datasets = [NfpDataset(x, image_transforms=copy.copy(tforms), default_caption="A view from a train window") for x in dirs]
    return torch.utils.data.ConcatDataset(datasets)


class VideoDataset(Dataset):
    def __init__(self, root_dir, image_transforms, caption_file, offset=8, n=2):
        self.root_dir = Path(root_dir)
        self.caption_file = caption_file
        self.n = n
        ext = "mp4"
        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.offset = offset

        if isinstance(image_transforms, ListConfig):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms
        with open(self.caption_file) as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        self.captions = dict(rows)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for i in range(10):
            try:
                return self._load_sample(index)
            except Exception:
                # Not really good enough but...
                print("uh oh")

    def _load_sample(self, index):
        n = self.n
        filename = self.paths[index]
        min_frame = 2*self.offset + 2
        vid = cv2.VideoCapture(str(filename))
        max_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        curr_frame_n = random.randint(min_frame, max_frames)
        vid.set(cv2.CAP_PROP_POS_FRAMES,curr_frame_n)
        _, curr_frame = vid.read()

        prev_frames = []
        for i in range(n):
            prev_frame_n = curr_frame_n - (i+1)*self.offset
            vid.set(cv2.CAP_PROP_POS_FRAMES,prev_frame_n)
            _, prev_frame = vid.read()
            prev_frame = self.tform(Image.fromarray(prev_frame[...,::-1]))
            prev_frames.append(prev_frame)

        vid.release()
        caption = self.captions[filename.name]
        data = {
            "image": self.tform(Image.fromarray(curr_frame[...,::-1])),
            "prev": torch.cat(prev_frames, dim=-1),
            "txt": caption
        }
        return data

# end hacky things


def make_tranforms(image_transforms):
    # if isinstance(image_transforms, ListConfig):
    #     image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms = []
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    image_transforms = transforms.Compose(image_transforms)
    return image_transforms


def make_multi_folder_data(paths, caption_files=None, **kwargs):
    """Make a concat dataset from multiple folders
    Don't suport captions yet

    If paths is a list, that's ok, if it's a Dict interpret it as:
    k=folder v=n_times to repeat that
    """
    list_of_paths = []
    if isinstance(paths, (Dict, DictConfig)):
        assert caption_files is None, \
            "Caption files not yet supported for repeats"
        for folder_path, repeats in paths.items():
            list_of_paths.extend([folder_path]*repeats)
        paths = list_of_paths

    if caption_files is not None:
        datasets = [FolderData(p, caption_file=c, **kwargs) for (p, c) in zip(paths, caption_files)]
    else:
        datasets = [FolderData(p, **kwargs) for p in paths]
    return torch.utils.data.ConcatDataset(datasets)



class NfpDataset(Dataset):
    def __init__(self,
        root_dir,
        image_transforms=[],
        ext="jpg",
        default_caption="",
        ) -> None:
        """assume sequential frames and a deterministic transform"""

        self.root_dir = Path(root_dir)
        self.default_caption = default_caption

        self.paths = sorted(list(self.root_dir.rglob(f"*.{ext}")))
        self.tform = make_tranforms(image_transforms)

    def __len__(self):
        return len(self.paths) - 1


    def __getitem__(self, index):
        prev = self.paths[index]
        curr = self.paths[index+1]
        data = {}
        data["image"] = self._load_im(curr)
        data["prev"] = self._load_im(prev)
        data["txt"] = self.default_caption
        return data

    def _load_im(self, filename):
        im = Image.open(filename).convert("RGB")
        return self.tform(im)

class ObjaverseDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, root_dir, batch_size, train=None, validation=None,
                 test=None, num_workers=4, objaverse_data_list=None, ext="png", 
                 target_name="albedo", use_wds=True, tar_config=None, **kwargs):
        super().__init__(self)
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs
        self.tar_config = tar_config
        self.use_wds = use_wds

        if train is not None:
            dataset_config = train
        if validation is not None:
            dataset_config = validation


        image_transforms = [transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))]
        image_transforms = torchvision.transforms.Compose(image_transforms)
        self.image_transforms = {
            "size": dataset_config.image_transforms.size,
            "totensor": image_transforms
        }

        self.target_name = target_name
        self.objaverse_data_list = objaverse_data_list
        self.ext = ext

    def naive_setup(self):
        # get object data list
        if self.objaverse_data_list is None or \
                          self.objaverse_data_list["image_list_cache_path"] == "None":
            # This is too slow..
            self.paths = sorted(list(Path(self.root_dir).rglob(f"*{self.target_name}*.{self.ext}")))
            if len(self.paths) == 0:
                # colmap format
                self.paths = sorted(list(Path(self.root_dir).rglob(f"*images_train/*.*")))
        else:      
            self.paths = get_rendered_objaverse_list_v0(self.root_dir, self.target_name, self.ext, **self.objaverse_data_list)
        random.shuffle(self.paths)
        # train val split
        total_objects = len(self.paths)
        self.paths_val = self.paths[math.floor(total_objects / 100. * 99.):] # used last 1% as validation
        self.paths_train = self.paths[:math.floor(total_objects / 100. * 99.)] # used first 99% as training
        if self.rank == 0:
            print('============= length of dataset %d =============' % len(self.paths))
            print('============= length of training dataset %d =============' % len(self.paths_train))
            print('============= length of Validation dataset %d =============' % len(self.paths_val))

        # Split into each GPU
        self.paths_train = self._get_local_split(self.paths_train, self.world_size, self.rank)
        logger.info(
            f"[rank {self.rank}]: {len(self.paths_train)} images assigned."
        )

    def _get_tar_length(self, tar_list, img_per_obj):
        dataset_size = 0
        for _name in tar_list:
            num_obj = int(_name.rsplit("_num_")[1].rsplit(".")[0])
            dataset_size += num_obj * img_per_obj
        return dataset_size

    def webdataset_setup(self, list_dir, tar_dir, img_per_obj, max_tars=None):
        # read data list and calculate size
        tar_name_list = sorted(os.listdir(list_dir))
        if not max_tars is None:
            # for debugging on small scale data
            tar_name_list = tar_name_list[:max_tars]
        total_tars = len(tar_name_list)
        # random shuffle
        random.shuffle(tar_name_list)
        print(f"Rank {self.rank} shuffle: {tar_name_list}")
        # train test split
        self.test_tars = tar_name_list[math.floor(total_tars / 100. * 99.):]
        # make sure each node has one tar
        if len(self.test_tars) < self.world_size:
            self.test_tars += [self.test_tars[0]]*(self.world_size-len(self.test_tars))

        self.train_tars = tar_name_list[:math.floor(total_tars / 100. * 99.)]

        # training tar truncation
        total_workers = self.num_workers * self.world_size
        num_tars_train = (len(self.train_tars) // total_workers) * total_workers
        if num_tars_train != len(self.train_tars):
            print(f"[WARNING] Total train tars: {len(self.train_tars)}, truncated: {len(self.train_tars)-num_tars_train}, remainnig: {num_tars_train}, total workers: {total_workers}")

        self.test_length = self._get_tar_length(self.test_tars, img_per_obj)
        self.train_length = self._get_tar_length(self.train_tars, img_per_obj)

        # name replace
        test_tars = [_name.rsplit("_num")[0]+".tar" for _name in self.test_tars]
        self.test_tars = [os.path.join(tar_dir, _name) for _name in test_tars]

        train_tars = [_name.rsplit("_num")[0]+".tar" for _name in self.train_tars]
        self.train_tars = [os.path.join(tar_dir, _name) for _name in train_tars]

        if self.rank == 0:
            print('============= length of dataset %d =============' % (self.test_length+self.train_length))
            print('============= length of training dataset %d =============' % (self.train_length))
            print('============= length of Validation dataset %d =============' % (self.test_length))

    def setup(self, stage=None):
        try:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        except:
            self.world_size = 1
            self.rank = 0
    
        if self.rank == 0:
            print("#### Data ####")
        
        if self.use_wds:
            self.webdataset_setup(**self.tar_config)
        else:
            self.naive_setup()

    def _get_local_split(self, items: list, world_size: int, rank: int, seed: int = 6):
        """The local rank only loads a split of the dataset."""
        n_items = len(items)
        items_permute = np.random.RandomState(seed).permutation(items)
        if n_items % world_size == 0:
            padded_items = items_permute
        else:
            padding = np.random.RandomState(seed).choice(
                items, world_size - (n_items % world_size), replace=True
            )
            padded_items = np.concatenate([items_permute, padding])
            assert (
                len(padded_items) % world_size == 0
            ), f"len(padded_items): {len(padded_items)}; world_size: {world_size}; len(padding): {len(padding)}"
        n_per_rank = len(padded_items) // world_size
        local_items = padded_items[n_per_rank * rank : n_per_rank * (rank + 1)]

        return local_items

    def train_dataloader(self):
        if self.use_wds:
            loader = self.train_dataloader_wds()
        else:
            loader = self.train_dataloader_naive()
        return loader

    def val_dataloader(self):
        if self.use_wds:
            loader = self.val_dataloader_wds()
        else:
            loader = self.val_dataloader_naive()
        return loader
        
    def train_dataloader_naive(self):
        dataset = ObjaverseData(root_dir=self.root_dir, \
                                image_transforms=self.image_transforms,
                                image_list = self.paths_train, target_name=self.target_name,
                                **self.kwargs)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader_naive(self):
        dataset = ObjaverseData(root_dir=self.root_dir, \
                                image_transforms=self.image_transforms,
                                image_list = self.paths_val, target_name=self.target_name,
                                **self.kwargs)
        return wds.WebLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
    

    def train_dataloader_wds(self):
        decoder = ObjaverseDecoerWDS(root_dir=self.root_dir, \
                                image_transforms=self.image_transforms,
                                image_list = None, target_name=self.target_name,
                                **self.kwargs)

        worker_batch = self.batch_size
        epoch_length = self.train_length // worker_batch // self.num_workers // self.world_size
        dataset = (wds.WebDataset(self.train_tars,
                                  shardshuffle=min(1000, len(self.train_tars)),
                                  nodesplitter=wds.shardlists.split_by_node)
                    .shuffle(5000, initial=1000)
                    .map(decoder.process_sample)
                    # .map(decoder.dict2tuple)
                    .batched(worker_batch, partial=False)
                    # .map(decoder.tuple2dict)
                    .map(decoder.batch_reordering)
                    .with_epoch(epoch_length)
                    .with_length(epoch_length)
        )
        loader = (wds.WebLoader(dataset, batch_size=None, num_workers=self.num_workers, shuffle=False)
            # .unbatched()
            # .shuffle(1000)
            # .batched(self.batch_size)
            # .map(decoder.tuple2dict)
        )

        print(f"# Training loader length for single worker {epoch_length} with {self.num_workers} workers")

        return loader

    def val_dataloader_wds(self):
        decoder = ObjaverseDecoerWDS(root_dir=self.root_dir, \
                                image_transforms=self.image_transforms,
                                image_list = None, target_name=self.target_name,
                                **self.kwargs)

        # adjust worker number, as test has much much fewer tars
        val_workers = min(self.num_workers, len(self.test_tars) // self.world_size)
        epoch_length = max(self.test_length // self.batch_size // val_workers // self.world_size, 1)
        dataset = (wds.WebDataset(self.test_tars,
                                  shardshuffle=min(1000, len(self.test_tars)),
                                  handler=wds.ignore_and_continue,
                                  nodesplitter=wds.shardlists.split_by_node)
                    .shuffle(1000)
                    .map(decoder.process_sample)
                    # .map(decoder.dict2tuple)
                    .batched(self.batch_size, partial=False)
                    .with_epoch(epoch_length)
                    .with_length(epoch_length)
        )
        loader = (wds.WebLoader(dataset, batch_size=None, num_workers=val_workers, shuffle=False)
            .unbatched()
            .shuffle(1000)
            .batched(self.batch_size)
            # .map(decoder.tuple2dict)
            .map(decoder.batch_reordering)
            )

        print(f"# Validation loader length for single worker {epoch_length} with {val_workers} workers")

        return loader
        
    def test_dataloader(self):
        # testing will use all given data
        return wds.WebLoader(ObjaverseData(root_dir=self.root_dir, test=True, 
                                           image_transforms=self.image_transforms,
                                           image_list = self.paths, target_name=self.target_name,
                                           **self.kwargs),
                          batch_size=32, num_workers=self.num_workers, shuffle=False,
                          )


class ObjaverseData(ObjaverseDataDecoder, Dataset):
    def __init__(self,
        root_dir='.objaverse/hf-objaverse-v1/views',
        image_list=None,
        threads=64,
        **kargs
        ) -> None:
        """Create a dataset from blender rendering results.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.paths = image_list
        self.root_dir = Path(root_dir)
        ObjaverseDataDecoder.__init__(self, **kargs)
        # pre-load data
        print(f"Data pre loading start with {threads}...")
        self.all_target_im = np.zeros((len(self.paths), self.img_size, self.img_size, 3), dtype=np.uint8) + 0
        self.all_cond_im = np.zeros((len(self.paths), self.img_size, self.img_size, 3), dtype=np.uint8) + 0
        self.all_filename = ["empty"] * len(self.paths)
        if self.condition_name == "normal":
            self.all_normal_img = np.zeros((len(self.paths), self.img_size, self.img_size, 3), dtype=np.uint8) + 0
        self.all_crop_idx = np.zeros((len(self.paths), 6), dtype=int) + 0

        print("Array allocated..")

        def parallel_load(index):
            pbar.update(1)
            self.preload_item(index)

        pbar = tqdm(total=len(self.paths))
        with ThreadPool(threads) as pool:
            pool.map(parallel_load, range(len(self.paths)))
            pool.close()
            pool.join()

        print("Data pre loading done...")

    def __len__(self):
        return len(self.paths)
        
    def load_mask(self, mask_filename, cond_im):
        # auto image file extention
        glob_files = glob.glob(mask_filename.rsplit(".", 1)[0] + ".*")
        if len(glob_files) == 0:
            print("Warning: no mask image find")
            img_mask = np.ones_like(cond_im)

            if cond_im.shape[-1] == 4:
                print("Use image mask")
                img_mask = img_mask * cond_im[:, :, -1:]
        elif len(glob_files) == 1:
            img_mask = np.array(self.normalized_read(glob_files[0]))
        else:
            raise NotImplementedError("Too many mask images found! {}")
        return img_mask
    
    def preload_item(self, index):
        path = self.paths[index]
        filename = os.path.join(path)
        filename, condition_filename, \
            mask_filename, normal_condition_filename, filename_targets = self.path_parsing(filename)

        # get file streams
        if filename_targets is None:
            filename_read = filename
        else:
            filename_read = filename_targets

        # image reading
        target_im, cond_im, normal_img = self.read_images(filename_read, 
                                                         condition_filename, normal_condition_filename)

        # mask reading
        img_mask = self.load_mask(mask_filename, cond_im)

        # post processing
        target_im, cond_im, normal_img, crop_idx = self.image_post_processing(img_mask, target_im, cond_im, normal_img)
        if self.test:
            # crop out valid_mask
            self.all_crop_idx[index] = crop_idx

        # put results
        self.all_target_im[index] = target_im
        self.all_cond_im[index] = cond_im
        self.all_filename[index] = filename
        if self.condition_name == "normal":
            self.all_normal_img[index] = normal_img

    def get_camera(self, input_filename):
        camera_file = input_filename.replace(f'{self.target_name}0001', \
                                             'camera').rsplit(".")[0] + ".pkl"
        cam_dir, cam_name = camera_file.rsplit("/", 1)
        cam_name = f"{cam_name:>15}"
        camera_file = os.path.join(cam_dir, cam_name)
        cam = pickle.load(open(camera_file, 'rb'))
        return cam
    
    
    def __getitem__(self, index):
        target_im = self.process_im(self.all_target_im[index])
        cond_img = self.process_im(self.all_cond_im[index])
        filename = self.all_filename[index]
        normal_img = self.process_im(self.all_normal_img[index]) \
            if self.condition_name == "normal" \
                else None

        sample = self.parse_item(target_im, cond_img, normal_img, filename)
        if self.test:
            sample["crop_idx"] = self.all_crop_idx[index]
        return sample


if __name__ == "__main__":
    import pyhocon

    class DictAsMember(dict):
        def __getattr__(self, name):
            value = self[name]
            if isinstance(value, dict):
                value = DictAsMember(value)
            return value
        
    def ConfigAsMember(config):
        config_dict = DictAsMember(config)
        for key in config_dict.keys():
            if isinstance(config_dict[key], pyhocon.config_tree.ConfigTree):
                config_dict[key] = ConfigAsMember(config_dict[key])
        return config_dict

    train_config = DictAsMember({
        "validation": False,
        "image_transforms": {"size": 256}
        })
    val_config = DictAsMember({
        "validation": True,
        "image_transforms": {"size": 256}
        })
    objaverse_data_list = DictAsMember({
        "image_list_cache_path": "image_lists/half_400000_image_list.npz",
        })
    data_module = ObjaverseDataModuleFromConfig(root_dir='/mnt/volumes/perception/hujunkang/codes/renders/material-diffusion/data/objaverse_rendering', 
    batch_size=4, train=train_config, validation=val_config,
                 test=None, num_workers=1, objaverse_data_list=objaverse_data_list, ext="png", 
                 target_name="albedo", use_wds=False, tar_config=None)

    data_module.setup()
    train_dataloader_naive = data_module.train_dataloader_naive()