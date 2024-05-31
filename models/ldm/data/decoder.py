import sys
sys.path.insert(1, '.')
import numpy as np
from omegaconf import DictConfig
import torch
from PIL import Image
import torchvision
import cv2
import matplotlib.pyplot as plt
from ldm.util import instantiate_from_config
import os
import io
import pickle
import webdataset as wds
import imageio
import time
from torch import distributed as dist
from itertools import chain


class ObjaverseDataDecoder:
    def __init__(self,
        target_name="albedo",
        image_transforms=[],
        default_trans=torch.zeros(3),
        postprocess=None,
        return_paths=False,
        mask_name="alpha",
        test=False,
        condition_name=None,
        bg_color="white",
        target_name_pool=None,
        **kargs
        ) -> None:
        """Create a dataset from blender rendering results.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        # testing behaves differently
        self.test = test
        self.target_name = target_name
        self.mask_name = mask_name
        self.default_trans = default_trans
        self.return_paths = return_paths
        if isinstance(postprocess, DictConfig):
            postprocess = instantiate_from_config(postprocess)
        self.postprocess = postprocess
        # extra condition
        self.condition_name = condition_name
        self.target_name_pool = target_name_pool if not target_name_pool is None else [target_name]
        self.counter = 0

        self.tform = image_transforms["totensor"]
        self.img_size = image_transforms["size"]
        self.tsize = torchvision.transforms.Compose([torchvision.transforms.Resize(self.img_size)])
        if bg_color == "white":
            self.bg_color = [1., 1., 1., 1.] 
        elif bg_color == "noise":
            self.bg_color = "noise"
        else:
            raise NotImplementedError

    def path_parsing(self, filename, cond_name=None):
        # cached path loads albedo
        if 'albedo' in filename:
            filename = filename.replace('albedo', self.target_name)
        if self.target_name=="gloss_shaded":
            filename = filename.replace('gloss_direct', self.target_name).replace("exr", "jpg")
            filename_targets = [filename.replace(self.target_name, "gloss_direct").replace("jpg", "exr"),
                                filename.replace(self.target_name, "gloss_color")]
        elif self.target_name=="diffuse_shaded":
            filename = filename.replace('diffuse_direct', self.target_name).replace("exr", "jpg")
            filename_targets = [filename.replace(self.target_name, "diffuse_direct").replace("jpg", "exr"),
                                filename.replace(self.target_name, "albedo")]
        else:
            filename_targets = None

        normal_condition_filename = None
        if self.test and "images_train" in filename:
            # Currently. "images_train" exists in test set, we write this for clearity
            condition_filename = filename
            mask_filename = filename.replace('images_train', 'masks')
            if self.condition_name == "normal":
                raise NotImplementedError("Testing with normal conditioning on custom data is not supported")
        else:
            cond_name_prefix = filename.split(".", 1)[0] + "." if cond_name is None else cond_name
            condition_filename = cond_name_prefix + filename.rsplit('.', 1)[1]
            mask_filename = filename.replace(self.target_name, self.mask_name)
            if self.condition_name == "normal":
                normal_condition_filename = filename.replace(self.target_name, "normal")

        return filename, condition_filename, mask_filename, normal_condition_filename, filename_targets
    
    def read_images(self, filename, condition_filename, normal_condition_filename):
        # image reading
        if self.target_name in ["gloss_shaded", "diffuse_shaded"]:
            target_im_0 = np.array(self.normalized_read(filename[0]))
            target_im_1 = np.array(self.normalized_read(filename[1]))
            target_im = np.clip(target_im_0 * target_im_1, 0, 1)
        else:
            target_im = np.array(self.normalized_read(filename))

        cond_im = np.array(self.normalized_read(condition_filename))
        
        if self.condition_name == "normal":
            normal_img = np.array(self.normalized_read(normal_condition_filename)) 
        else:
            normal_img = None

        return target_im, cond_im, normal_img
    

    def image_post_processing(self, img_mask, target_im, cond_im, normal_img):
        # make sure image has 3 dimension
        if len(img_mask.shape) == 2:
            img_mask = img_mask[:, :, np.newaxis]
        else:
            img_mask = img_mask[:, :, :3]

        # transform into desired format
        target_im, crop_idx = self.load_im(target_im, img_mask, self.bg_color, crop_idx=True)
        target_im = np.uint8(self.tsize(target_im))
        cond_im = np.uint8(self.tsize(self.load_im(cond_im, img_mask, self.bg_color)))

        if self.condition_name == "normal":
            normal_img = np.uint8(self.tsize(self.load_im(normal_img, img_mask, self.bg_color)))
        else:
            normal_img = None
        return target_im, cond_im, normal_img, crop_idx

    # def cartesian_to_spherical(self, xyz):
    #     ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    #     xy = xyz[:,0]**2 + xyz[:,1]**2
    #     z = np.sqrt(xy + xyz[:,2]**2)
    #     theta = np.arctan2(np.sqrt(xy), xyz[:,2]) # for elevation angle defined from Z-axis down
    #     #ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    #     azimuth = np.arctan2(xyz[:,1], xyz[:,0])
    #     return np.array([theta, azimuth, z])


    def load_im(self, img, img_mask, color, crop_idx=False):
        '''
        replace background pixel with random color in rendering
        '''
        # our rendering do not have a valid alpha channel. 
        # We use a seperate mask, which also do not have a valid alpha
        if img.shape[-1] == 3:
            img = np.concatenate([img, np.ones_like(img[..., :1])], axis=-1)

        # image maske shape align with image size
        if (img.shape[0] != img_mask.shape[0]) or (img.shape[1] != img_mask.shape[1]):
            img_mask = cv2.resize(img_mask,
                                  (img.shape[1], img.shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)[:, :, np.newaxis]    

        if isinstance(color, str):
            random_img = np.random.rand(*(img.shape))
            img[img_mask[:, :, -1] <= 0.5] = random_img[img_mask[:, :, -1] <= 0.5]
        else:
            img[img_mask[:, :, -1] <= 0.5] = color

        if self.test:
            # crop out valid_mask
            img, crop_uv = self.center_crop(img[:, :, :3], img_mask)
        else:
            crop_uv = None

        # center crop
        if img.shape[0] > img.shape[1]:
            margin = int((img.shape[0] - img.shape[1]) // 2)
            img = img[margin:margin+img.shape[1]]
        elif img.shape[1] > img.shape[0]:
            margin = int((img.shape[1] - img.shape[0]) // 2)
            img = img[:, margin:margin+img.shape[0]]    

        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        if crop_idx:
            return img, crop_uv
        return img
    
    def center_crop(self, img, mask, mask_ratio=.8):
        mask_uvs = np.vstack(np.nonzero(mask[:, :, -1] > 0.5))
        min_uv, max_uv = np.min(mask_uvs, axis=-1), np.max(mask_uvs, axis=-1)
        img = img + (mask[..., -1:] <= 0.5)
        
        half_size = int(max(max_uv - min_uv) // 2)
        crop_length = (max_uv - min_uv) // 2
        center_uv = min_uv + crop_length
        expand_hasl_size = int(half_size / mask_ratio)
        size = expand_hasl_size * 2 + 1

        img_new = np.ones((size, size, 3))
        img_new[expand_hasl_size-crop_length[0]:expand_hasl_size+crop_length[0]+1, expand_hasl_size-crop_length[1]:expand_hasl_size+crop_length[1]+1] = \
            img[center_uv[0]-crop_length[0]:center_uv[0]+crop_length[0]+1, center_uv[1]-crop_length[1]:center_uv[1]+crop_length[1]+1]
        crop_uv = np.array([expand_hasl_size, crop_length[0], crop_length[1], center_uv[0], center_uv[1], size], dtype=int)
        return img_new, crop_uv

    def transform_normal(self, normal_input, cam):
        # load camera
        img_mask = torch.linalg.norm(normal_input, dim=-1) > 1.5
        extrinsic, K = cam
        extrinsic = np.concatenate([extrinsic, np.zeros(4).reshape(1, 4)], axis=0)
        extrinsic[3, 3] = 1
        pose = np.linalg.inv(extrinsic)
        temp = pose[1] + 0.0
        pose[1] = -pose[2]
        pose[2] = temp
        extrinsic = torch.from_numpy(np.linalg.inv(pose)).float()

        # to normal
        normal_img = extrinsic[None, :3, :3] @ normal_input[..., :3].reshape(-1, 3, 1)
        normal_img = normal_img.reshape(normal_input.shape[0], normal_input.shape[1], 3)

        normal_img[img_mask] = 1.0
        return normal_img
    
    def parse_item(self, target_im, cond_img, normal_img, filename, target_ids, **args):
        data = {}

        # we need to transform normal to cmaera frame
        if self.target_name == "normal":
            target_im = self.transform_normal(target_im, self.get_camera(filename, **args))
        
        # normal conditioning
        if self.condition_name == "normal":
            normal_img = self.transform_normal(normal_img, self.get_camera(filename, **args))

        data["image_target"] = target_im
        data["image_cond"] = cond_img
        if self.condition_name == "normal":
            data["img_normal"] = normal_img

        if self.test or self.return_paths:
            data["path"] = str(filename)   

        data["label"] = torch.zeros(1).reshape(1, 1, 1)+target_ids   

        if self.postprocess is not None:
            data = self.postprocess(data)
        return data

    def normalized_read(self, imgpath):
        img = np.array(imageio.imread(imgpath))
        if img.dtype == np.uint8:
           img = img / 255.0
        else:
           img = img ** (1 / 2.2)
        return img 
    
    def process_im(self, im):
        im = Image.fromarray(im)
        im = im.convert("RGB")
        return self.tform(im)
    

class ObjaverseDecoerWDS(ObjaverseDataDecoder):
    def __init__(self, **kargs) -> None:
        super().__init__(**kargs)

    def dict2tuple(self, data):
        returns = (data["image_target"], data["image_cond"],data["label"],)
        if self.condition_name == "normal":
            returns +=(data["img_normal"], )
        if self.test or self.return_paths:
            returns += (data["path"],)
        return returns
    
    def tuple2dict(self, data):
        returns = {}
        returns["image_target"] = data[0]
        returns["image_cond"] = data[1]
        returns["label"] = data[2]
        
        if self.condition_name == "normal":
            returns["img_normal"] = data[3]
    
        if self.test or self.return_paths:
            returns["path"] = data[-1]

        return returns

    def data_filter(self, albedo, spec, diffuse_shad, spec_shad):
        returns = {}
        returns["image_target"] = data[0]
        returns["image_cond"] = data[1]
        if self.condition_name == "normal":
            returns["img_normal"] = data[2]
    
        if self.test or self.return_paths:
            returns["path"] = data[-1]

        return returns

    def get_camera(self, input_filename, sample):
        camera_file = input_filename.replace(f'{self.target_name}0001', \
                                             'camera').rsplit(".")[0] + ".pkl"
        mask_filename_byte = io.BytesIO(sample[camera_file])
        cam = pickle.load(mask_filename_byte)
        return cam

    def process_sample(self, sample):
        # start_worker=time.time()
        results = []
        for target_ids, target_name in enumerate(self.target_name_pool):
            _result = self.process_sample_single(sample, target_ids, target_name)
            results.append(self.dict2tuple(_result))
        results = wds.filters.default_collation_fn(results)
        return results

    def batch_reordering(self, sample):
        batch_splits = []
        for data_idx, _ in enumerate(sample):
            batch_splits.append(
                torch.cat(
                    torch.chunk(sample[data_idx], dim=1, 
                                chunks=len(self.target_name_pool)),
                    dim=0)[:,0]
            )
        return self.tuple2dict(batch_splits)

    def process_sample_single(self, sample, target_ids, target_name):

        # get target image filename
        self.target_name = target_name
        target_file_name = self.target_name
        if self.target_name=="gloss_shaded":
            target_file_name = "gloss_direct"
        elif self.target_name=="diffuse_shaded":
            target_file_name = "diffuse_direct"

        for k in list(sample.keys()):
            if target_file_name not in k:
                continue
            target_key = k
            break

        # ##############
        # prev_time = start_worker
        # current_time = time.time()
        # print(f"find target takes: {current_time - prev_time}")
        # ##############
        
        filename, condition_filename, \
            mask_filename, normal_condition_filename, filename_targets = self.path_parsing(target_key, "")

        # get file streams
        if filename_targets is None:
            filename_byte = io.BytesIO(sample[filename])
        else:
            filename_byte = [io.BytesIO(sample[filename_target]) for filename_target in filename_targets]
        condition_filename_byte = io.BytesIO(sample[condition_filename])
        normal_condition_filename_byte = io.BytesIO(sample[normal_condition_filename]) \
            if self.condition_name == "normal" else None
        mask_filename_byte = io.BytesIO(sample[mask_filename])

        # image reading
        target_im, cond_im, normal_img = self.read_images(filename_byte, 
                                                         condition_filename_byte, normal_condition_filename_byte)

        # mask reading
        img_mask = np.array(self.normalized_read(mask_filename_byte))

        # post processing
        target_im, cond_im, normal_img, _ = self.image_post_processing(img_mask, target_im, cond_im, normal_img)

        # transform
        target_im = self.process_im(target_im)
        cond_im = self.process_im(cond_im)
        normal_img = self.process_im(normal_img) \
            if self.condition_name == "normal" \
                else None

        data = self.parse_item(target_im, cond_im, normal_img, filename, target_ids, sample=sample)
        # override for file path
        if self.test or self.return_paths:
            data["path"] = sample["__key__"]

        result = dict(__key__=sample["__key__"])
        result.update(data)
        return result


if __name__=="__main__":
    from torchvision import transforms
    from einops import rearrange
    torch.distributed.init_process_group(backend="nccl")
    image_transforms = [transforms.ToTensor(),
                            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))]
    image_transforms = torchvision.transforms.Compose(image_transforms)
    image_transforms = {
        "size": 256,
        "totensor": image_transforms
    }
    
    data_list_dir = "/home/chenxi/code/material-diffusion/data/big_data_lists"
    tar_name_list = sorted(os.listdir(data_list_dir))[1:4]
    tar_list = [_name.rsplit("_num")[0]+".tar" for _name in tar_name_list]
    tar_dir = "/home/chenxi/code/material-diffusion/data/big_data_transed"
    tars = [os.path.join(tar_dir, _name) for _name in tar_list]
    dataset_size = 0
    imgperobj = 10
    print("list dirs...")
    for _name in tar_name_list:
        num_obj = int(_name.rsplit("_num_")[1].rsplit(".")[0])
        print(num_obj, " : ", _name)
        dataset_size += num_obj * imgperobj

    decoder = ObjaverseDecoerWDS(image_transforms=image_transforms, 
                                 return_paths=True)
    batch_size = 8

    print('============= length of training dataset %d =============' % (dataset_size // batch_size // 2))
    dataset = (wds.WebDataset(tars,
                                repeat=0,
                                nodesplitter=wds.shardlists.split_by_node)
                .shuffle(100)
                .map(decoder.process_sample)
                .map(decoder.dict2tuple)
                .batched(batch_size, partial=False)
                .map(decoder.tuple2dict)
                .with_epoch(dataset_size // batch_size // 2)
                .with_length(dataset_size // batch_size)
    )
    from torch.utils.data import DataLoader
    # loader = DataLoader(dataset, batch_size=None, num_workers=8, shuffle=False)
    loader = (wds.WebLoader(dataset, batch_size=None, num_workers=2, shuffle=False)
            .map(decoder.dict2tuple)
            .unbatched()
            # .shuffle(100)
            .batched(batch_size)
            .map(decoder.tuple2dict)
    )


    print("# loader length", len(dataset))
    
    for epoch in range(2):
        ind = -1
        for sample in loader:
            assert "image_target" in sample
            assert "image_cond" in sample
            assert "path" in sample
            ind += 1
            if ind != 0:
                continue

            # replace to this for file path
            # worker_info = torch.utils.data.get_worker_info()
            # if worker_info is not None:
            #     worker = worker_info.id
            #     num_workers = worker_info.num_workers
            # data["path"] = sample["__url__"]+"--"+sample["__key__"] +f".{worker}/{num_workers}"

            # print(f"{ind}: shape {sample['image_target'].shape} {sample['path'][0].rsplit('/', 1)[-2]}")
            print("##############")
            for i in range(len(sample['path'])):
                print(f"epoch {epoch}, it {ind}: shape {sample['image_target'].shape} {sample['path'][i].rsplit('--', 1)[0].rsplit('/', 2)[-1]} {sample['path'][i].rsplit('--', 1)[1].rsplit('/', 3)[-3]} {sample['path'][i].rsplit('--', 1)[1].rsplit('/',4)[-4]} {sample['path'][i].rsplit('.', 1)[-1]} rank: {dist.get_rank()}")
            print("##############")


        print(sample["path"])
            
        print(sample["path"])

    print(f"NUmber of samples: {ind} {dataset_size} {len(dataset)} rank: {dist.get_rank()}")
    # 1.  Remember samples are batched inside each worker, the outside data loader only sees one sample
    # 2.  All batch, epoch, and length settings are only visible within each worker
    # 3.  Unbatch and Suffle and then re-batch in loader result in between worker shuffle.
    #     This also allows to control of loader batching and worker batching for CPU optimization of worker-loader data transfer.
    #     https://github.com/webdataset/webdataset/issues/141#issuecomment-1043190147
    # 4.  It seems that data just repeat forever to satisfy with_epoch 
    # 5.  Torch datalogger requires the dataset to have a len() method, which is used to schdule sample idx
    # 6.  DDP sampler will return its only length
    # 7.  WebLoader does not need length, it only raises the end of the iteration when data is running out
    # 8.  How does torch loader deal with datasets with fewer sizes than claims?
    # 9.  Set epoch will make sampling start from the beginning when a new epoch starts. Observed by disable shuffle and one batch repeat
    #     And each epoch will have a different sampling seed
    # 10. DataLoader with IterableDataset: expected unspecified sampler option. DDP sampler will not be usable.
    # !0. In summary:
    # For ddp multi-worker training, the worker splitter and node splitter will make sure tars are splitted into each worker
    # We have to manually adjust with_epoch with respect to num_worker and num_node and batch_size

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src