from tqdm import tqdm
import os
import objaverse
import random
import numpy as np
def get_rendered_objaverse_list_v0(data_dir, target_name, exr, **kargs):
    "This function is to fast obtain unfinined objaverse rendering images"
    image_list_cache_path = kargs["image_list_cache_path"]
    if os.path.exists(image_list_cache_path):
        return np.load(image_list_cache_path)["image_list"].tolist()
    random.seed(7564)
    uids = objaverse.load_uids()
    random.shuffle(uids)
    
    obj_starts = kargs["obj_starts"]
    obj_ends = kargs["obj_ends"]
    num_envs = kargs["num_envs"]
    num_imgs = kargs["num_imgs"]


    selected_uids = []
    for _start, _end in zip(obj_starts, obj_ends):
        selected_uids += uids[_start:_end]
    
    all_imgs = []

    envpaths_all = os.listdir(os.path.join(data_dir, selected_uids[0]))
    envpaths_raw = [_env for _env in envpaths_all if not ".txt" in _env]

    for _uid in tqdm(selected_uids):
        random.shuffle(envpaths_raw)
        envpaths = envpaths_raw[:num_envs]
        if not os.path.exists(os.path.join(data_dir, _uid)):
            print(f"WARNING NONE EXIST OBJECT {os.path.join(data_dir, _uid)}")
            continue
        for _env in envpaths:
            if not os.path.exists(os.path.join(data_dir, _uid, _env)):
                print(f"WARNING NONE EXIST ENV {os.path.join(data_dir, _uid, _env)}")
                continue
            img_ids = list(range(int(len(os.listdir(os.path.join(data_dir, _uid, _env))) // 7)))
            random.shuffle(img_ids)
            img_ids = img_ids[:num_imgs]

            for _img_ids in img_ids: 
                if not os.path.exists(os.path.join(data_dir, _uid, _env, f"{_img_ids}-{target_name}0001.{exr}")):
                    print(f"WARNING NONE EXIST IMAGE {os.path.join(data_dir, _uid, _env, f'{_img_ids}-{target_name}0001.{exr}')}")
                    continue
                all_imgs += [os.path.join(data_dir, _uid, _env, f"{_img_ids}-{target_name}0001.{exr}")]

    np.savez(image_list_cache_path, image_list=all_imgs)
    return all_imgs     
    
if __name__ == "__main__":
    all_imgs = get_rendered_objaverse_list_v0("/home/chenxi/code/material-diffusion/data/objaverse_rendering/samll-dataset", "albedo", "png", 
    obj_starts=[20], obj_ends=[80], num_envs=100, num_imgs=1)

    print(len(all_imgs), all_imgs[:10])
    for img in all_imgs[:10]:
        print(img, os.path.exists(img))