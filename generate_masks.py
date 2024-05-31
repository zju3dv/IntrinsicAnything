from pathlib import Path
from rembg import remove, new_session
import os
from tqdm import tqdm
import cv2
import numpy as np
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate samples from pre-trained diffusion models')
    parser.add_argument('--input_dir', type=str, default='data/objects', 
                        help="Input image directory")
    parser.add_argument('--output_dir', type=str, default='data/object_mask')
    args = parser.parse_args()


    session = new_session()
    os.makedirs(args.output_dir, exist_ok=True)
    img_names = sorted([_name for _name in os.listdir(args.input_dir)])

    for name in tqdm(img_names):
        input_path = os.path.join(args.input_dir, name)
        print(input_path)
        out_path = os.path.join(args.output_dir, name.rsplit(".", 1)[0] + ".png")

        image = cv2.imread(input_path)
        output = remove(image, session=session)    
        cv2.imwrite(out_path, output)