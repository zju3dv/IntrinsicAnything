# IntrinsicAnything: Learning Diffusion Priors for Inverse Rendering Under Unknown Illumination

### [Project Page](https://zju3dv.github.io/IntrinsicAnything/) | [Paper](https://arxiv.org/abs/2404.11593) |  [Supplementary](https://drive.google.com/file/d/1vvavfbiiR_Tfqe3QmKFCK1JMAU45qQ6F/view?usp=sharing) |  [Hugging Face](https://huggingface.co/spaces/LittleFrog/IntrinsicAnything)

<br/>

> **IntrinsicAnything: Learning Diffusion Priors for Inverse Rendering Under Unknown Illumination** <br>
> [Xi Chen](https://github.com/Burningdust21), [Sida Peng](https://pengsida.net/), [Dongchen Yang](https://dongchen-yang.github.io/), [Yuan Liu](https://liuyuan-pal.github.io/), [Bowen Pan](https://o0helloworld0o-ustc.github.io/), [Chengfei Lv](https://www.mnn.zone/m/0.3/), [Xiaowei Zhou](https://xzhou.me)<br>

<img src="./assets/pipeline.png"/>

## News
- 2024-5-22: 🤗 Live demo released at https://huggingface.co/spaces/LittleFrog/IntrinsicAnything.
- 2024-5-31: Code release for single-view inference.

## Results

### Intrinsic Decomposition
<img src="./assets/intrinsic.png"/>

### Single view Relighting

https://github.com/zju3dv/IntrinsicAnything/assets/62891073/2e95855c-6d72-4bcb-8c79-577e55e6c926


### Single view inference
1. Installation
```shell
conda create -n anyintrinsic python=3.10
conda activate anyintrinsic
pip install -r requirements.txt
```

2. Download the pre-trained diffusion models from [hugghing face](https://huggingface.co/spaces/LittleFrog/IntrinsicAnything) as follow:
```shell
# albedo checkpoint
huggingface-cli download --repo-type space --cache-dir  weights/albedo/checkpoints/ LittleFrog/IntrinsicAnything weights/albedo/checkpoints/last.ckpt 

# specular shaing checkpoint
huggingface-cli download --repo-type space --cache-dir  weights/specular/checkpoints/ LittleFrog/IntrinsicAnything weights/specular/checkpoints/last.ckpt 
```

3. Run inference to get intrinsic images:
```shell
python inference.py \
--input_dir  examples  \
--model_dir  weights/albedo \
--output_dir out/albedo \
--ddim 100 \
--batch_size 4
```

Parameter explanation:
-  `--input_dir`: Path to the folder containing test images. The foreground object mask can be supplied as either RGBA input or specify a mask directory through `--mask_dir`. The mask can be obtained interactively using tools like [Segment-Anything](https://segment-anything.com/). We also provide a script `generate_masks.py` to generate masks automatically based on [Rembg](https://github.com/danielgatis/rembg). (Note: Scene-level images can be directly referenced without masks)

- `--model_dir`: Path to the pre-trained checkpoint. Set `weights/albedo`
 for the albedo generation and `weights/specular` for the specular shading generation.
- `--output_dir`: Output path for the generated samples.
- `--batch_size`: Inference batch size. 

 4. Optionally, you can generate high-resolution samples under the guidance of samples from step 3.
```shell
python inference.py \
--input_dir examples  \
--model_dir weights/albedo \
--output_dir out/albedo_high_res \
--ddim 200 \
--batch_size 4 \
--guidance_dir out/albedo \
--guidance 3  \
--splits_vertical 2 \
--splits_horizontal 2 \
--splits_overlap 1
```

Extra parameter explanation:
-  `--guidance_dir`: Path to the low resolution output (step 3). 
-  `--guidance`: Guidance scale of low-resolution images. We impartially found a scale from 2 to 5 generally yields good results.
- `--splits_vertical` & `--splits_horizontal`: The number of splits in the vertical and horizontal direction. 
- `--splits_overlap`:  The number of overlaps evenly distributed between two adjacent patches. The final value of each pixel is the average of all overlapped patches to improve the patches' consistency.

### Multiview Inverse Rendering
Comming Soon.


## Citation

```bibtex
@article{chen2024intrinsicanything,
    title     = {IntrinsicAnything: Learning Diffusion Priors for Inverse Rendering Under Unknown Illumination},
    author    = {Xi, Chen and Sida, Peng and Dongchen, Yang and Yuan, Liu and Bowen, Pan and Chengfei, Lv and Xiaowei, Zhou.},
    journal   = {arxiv: 2404.11593},
    year      = {2024},
    }
```