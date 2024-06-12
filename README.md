## Building Bridges across Spatial and Temporal Resolutions: Reference-Based Super-Resolution via Change Priors and Conditional Diffusion Model (Ref-Diff) <br><sub>Official Pytorch implementation of the CVPR 2024 paper

Abstract: Reference-based super-resolution (RefSR) has the potential to build bridges across spatial and temporal resolutions of remote sensing images. However, existing RefSR methods are limited by the faithfulness of content reconstruction and the effectiveness of texture transfer in large scaling factors. Conditional diffusion models have opened up new opportunities for generating realistic high-resolution images, but effectively utilizing reference images within these models remains an area for further exploration. Furthermore, content fidelity is difficult to guarantee in areas without relevant reference information. To solve these issues, we propose a change-aware diffusion model named
Ref-Diff for RefSR, using the land cover change priors to guide the denoising process explicitly. Specifically, we inject the priors into the denoising model to improve the utilization of reference information in unchanged areas and regulate the reconstruction of semantically relevant content in changed areas. With this powerful guidance, we decouple the semantics-guided denoising and reference textureguided denoising processes to improve the model performance. Extensive experiments demonstrate the superior effectiveness and robustness of the proposed method compared with state-of-the-art RefSR methods in both quantitative and qualitative evaluations.

https://arxiv.org/abs/2403.17460

## Setup

### Environment

* 64-bit Python 3.8 and PyTorch 1.12.0 (or later).
* Download the repo and setup the environment with:

```bash
git clone https://github.com/dongrunmin/RefDiff.git
cd RefDiff
conda env create -f environment.yml
conda activate refdiff
```

Linux servers with Nvidia A100s are recommended. However, by setting smaller `--batch-gpu` (batch size on a single gpu), you can still run the inference and training scripts on less powerful GPUs.

### Dataset

We reconstructed SECOND and CNAM-CD datasets for Ref-Diff training and test. The reconstructed SECOND and CNAM-CD datasets can be downloaded from [baidu pan](https://pan.baidu.com/s/1XU4EuyOTWUtTJFLg9TYvIw), password:x4n8, and [google drive](https://drive.google.com/file/d/1sb3SbMRbhyHzEAh_T3os1Jssh-UdK0RL/view?usp=share_link).

You can find the original SECOND dataset from https://captain-whu.github.io/SCD, and original CNAM-CD dataset from https://github.com/Silvestezhou/CNAM-CD.


**SECOND:** SECOND is a semantic change detection dataset with 7 land cover class annotations, including non-vegetated ground surface, tree, low vegetation, water, buildings, playgrounds, and unchanged areas. The images are collected from different sensors and areas with resolutions between 0.5 and 1 meters, guaranteeing style diversity and scene diversity. In this work, we use 2,668 image pairs with a size of 512 × 512 for training and 1,200 image pairs with a size of 256 × 256 for testing.

Note that during training, each high-resolution (HR) image, Ref image, and land cover change mask are randomly cropped to a size of 256 × 256, and the size of the corresponding LR image is associated with the scaling factors. 

The SECOND dataset is organized in the following structure:

```bash
SECOND/RefSR_dataset/
- train/ # Training set
  - CD_mask/ # 2,668 change detection labels (grayscale)
  - CD_mask_RGB/ # 2,668 change detection labels (RGB)
  - HR_512/ # 2,668 HR images with a size of 512 × 512
  - real_LR_D8_64/ # 2,668 8X downsampled LR images with a size of 64 × 64 using the real-world degradation model
  - real_LR_D16_32/ # 2,668 16X downsampled LR images with a size of 32 × 32 using the real-world degradation model
  - Ref/ # 2,668 reference images with a size of 512 × 512
- test/ # Test set
  - CD_mask/ # 1,200 change detection labels (grayscale)
  - CD_mask_RGB/ # 1,200 change detection labels (RGB)
  - HR_512/ # 1,200 HR images with a size of 256 × 256
  - Bic_LR_D8_32/ # 1,200 8X downsampled LR images with a size of 32 × 32 using the bicubic degradation model
  - Bic_LR_D16_16/ # 1,200 16X downsampled LR images with a size of 16 × 16 using the bicubic degradation model
  - Ref/ # 1,200 reference images with a size of 256 × 256
- `CLASS_Table.txt` # CD mask description
```

**CNAM-CD:** CNAM-CD is a multi-class change detection dataset with a resolution of 0.5 meter, including 6 land cover classes, i.e., bare land, vegetation, water, impervious surfaces (buildings, roads, parking lots, squares, etc.), others (clouds, hard shadows, clutter, etc.), and unchanged areas. The image pairs are collected from Google Earth from 2013 to 2022. We use 2,258 image pairs with a size of 512 × 512 for training and 1,000 image pairs with a size of 256 × 256 for testing.

The CNAM-CD dataset is organized in the following structure:

```bash
CNAM-CD/RefSR_dataset/
- train/ # Training set
  - CD_mask/ # 2,258 change detection labels (grayscale)
  - HR_512/ # 2,258 HR images with a size of  512 × 512
  - real_LR_D8_64/ # 2,258 8X downsampled LR images with a size of  64 × 64 using the real-world degradation model
  - real_LR_D16_32/ real_LR_D16_32/ # 2,258 16X downsampled LR images with a size of  32 × 32using the real-world degradation model
  - Ref/ # 2,258 reference images with a size of 512 × 512
- test/ # Test set
  - CD_mask/ # 1,000 change detection labels (grayscale)
  - HR_512/ # 1,000 HR images with a size of 256 × 256
  - Bic_LR_D8_32/ # 1,000 8X downsampled LR images with a size of 32 × 32 using the bicubic degradation model
  - Bic_LR_D16_16/ # 1,000 16X downsampled LR images with a size of 16 × 16 using the bicubic degradation model
  - Ref/ # 1,000 reference images with a size of 256 × 256 
- `CLASS_Table.txt` # CD mask description
```


## Inference

We provide the pre-trained checkpoint of RefDiff on SECOND_X8:

  Download checkpoints of [SECOND_X8](https://drive.google.com/file/d/15zSQdz7qAv4v0uS9_jnDg5M3YlDaOIzf/view?usp=share_link), place it in `RefDiff-main/` and modify the paths in 'generate_SECOND_X8.sh', generate samples on the test set by command:

  ```bash
  sh generate_SECOND_X8.sh
  ```


## Training

We provide an example for training RefDiff on the SECOND_X8 dataset. Please modify the paths in 'train_SECOND_X8.sh' and run command:

  ```bash
  sh train_SECOND_X8.sh
  ```

Some important arguments for configurations of the training are:

- `outdir`: Where to save the results.
- `data`: Path to the HR images.
- `lr_data`: Path to the LR images.
- `ref_data`: Path to the Ref images.
- `mask_data`: Path to the cd mask.
- `batch`: Total batch size.
- `batch-gpu`: Limit batch size per GPU.
- `up_scale`: Upsampling scale.


If you want to train RefDiff on SECOND_X16 dataset. Please modify the paths and the upsampling scale (i.e., up_scale=16). Similarly for CNAM-CD_X8 and CNAM-CD_X16 datasets.


## Citation

```
@article{dong2024building,
  title={Building Bridges across Spatial and Temporal Resolutions: Reference-Based Super-Resolution via Change Priors and Conditional Diffusion Model},
  author={Dong, Runmin and Yuan, Shuai and Luo, Bin and Chen, Mengxuan and Zhang, Jinxiao and Zhang, Lixian and Li, Weijia and Zheng, Juepeng and Fu, Haohuan},
  journal={arXiv preprint arXiv:2403.17460},
  year={2024}
}
```

## Acknowledgements

This implementation is based on [EDM](https://github.com/NVlabs/edm) and [RDM](https://github.com/THUDM/RelayDiffusion). Thanks for their public codes.
