![Python 3.5](https://img.shields.io/badge/python-3.5.5-green.svg)
![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)
![Last Commit](https://img.shields.io/github/last-commit/yhlleo/RG-UNIT)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)]((https://github.com/yhlleo/RG-UNIT/graphs/commit-activity))
![Contributing](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)

## RG-UNIT

Retrieval Guided Unsupervised Multi-domain Image to Image Translation, accepted to ACM International Conference on Multimedia(**ACM MM**), 2020. [[Paper]](https://dl.acm.org/doi/pdf/10.1145/3394171.3413785)|[[arXiv]](http://arxiv.org/abs/2008.04991)|[[code]](https://github.com/yhlleo/RG-UNIT)

![](./figures/framework.png)


### 1 Configuration

 - See the [`environment.yml`](./environment.yaml). We provide an user-friendly configuring method via Conda system, and you can create a new Conda environment using the command:

```
conda env create -f environment.yaml
```

### 2 CelebA Faces

 - Dataset: Download from [Google Drive](https://drive.google.com/open?id=1HnayuXVgqhT1RPzjSV_-yvCp5SxMXPo6), then copy it into `datasets/celeba`:

```
cd datasets/celeba
unzip img_align_celeba.zip
mv img_align_celeba images
```

 - Folder structure:

```
../datasets
  |__celeba
       |__images
       |    |__xxx.jpg
       |    |__...
       |__list_attr_celeba.txt
```

### 3 Pretrained Models

(1) **GMMUNIT + Retrieval**:

 - Download pretrained models: 
   - [`models/celeba_gmm_gen.pt`](https://drive.google.com/file/d/1wVYCx5JwTs_UsyxsxYbAAcDAybucdk4-/view?usp=sharing)
   - [`models/celeba_gmm_dis.pt`](https://drive.google.com/file/d/1kMUy4H5aaRR7_F7LX2v4Cqjgi7MSSH9O/view?usp=sharing)
   
 - Retrieval models & image embeddings:
   - [`models/ret_gmm_00020000.pt`](https://drive.google.com/file/d/1DQysTWCLgL7b-izdoJdzr8unzTXv0dKA/view?usp=sharing)
     - [`models/images_embeddings_gmm.json`](https://drive.google.com/file/d/11nSjoFfIo4kcwkULJ6HsTFkYgrQlY9BL/view?usp=sharing)

### 4 Training

 - a) Preparing GMM-UNIT: copy the model into `./models/celeba_gmm_gen.pt`

 - b) Training Retrieval-Net: `sh ./scripts/train_retrieval_gmm.sh 0 0`

 - c) Training GMM-UNIT + Retrieval: `sh ./scripts/celeba_gmmunit_retrieval.sh 0 0`

### Evaluation codes

We evaluate the performances of the compared models mainly based on this repo: [GAN-Metrics](https://github.com/yhlleo/GAN-Metrics)

### References

If our project is useful for you, please cite our papers:

```
@inproceedings{raul2020retrieval,
author = {Gomez, Raul and Liu, Yahui and De Nadai, Marco and Karatzas, Dimosthenis and Lepri, Bruno and Sebe, Nicu},
title = {Retrieval Guided Unsupervised Multi-Domain Image to Image Translation},
booktitle = {Proceedings of the 28th ACM International Conference on Multimedia},
year = {2020},
doi = {10.1145/3394171.3413785},
url = {https://doi.org/10.1145/3394171.3413785}
}
```

The baseline model GMM-UNIT is based on this paper: [GMM-UNIT: Unsupervised Multi-Domain and Multi-Modal Image-to-Image Translation via Attribute Gaussian Mixture Modeling](https://arxiv.org/pdf/2003.06788.pdf).
