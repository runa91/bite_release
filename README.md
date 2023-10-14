# BITE
[[Project Page](https://bite.is.tue.mpg.de/)] 


## Table of Contents
  * [Description](#description)
  * [Installation](#installation)
    * [Environment](#environment)
    * [Data Preparation](#data-preparation)
    * [Configurations](#configurations)
  * [Usage](#usage)
    * [Demo](#demo)
    * [Training](#training)
    * [Inference](#inference)
  * [Citation](#citation)
  * [License](#license)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)

## Description

**BITE: Beyond priors for Improved Three-D dog pose Estimation** is a method for dog pose and shape estimation.


## Installation

### Environment

The code has been tested with CUDA 10.1, CuDNN 7.5, Python 3.7 and PyTorch 1.6.0. 
```shell
    conda create -n "conda_bite" python=3.7.6 ipython
    conda activate conda_bite
    conda install pytorch==1.6.0 torchvision cudatoolkit=10.1 -c pytorch
```

To install the remaining dependencies run the following command:
```shell
    pip install -r requirements.txt
```


### Data Preparation

All necessary data be downloaded [here](https://owncloud.tuebingen.mpg.de/index.php/s/BpPWyzsmfycXdyj). A folder named 'checkpoint' contains pretrained models, copy it to the main folder of this project. Subfolders in a folder called 'data' should be copied to the corresponding location in your project folder.
Data from stanext_related_data_ground_contact_annotations should be copied to stanext_related_data/ground_contact_annotations, but note that this folder also contains visualization ("vis") subfolders, which you can omit if you are not specifically interested in that part.

Download the Stanford Extra image dataset from https://github.com/benjiebob/StanfordExtra and store it in datasets/StanfordExtra_V12. 

Your folder structure should look as follows:
```bash
folder
├── checkpoint
│   └── ...
├── data
│   ├── breed_data
│   ├── graphcmr_data
│   ├── ground_contact_annotations
│   ├── smal_data
│   ├── smal_data_remeshed
│   ├── stanext_related_data
│   └── statistics
├── datasets
│   ├── test_image_crops
│   ├── StanfordExtra_V12
│   │   ├── StanExtV12_Images
│   │   └── labels
├── results
│   └── ...
├── scripts
│   └── ...
├── src
│   └── ...
```

### Configurations

All configuration files can be found in src/configs. You will need to adjust paths in dataset_path_configs.py. If desired you can change the weights for different loss functions used at training time, see barc_loss_weights.json and refinement_loss_weights_withgc_withvertexwise_addnonflat.json. We do not recommend changing zero-value weights to non-zero values, as some of the unused loss functions were removed.



## Usage

### Inference
#### Gradio Demo 
We release code that is used for the our [huggingface demo](https://bite.is.tue.mpg.de/). You can create your own local demo with a similar interface by running: 
```shellInterface
    python scripts/gradio_demo.py
```

#### Inference for Stanford Extra dataset or full Image Folders
In order to run our pretrained model on new sample images, prepare image crops and put them into the folder datasets/test_image_crops. The crops can have arbitrary rectangular shape, but should show the dog more or less in the center of the image. 

Demo on all images within the folder datasets/test_image_crops:
```shell
python scripts/full_inference_including_ttopt.py \
--workers 12 \
--config refinement_cfg_test_withvertexwisegc_csaddnonflat_crops.yaml \
--model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar \
--suffix ttopt_vtest1
```

Demo on the Stanford Extra dataset:
```shell
python scripts/full_inference_including_ttopt.py \
--workers 12 \
--config refinement_cfg_test_withvertexwisegc_csaddnonflat.yaml \
--model-file-complete cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0/checkpoint.pth.tar \
--suffix ttopt_vtest1
```

### Training
Train BARC+ part of the network:
* basically the same as BARC training code released in [this repo](https://github.com/runa91/barc_release)
* we call it BARC+ because of the new dog model
```shell
python scripts/train.py \
--workers 12 \
--checkpoint debug_dm39dnnv3_barc_v2b \
--loss-weight-path barc_loss_weights_with3dcgloss_higherbetaloss_v2_dm39dnnv3v2.json \
--config barc_cfg_train.yaml start \
--model-file-hg hg_ksp_fromnewanipose_stanext_v0/checkpoint.pth.tar \
--model-file-3d barc_3d_pret/checkpoint.pth.tar
```

Train refinement network (full network = BITE):
```shell
python scripts/train_withref.py \
--checkpoint debug_cvpr23_dm39dnnv3barcv2b_refwithgcpervertisflat0morestanding0 \
--workers 12 \
-lw barc_loss_weights_allzeros.json \
-lwr refinement_loss_weights_withgc_withvertexwise_addnonflat.json \
--config refinement_cfg_train_withvertexwisegc_isflat_csmorestanding.yaml \
continue \
--model-file-complete dm39dnnv3_barc_v2b/checkpoint.pth.tar \
--new-optimizer 1

```


## Citation

If you find this Model & Software useful in your research we would kindly ask you to cite:

```bibtex
@inproceedings{rueegg2023bite,
    title = {BITE: Beyond priors for Improved Three-D dog pose Estimation},
    author = {Rueegg, Nadine and Tripathi, Shashank and Schindler, Konrad and Black, Michael J. and Zuffi, Silvia},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2023}
}
```
Please consider also citing its predecessor BARC:
```bibtex
@inproceedings{rueegg2022barc,
    title = {BARC: Learning to Regress 3D Dog Shape from Images by Exploiting Breed Information},
    author = {Rueegg, Nadine and Zuffi, Silvia and Schindler, Konrad and Black, Michael J.},
    booktitle = {Proceedings IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
    year = {2022}
}
```


## License

Software Copyright License for non-commercial scientific research purposes.
Please read carefully the following [terms and conditions](LICENSE) and any accompanying
documentation before you download and/or use BITE data, model and
software, (the "Data & Software"), including 3D meshes, images, videos,
textures, software, scripts, and animations. By downloading and/or using the
Data & Software (including downloading, cloning, installing, and any other use
of the corresponding github repository), you acknowledge that you have read
these [terms and conditions](LICENSE), understand them, and agree to be bound by them. If
you do not agree with these [terms and conditions](LICENSE), you must not download and/or
use the Data & Software. Any infringement of the terms of this agreement will
automatically terminate your rights under this [License](LICENSE).


## Note
In addition to the code for our CVPR paper which may be run using the commands within the "Usage" section, you can find different functionalities which did not end up in the final version of the project (body part segmentation for example). However, I have in the meanwhile left University and while you are free to use those code snipplets, I will not be able to provide support.


## Contact

For commercial licensing (and all related questions for business applications), please contact [ps-licensing@tue.mpg.de](mailto:ps-licensing@tue.mpg.de).

