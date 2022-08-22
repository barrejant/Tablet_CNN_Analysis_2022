Author: Daichi Kohmoto, Katsutoshi Fukuda, Daisuke Yoshida, Takafumi Matsui, Sachihiro Omura 

Papers: [Main Body][Supplementary Material](not yet published)

Copyright (c) 2022 Daichi Kohmoto  
Released under the MIT license  
https://github.com/barrejant/Tablet_CNN_Analysis_2022/blob/main/LICENSE

These codes are provided to reproduce results of our study, *"CNN-BASED IMAGE MODELS VERIFY A HYPOTHESIS ON TRAINING SKILLS OF WRITING CUNEIFORM TABLETS AT THE AGE OF HITTITE EMPIRE"*. Details for running codes are described in the below. All codes are provided under the MIT license. 

## Our Working Environment


- Ubuntu 18.04 LTS
- at least 1 GPU
- Anaconda
  - an virtual env: tablet_CNN_analysis_2022.yml

Starting by cloning this repository at an appropriate place in your machine
```
git clone https://github.com/barrejant/Tablet_CNN_Analysis_2022.git
cd Tablet_CNN_Analysis_2022
```
and by setting the above virtual environment. 

## Notes

- Each step (except for Step 0) depends on results of previous step. 
- Each step automatically generates necessary directories to store results. 
The parts of names of such directories are necessary to specify via argument parameters by yourself.
- All images used in this study have the extension jpg.

## Steps

Proceed following steps one-by-one. 

### 0. Prepare images from Catalog der Texte der Hethiter of Hethitologie-Portals Mainz.
Downloading image files from [Catalog der Texte der Hethiter of Hethitologie-Portals Mainz](https://www.hethport.uni-wuerzburg.de/CTH/) via the following:
```
python download_raw_images_from_CTH.py --output_dir_name raw_images
```
As a result, our directory has the following structure:
<pre>
working_directory_path
└── raw_images
  ├── 01.jpg
  ├── 02.jpg
  └── 03.jpg
</pre>

### 1. Cropping rectangular image pieces from raw images, defining classes (4 or 8 classes).
In the case that the number of classes is 4, the follwoing command works for our purpose:
```
python cropping_rectangular_image_pieces.py --n_classes 4 --output_dir_name rectangular_images --raw_imagedata_dir raw_images
```
As a result, our directory has the following structure:
<pre>
working_directory_path
├── raw_images
└── rectangular_images__n_classes_4
  ├── class_01
  ├── ...
  └── class_04
</pre>

### 2. Generating 40 main datasets via data augmentation with train/test splitting.
```
python generating_main_datasets.py
```

### 3. Fine-tuning VGG19/ResNet50/InceptionV3 pre-trained models for all main datasets.
```
python 
```
### 4. Outputing results

#### 4.1. Outputing results 1: Learning curves.
```
python generating_learning_curves.py
```
#### 4.2. Outputing results 2: Confusion matrices.
```
python generating_confusion_matrices.py 
```
#### 4.3. Outputing results 3: Class activation mapping for VGG19 fine-tuned models. 
```
python generating_CAM_forVGG19.py 
```

### Citing this repository via BibTeX
```
@software{Tablet_CNN_Analysis_2022,
  author = {Daichi Kohmoto, Katsutoshi Fukuda, Daisuke Yoshida, Takafumi Matsui, Sachihiro Omura},
  month = {8},
  title = {{CNN-BASED IMAGE MODELS VERIFY A HYPOTHESIS ON TRAINING SKILLS OF WRITING CUNEIFORM TABLETS AT THE AGE OF HITTITE EMPIRE}},
  url = {https://github.com/barrejant/Tablet_CNN_Analysis_2022/},
  version = {1.0.0},
  year = {2022}
}
```
