Author: Daichi Kohmoto, Katsutoshi Fukuda, Daisuke Yoshida, Takafumi Matsui, Sachihiro Omura

These codes are provided to reproduce results of our study, "CNN-BASED IMAGE MODELS VERIFY A HYPOTHESIS ON TRAINING SKILLS OF WRITING CUNEIFORM TABLETS AT THE AGE OF HITTITE EMPIRE". Details for running codes are described in the below. All codes are provided under the MIT license. 

Our working environment is based on 


- Ubuntu 18.04 LTS
- Anaconda

and is provided as .yml file.


Starting by moving to working directory as 

```
cd working_directory_path
```

0. Prepare images from Catalog der Texte der Hethiter of Hethitologie-Portals Mainz.

```
python download_raw_images_from_CTH.py --output_dir ./raw_images
```


1. Cropping rectangular image pieces from raw images.

```
python cropping_rectangular_image_pieces.py --output_dir ./rectangular_images
```

2. Generating main datasets via data augmentation.
```
python generating_main_datasets.py
```

3. Fine-tuning VGG19/ResNet50/InceptionV3 pre-trained models for our study.
```
python 
```
4. Outputing results 1: Learning curves.
```
python generating_learning_curves.py
```
5. Outputing results 2: Confusion matrices.
```
python generating_confusion_matrices.py 
```
6. Outputing results 3: Class activation mapping for VGG19 fine-tuned models. 
```
python generating_CAM_forVGG19.py 
```


Copyright (c) 2022 Daichi Kohmoto
Released under the MIT license
https://github.com/barrejant/Tablet_CNN_Analysis_2022/blob/main/MIT-LICENSE
