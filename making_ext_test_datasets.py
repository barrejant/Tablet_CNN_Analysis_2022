import sys
import argparse
import os
import urllib.error
import urllib.request
from argparse import Namespace
import requests
from bs4 import BeautifulSoup
import shutil
import cv2
import glob
from PIL import Image

# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='frontbottom')
parser.add_argument('--output_dir_name', type=str, default='')
args: Namespace = parser.parse_args()

# setup

current_dir: str = os.getcwd()
intermediate_dir_path = os.path.join(current_dir, 'intermediate_dir')
output_dir_path = os.path.join(current_dir, args.output_dir_name)

if os.path.exists(intermediate_dir_path):
    print('A directory intermediate_dir for our temporary usage has already exist.')
    sys.exit()
else:
    os.makedirs(intermediate_dir_path)
    os.makedirs(os.path.join(intermediate_dir_path, 'data'))    

if os.path.exists(output_dir_path):
    print('Specified directory has already exist.')
    sys.exit()
else:
    os.makedirs(output_dir_path)


# download images

def download_image(url, file_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(r.content)

def rotation_(image_file, angle):
    h, w = image_file.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    image = cv2.warpAffine(image_file, M, (w, h))
    return image

# note: if the following method doesn't work well, please check the original website via keys 'bildnr'

base_url = 'https://www.hethport.adwmainz.de/temp_photos/'

if args.mode == 'frontbottom':
    url_01 = ('https://www.hethport.adwmainz.de/fotarch/Bildbetrachter.php?'
            'ori=&po=0&si=100&bildnr=BF00590&fundnr=1991%2Fc%20%2B%202059%2Fc%20%2B%'
            '202387%2Fc%20%2B%20AnAr%207003%20%2B%20'
            'AnAr%2010286&xy=2d2f88b92b792756fc10bd8c1ed5d8ba')
    r_01 = requests.get(url_01)
    soup = BeautifulSoup(r_01.text, "html.parser")
    download_image(
        os.path.join(base_url, soup.find_all("img")[-1]['src'].split('/')[-1].split('wo')[-1]),
        os.path.join(intermediate_dir_path, 'frontbottom_raw_image.jpg')
    )

    img = cv2.imread(os.path.join(intermediate_dir_path, 'frontbottom_raw_image.jpg'))
    rotated_img_01 = rotation_(img, -6)
    cv2.imwrite(os.path.join(intermediate_dir_path, 'data/frontbottom_raw_image_' + '01.jpg'), rotated_img_01[1325:1415,360:780,:])
    
    RecImage_File_List = glob.glob(os.path.join(intermediate_dir_path, 'data/' + "**.jpg"), recursive=True)

    h_list = []
    w_list = []
    for idx in range(len(RecImage_File_List)):
        img = cv2.imread(RecImage_File_List[idx], cv2.IMREAD_UNCHANGED)
        h_list.append(img.shape[0])
        w_list.append(img.shape[1])

    cutting_size = 60
    cutting_size = max(min(h_list), cutting_size)

    for file in RecImage_File_List:
        img = Image.open(file)
        num_horizontal_pieces = int(img.width/cutting_size)
        num_vertical_pieces = int(img.height/cutting_size)
        n_images_width = int((20 + img.width - cutting_size)/20)
        n_images_height = int((20 + img.height - cutting_size)/20)
        for d in range(n_images_width):
            for c in range(n_images_height):
                img_crop = img.crop((20 * d, 20 * c, 20 * d + cutting_size, 20 * c + cutting_size))
                img_crop.save(os.path.join(output_dir_path, file.split('/')[-1].split('.')[-2] + '_' + str(d).zfill(2) + '_' + str(c).zfill(2) + '.jpg'), quality=95)

    shutil.rmtree(intermediate_dir_path)

elif args.mode == 'side':
    url_02 = ('https://www.hethport.adwmainz.de/fotarch/Bildbetrachter.php?'
            'ori=&po=0&si=100&bildnr=BoFN03699&fundnr=709%2Fb'
            '&xy=bd732af412126f2c36a0e52366bb7347')
    r_02 = requests.get(url_02)
    soup = BeautifulSoup(r_02.text, "html.parser")
    download_image(
        os.path.join(base_url, soup.find_all("img")[-1]['src'].split('/')[-1].split('wo')[-1]),
        os.path.join(intermediate_dir_path, 'side_raw_image.jpg')
    )

    img = cv2.imread(os.path.join(intermediate_dir_path, 'side_raw_image.jpg'))
    rotated_img_01 = rotation_(img, -4)
    cv2.imwrite(os.path.join(intermediate_dir_path, 'data/side_raw_image_' + '01.jpg'), rotated_img_01[760:960, 1930:2750, :])
    rotated_img_02 = rotation_(rotated_img_01, +6)
    cv2.imwrite(os.path.join(intermediate_dir_path, 'data/side_raw_image_' + '02.jpg'), rotated_img_02[660:880, 2850:3850, :])

    RecImage_File_List = glob.glob(os.path.join(intermediate_dir_path, 'data/' + "**.jpg"), recursive=True)

    h_list = []
    w_list = []
    for idx in range(len(RecImage_File_List)):
        img = cv2.imread(RecImage_File_List[idx], cv2.IMREAD_UNCHANGED)
        h_list.append(img.shape[0])
        w_list.append(img.shape[1])

    cutting_size = 60
    cutting_size = max(min(h_list), cutting_size)

    for file in RecImage_File_List:
        img = Image.open(file)
        num_horizontal_pieces = int(img.width/cutting_size)
        num_vertical_pieces = int(img.height/cutting_size)
        n_images_width = int((20 + img.width - cutting_size)/20)
        n_images_height = int((20 + img.height - cutting_size)/20)
        for d in range(n_images_width):
            for c in range(n_images_height):
                img_crop = img.crop((20 * d, 20 * c, 20 * d + cutting_size, 20 * c + cutting_size))
                img_crop.save(os.path.join(output_dir_path, file.split('/')[-1].split('.')[-2] + '_' + str(d).zfill(2) + '_' + str(c).zfill(2) + '.jpg'), quality=95)

    shutil.rmtree(intermediate_dir_path)

else:
    print('Please specify side/frontbottom mode here.')
    sys.exit()
