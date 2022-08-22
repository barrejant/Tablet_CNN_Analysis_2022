# modules

import argparse
import glob
import os
import sys
from argparse import Namespace

import cv2

# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', type=int, default=4)
parser.add_argument('--output_dir_name', type=str, default='rectangular_images')
parser.add_argument('--raw_imagedata_dir', type=str, default='raw_images')
args: Namespace = parser.parse_args()

# project_name

project_name = args.output_dir_name

# making output_dir

current_dir: str = os.getcwd()
output_dir_name: str = args.output_dir_name
output_dir = os.path.join(current_dir, output_dir_name + '__n_classes_' + str(args.n_classes))

if args.n_classes == 4:
    if os.path.exists(output_dir):
        print('Specified directory has already exist.')
        sys.exit()
    else:
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'class_01'))
        os.makedirs(os.path.join(output_dir, 'class_02'))
        os.makedirs(os.path.join(output_dir, 'class_03'))
        os.makedirs(os.path.join(output_dir, 'class_04'))
elif args.n_classes == 8:
    if os.path.exists(output_dir):
        print('Specified directory has already exist.')
        sys.exit()
    else:
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'class_01'))
        os.makedirs(os.path.join(output_dir, 'class_02'))
        os.makedirs(os.path.join(output_dir, 'class_03'))
        os.makedirs(os.path.join(output_dir, 'class_04'))
        os.makedirs(os.path.join(output_dir, 'class_05'))
        os.makedirs(os.path.join(output_dir, 'class_06'))
        os.makedirs(os.path.join(output_dir, 'class_07'))
        os.makedirs(os.path.join(output_dir, 'class_08'))
else:
    print('The value of n_classes is invalid. Please specify one of values, 4 or 8. Stopping this process.')
    sys.exit()

# raw_imagedata_dir

raw_imagedata_files = args.raw_imagedata_dir + '/**.jpg'
rawimage_file_list = glob.glob(os.path.join(current_dir, raw_imagedata_files), recursive=True)


# utils

def rotation_(before_rotation_image, angle: int):
    h, w = before_rotation_image.shape[:2]
    rm2_d = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    after_rotated_image = cv2.warpAffine(before_rotation_image, rm2_d, (w, h))
    return after_rotated_image


# main process

if args.n_classes == 4:
    for idx in range(3):
        if rawimage_file_list[idx].split('.')[-2].split('/')[-1] == '01':
            img = cv2.imread(rawimage_file_list[idx])
            rotated_img_01 = rotation_(img, -5)
            rotated_img_02 = rotation_(rotated_img_01, -2)
            rotated_img_03 = rotation_(rotated_img_02, -1)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_03[470:540, 400:780, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_03[610:690, 400:760, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_03[750:820, 400:770, :]
            )
            rotated_img_04 = rotation_(rotated_img_03, +1)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_04[960:1030, 400:850, :]
            )
            rotated_img_05 = rotation_(rotated_img_04, +2)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '05.jpg'
                ),
                rotated_img_05[1060:1130, 400:810, :]
            )
        elif rawimage_file_list[idx].split('.')[-2].split('/')[-1] == '02':
            img = cv2.imread(rawimage_file_list[idx])
            rotated_img_01 = rotation_(img, -3)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_01[1000:1080, 240:500, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_01[1080:1170, 230:470, :]
            )
            rotated_img_02 = rotation_(rotated_img_01, -2)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_02[440:520, 880:1230, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_02[540:620, 820:1230, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '05.jpg'
                ),
                rotated_img_02[810:890, 700:980, :]
            )
        elif rawimage_file_list[idx].split('.')[-2].split('/')[-1] == '03':
            img = cv2.imread(rawimage_file_list[idx])
            rotated_img_01 = rotation_(img, -7)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_03/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_01[940:1010, 190:440, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_03/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_01[1060:1130, 140:440, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_03/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_01[1140:1210, 120:450, :]
            )
            rotated_img_02 = rotation_(rotated_img_01, +12)
            rotated_img_03 = rotation_(rotated_img_02, -7)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_03[1010:1090, 740:930, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_03[1090:1170, 740:1000, :]
            )
            rotated_img_04 = rotation_(rotated_img_03, +4)
            rotated_img_05 = rotation_(rotated_img_04, -7)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_05[1240:1320, 730:1080, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_05[1320:1400, 740:1120, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir, 'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '05.jpg'
                ),
                rotated_img_05[1400:1480, 790:1110, :]
            )
elif args.n_classes == 8:
    for idx in range(3):
        if rawimage_file_list[idx].split('.')[-2].split('/')[-1] == '01':
            img = cv2.imread(rawimage_file_list[idx])
            rotated_img_01 = rotation_(img, -5)
            rotated_img_02 = rotation_(rotated_img_01, -2)
            rotated_img_03 = rotation_(rotated_img_02, -1)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_03[470:540, 400:590, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_03[610:690, 400:580, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_03[750:820, 400:580, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_03[470:540, 590:780, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_03[610:690, 580:760, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_03[750:820, 580:770, :]
            )
            rotated_img_04 = rotation_(rotated_img_03, +1)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_04[960:1030, 400:620, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_04[960:1030, 620:850, :]
            )
            rotated_img_05 = rotation_(rotated_img_04, +2)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_01/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '05.jpg'
                ),
                rotated_img_05[1060:1130, 400:600, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_02/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '05.jpg'
                ),
                rotated_img_05[1060:1130, 600:810, :]
            )
        elif rawimage_file_list[idx].split('.')[-2].split('/')[-1] == '02':
            img = cv2.imread(rawimage_file_list[idx])
            rotated_img_01 = rotation_(img, -3)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_03/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_01[1000:1080, 240:500, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_01[1080:1170, 230:470, :]
            )
            rotated_img_02 = rotation_(rotated_img_01, -2)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_03/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_02[440:520, 880:1050, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_03/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_02[540:620, 820:1020, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_03/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_02[810:890, 700:840, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_02[440:520, 1050:1230, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_02[540:620, 1020:1230, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_04/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_02[810:890, 840:980, :]
            )
        elif rawimage_file_list[idx].split('.')[-2].split('/')[-1] == '03':
            img = cv2.imread(rawimage_file_list[idx])
            rotated_img_01 = rotation_(img, -7)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_05/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_01[940:1010, 190:310, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_05/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_01[1060:1130, 140:290, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_05/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_01[1140:1210, 120:280, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_06/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_01[940:1010, 310:440, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_06/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_01[1060:1130, 290:440, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_06/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_01[1140:1210, 280:450, :]
            )
            rotated_img_02 = rotation_(rotated_img_01, +12)
            rotated_img_03 = rotation_(rotated_img_02, -7)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_07/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_03[1010:1090, 740:930, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_08/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '01.jpg'
                ),
                rotated_img_03[1090:1170, 740:1000, :]
            )
            rotated_img_04 = rotation_(rotated_img_03, +4)
            rotated_img_05 = rotation_(rotated_img_04, -7)
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_07/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_05[1240:1320, 730:900, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_07/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_05[1320:1400, 740:940, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_07/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_05[1400:1480, 790:950, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_08/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '02.jpg'
                ),
                rotated_img_05[1240:1320, 900:1080, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_08/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '03.jpg'
                ),
                rotated_img_05[1320:1400, 940:1120, :]
            )
            cv2.imwrite(
                os.path.join(
                    output_dir,
                    'class_08/' + rawimage_file_list[idx].split('/')[-1].split('.')[-2] + '_' + '04.jpg'
                ),
                rotated_img_05[1400:1480, 950:1110, :]
            )
