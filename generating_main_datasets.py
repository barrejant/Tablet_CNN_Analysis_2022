from argparse import Namespace
import numpy as np
import cv2 as cv2
import os
import sys
import shutil
import random
import glob
from PIL import Image
from statistics import mean
import datetime
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir_name', type=str, default='main_dataset')
parser.add_argument('--rectangular_pieces_data_dir', type=str, default='')
parser.add_argument('--cutting_size', type=int, default=60)
parser.add_argument('--th_for_trash_ratio', type=float, default=1.0)
parser.add_argument('--cutting_shift_type', type=str, default='10')
parser.add_argument('--train_test_split_ratio', type=float, default=0.8)
parser.add_argument('--aug_option_zoom', action='store_true')
parser.add_argument('--seed', type=int, default=60, help='seeds = [2201, 9325, 1033, 4179, 1931]')
args: Namespace = parser.parse_args()

current_dir: str = os.getcwd()

# defining n_classes

n_classes = len(os.listdir(os.path.join(current_dir, args.rectangular_pieces_data_dir)))

if n_classes != 4 and n_classes != 8:
	print('Please check the number of classes in the data directory.')
	sys.exit()
else:
	pass

# defining FIVE seeds for validations:

# seeds = [2201, 9325, 1033, 4179, 1931]

seed = args.seed
random.seed(seed)

# defining dataset type

if args.cutting_shift_type == '20' and args.aug_option_zoom is False and n_classes == 4:
	dataset_type_id = 'v01'
elif args.cutting_shift_type == '20' and args.aug_option_zoom is True and n_classes == 4:
	dataset_type_id = 'v02'
elif args.cutting_shift_type == '10' and args.aug_option_zoom is False and n_classes == 4:
	dataset_type_id = 'v03'
elif args.cutting_shift_type == '10' and args.aug_option_zoom is True and n_classes == 4:
	dataset_type_id = 'v04'
elif args.cutting_shift_type == '20' and args.aug_option_zoom is False and n_classes == 8:
	dataset_type_id = 'v001'
elif args.cutting_shift_type == '20' and args.aug_option_zoom is True and n_classes == 8:
	dataset_type_id = 'v002'
elif args.cutting_shift_type == '10' and args.aug_option_zoom is False and n_classes == 8:
	dataset_type_id = 'v003'
elif args.cutting_shift_type == '10' and args.aug_option_zoom is True and n_classes == 8:
	dataset_type_id = 'v004'
else:
	print('Cannot verify dataset type v01 ~ v004')
	sys.exit()

# defining "version" of this process via datetime

dt_now = datetime.datetime.now()
n_version = str(dt_now).split(' ')[-1].replace(':', '_').replace('.', '_')

# defining project_name

project_name = str(args.output_dir_name) + '_' + str(n_version)

# defining train/test split tmp data dirs

train_db_path = os.path.join(current_dir, 'rectangular_pieces_train__' + dataset_type_id + '__' + n_version)
test_db_path = os.path.join(current_dir, 'rectangular_pieces_test__' + dataset_type_id + '__' + n_version)


os.makedirs(train_db_path)
os.makedirs(os.path.join(train_db_path, 'class_01'))
os.makedirs(os.path.join(train_db_path, 'class_02'))
os.makedirs(os.path.join(train_db_path, 'class_03'))
os.makedirs(os.path.join(train_db_path, 'class_04'))
if n_classes == 8:
	os.makedirs(os.path.join(train_db_path, 'class_05'))
	os.makedirs(os.path.join(train_db_path, 'class_06'))
	os.makedirs(os.path.join(train_db_path, 'class_07'))
	os.makedirs(os.path.join(train_db_path, 'class_08'))
os.makedirs(test_db_path)
os.makedirs(os.path.join(test_db_path, 'class_01'))
os.makedirs(os.path.join(test_db_path, 'class_02'))
os.makedirs(os.path.join(test_db_path, 'class_03'))
os.makedirs(os.path.join(test_db_path, 'class_04'))
if n_classes == 8:
	os.makedirs(os.path.join(test_db_path, 'class_05'))
	os.makedirs(os.path.join(test_db_path, 'class_06'))
	os.makedirs(os.path.join(test_db_path, 'class_07'))
	os.makedirs(os.path.join(test_db_path, 'class_08'))


# define & make output_folder

output_folder_name: str =\
	'__TrainTestRatio__' + str(args.train_test_split_ratio)\
	+ '__CutSize__' + str(args.cutting_size)\
	+ '__seed__' + str(seed)\
	+ '__TrashTHRatio__' + str(args.th_for_trash_ratio)\
	+ str(project_name)\
	+ '__DataSetType__' + str(dataset_type_id)
    
output_folder = os.path.join(current_dir, output_folder_name)

os.makedirs(output_folder)
os.makedirs(os.path.join(output_folder, 'train'))
os.makedirs(os.path.join(os.path.join(output_folder, 'train'), 'class_01'))
os.makedirs(os.path.join(os.path.join(output_folder, 'train'), 'class_02'))
os.makedirs(os.path.join(os.path.join(output_folder, 'train'), 'class_03'))
os.makedirs(os.path.join(os.path.join(output_folder, 'train'), 'class_04'))
if n_classes == 8:
	os.makedirs(os.path.join(os.path.join(output_folder, 'train'), 'class_05'))
	os.makedirs(os.path.join(os.path.join(output_folder, 'train'), 'class_06'))
	os.makedirs(os.path.join(os.path.join(output_folder, 'train'), 'class_07'))
	os.makedirs(os.path.join(os.path.join(output_folder, 'train'), 'class_08'))

os.makedirs(os.path.join(output_folder, 'test'))
os.makedirs(os.path.join(os.path.join(output_folder, 'test'), 'class_01'))
os.makedirs(os.path.join(os.path.join(output_folder, 'test'), 'class_02'))
os.makedirs(os.path.join(os.path.join(output_folder, 'test'), 'class_03'))
os.makedirs(os.path.join(os.path.join(output_folder, 'test'), 'class_04'))
if n_classes == 8:
	os.makedirs(os.path.join(os.path.join(output_folder, 'test'), 'class_05'))
	os.makedirs(os.path.join(os.path.join(output_folder, 'test'), 'class_06'))
	os.makedirs(os.path.join(os.path.join(output_folder, 'test'), 'class_07'))
	os.makedirs(os.path.join(os.path.join(output_folder, 'test'), 'class_08'))

# list of paths of rectangular pieces

RecData_Dir = os.path.join(current_dir, args.rectangular_pieces_data_dir)
base_db_file_list = glob.glob(os.path.join(RecData_Dir, "**/**.jpg"), recursive=True)

# splitting train/test

_id_list_class01 = []
_id_list_class02 = []
_id_list_class03 = []
_id_list_class04 = []

_id_list_class05 = []
_id_list_class06 = []
_id_list_class07 = []
_id_list_class08 = []

for idx in base_db_file_list:
	if idx.split('/')[-2] == 'class_01':
		_id_list_class01.append(idx.split('_')[-1].split('.')[-2])
	elif idx.split('/')[-2] == 'class_02':
		_id_list_class02.append(idx.split('_')[-1].split('.')[-2])
	elif idx.split('/')[-2] == 'class_03':
		_id_list_class03.append(idx.split('_')[-1].split('.')[-2])
	elif idx.split('/')[-2] == 'class_04':
		_id_list_class04.append(idx.split('_')[-1].split('.')[-2])
	elif idx.split('/')[-2] == 'class_05':
		_id_list_class05.append(idx.split('_')[-1].split('.')[-2])
	elif idx.split('/')[-2] == 'class_06':
		_id_list_class06.append(idx.split('_')[-1].split('.')[-2])
	elif idx.split('/')[-2] == 'class_07':
		_id_list_class07.append(idx.split('_')[-1].split('.')[-2])
	elif idx.split('/')[-2] == 'class_08':
		_id_list_class08.append(idx.split('_')[-1].split('.')[-2])

train_test_split_ratio = args.train_test_split_ratio

# defining dummies

_id_list_class05_train = []
_id_list_class05_test = []
_id_list_class06_train = []
_id_list_class06_test = []
_id_list_class07_train = []
_id_list_class07_test = []
_id_list_class08_train = []
_id_list_class08_test = []

_id_list_class01_train, _id_list_class01_test\
	= train_test_split(
		sorted(set(_id_list_class01), key=_id_list_class01.index),
		train_size=train_test_split_ratio,
		random_state=seed
		)
_id_list_class02_train, _id_list_class02_test\
	= train_test_split(
		sorted(set(_id_list_class02), key=_id_list_class01.index),
		train_size=train_test_split_ratio,
		random_state=seed
		)
_id_list_class03_train, _id_list_class03_test\
	= train_test_split(
		sorted(set(_id_list_class03), key=_id_list_class01.index),
		train_size=train_test_split_ratio,
		random_state=seed
		)
_id_list_class04_train, _id_list_class04_test\
	= train_test_split(
		sorted(set(_id_list_class04), key=_id_list_class01.index),
		train_size=train_test_split_ratio,
		random_state=seed
		)
if n_classes == 8:
	_id_list_class05_train, _id_list_class05_test\
		= train_test_split(
			sorted(set(_id_list_class05), key=_id_list_class01.index),
			train_size=train_test_split_ratio,
			random_state=seed
			)
	_id_list_class06_train, _id_list_class06_test\
		= train_test_split(
			sorted(set(_id_list_class06), key=_id_list_class01.index),
			train_size=train_test_split_ratio,
			random_state=seed
			)
	_id_list_class07_train, _id_list_class07_test\
		= train_test_split(
			sorted(set(_id_list_class07), key=_id_list_class01.index),
			train_size=train_test_split_ratio,
			random_state=seed
			)
	_id_list_class08_train, _id_list_class08_test\
		= train_test_split(
			sorted(set(_id_list_class08), key=_id_list_class01.index),
			train_size=train_test_split_ratio,
			random_state=seed
			)
else:
	pass

for idx in base_db_file_list:
	if idx.split('/')[-2] == 'class_01':
		if idx.split('_')[-1].split('.')[-2] in _id_list_class01_train:
			shutil.copy(idx, train_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
		else:
			shutil.copy(idx, test_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
	elif idx.split('/')[-2] == 'class_02':
		if idx.split('_')[-1].split('.')[-2] in _id_list_class02_train:
			shutil.copy(idx, train_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
		else:
			shutil.copy(idx, test_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
	elif idx.split('/')[-2] == 'class_03':
		if idx.split('_')[-1].split('.')[-2] in _id_list_class03_train:
			shutil.copy(idx, train_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
		else:
			shutil.copy(idx, test_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
	elif idx.split('/')[-2] == 'class_04':
		if idx.split('_')[-1].split('.')[-2] in _id_list_class04_train:
			shutil.copy(idx, train_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
		else:
			shutil.copy(idx, test_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
	elif idx.split('/')[-2] == 'class_05':
		if idx.split('_')[-1].split('.')[-2] in _id_list_class05_train:
			shutil.copy(idx, train_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
		else:
			shutil.copy(idx, test_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
	elif idx.split('/')[-2] == 'class_06':
		if idx.split('_')[-1].split('.')[-2] in _id_list_class06_train:
			shutil.copy(idx, train_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
		else:
			shutil.copy(idx, test_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
	elif idx.split('/')[-2] == 'class_07':
		if idx.split('_')[-1].split('.')[-2] in _id_list_class07_train:
			shutil.copy(idx, train_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
		else:
			shutil.copy(idx, test_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
	elif idx.split('/')[-2] == 'class_08':
		if idx.split('_')[-1].split('.')[-2] in _id_list_class08_train:
			shutil.copy(idx, train_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])
		else:
			shutil.copy(idx, test_db_path + '/' + idx.split('/')[-2] + '/' + idx.split('/')[-1])

# defining data augmentation for rectangular image pieces

shining = True
resizing = False
clahe = False
noising_white = False
noising_black = False
zoom = args.aug_option_zoom
horizontal_flip = True
vertical_flip = True
rotation = False


def fill(image_file, h, w):
	image_file = cv2.resize(image_file, (w, h), cv2.INTER_CUBIC)
	return image_file


def zoom_(image_file, value):
	if value > 1 or value < 0:
		print('Value for zoom should be less than 1 and greater than 0')
		return image_file
	value = random.uniform(value, 1)
	h, w = image_file.shape[:2]
	h_taken = int(value*h)
	w_taken = int(value*w)
	h_start = random.randint(0, h-h_taken)
	w_start = random.randint(0, w-w_taken)
	image_file_cp = image_file[h_start:h_start+h_taken, w_start:w_start+w_taken, :]
	image_file_final = fill(image_file_cp, h, w)
	return image_file_final


def horizontal_flip_(image_file, flag):
	if flag:
		return cv2.flip(image_file, 1)
	else:
		return image_file


def vertical_flip_(image_file, flag):
	if flag:
		return cv2.flip(image_file, 0)
	else:
		return image_file


def rotation_(image_file, angle):
	angle = int(random.uniform(-angle, angle))
	h, w = image_file.shape[:2]
	M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
	image_file = cv2.warpAffine(image_file, M, (w, h))
	return image_file


def data_augmentation(image_dir_path):
	file_list_01 = glob.glob(os.path.join(image_dir_path, "**/**.jpg"), recursive=True)

	for item in file_list_01:
		split_name = item.split('/')
		output_name = split_name[-1].split('.')[-2] + '_'
		class_name = '__' + item.split('/')[-2] + '__'

		# shining
		if shining is True:
			for _id in range(7):
				_image = cv2.imread(item, cv2.IMREAD_UNCHANGED)
				chg_img_01 = _image/255*(0.6 + _id * 0.2)
				cv2.imwrite(
					os.path.join(
						step_01_dir,
						class_name + output_name + 'shine_' + str(_id) + '_raw.jpg'
					),
					chg_img_01
				)
		else:
			pass

		# clahe
		_image = cv2.imread(item, cv2.IMREAD_UNCHANGED)
		if clahe is True:
			clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
			chg_img_02 = clahe_.apply(gray_image)
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'clahe_' + 'raw.jpg'
				),
				chg_img_02
			)
		else:
			pass

		# adding noises white/black
		_image = cv2.imread(item, cv2.IMREAD_UNCHANGED)
		row, col, ch = _image.shape

		if noising_white is True:
			pts_x = np.random.randint(0, col-1, 1000)
			pts_y = np.random.randint(0, row-1, 1000)
			_image[(pts_y, pts_x)] = (255, 255, 255)
			chg_img_03 = _image
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'noize_white_' + 'raw.jpg'
				),
				chg_img_03
			)

		_image = cv2.imread(item, cv2.IMREAD_UNCHANGED)
		row, col, ch = _image.shape
		if noising_black is True:
			pts_x = np.random.randint(0, col-1, 1000)
			pts_y = np.random.randint(0, row-1, 1000)
			_image[(pts_y, pts_x)] = (0, 0, 0)
			chg_img_04 = _image
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'noize_black_' + 'raw.jpg'
				),
				chg_img_04
			)

		# zoom
		_image = cv2.imread(item, cv2.IMREAD_UNCHANGED)
		if zoom is True:
			for i in range(10):
				shift_value = 0.4 + 0.06 * i
				chg_img_05 = zoom_(_image, shift_value)
				cv2.imwrite(
					os.path.join(
						step_01_dir,
						class_name + output_name + 'zoom' + '_raw_' + str(i).zfill(2) + '.jpg'
					),
					chg_img_05
				)
		else:
			pass

		# horizontal flip
		_image = cv2.imread(item, cv2.IMREAD_UNCHANGED)

		if horizontal_flip is True:
			chg_img_09 = horizontal_flip_(_image, True)
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'horizontal_flip_' + 'raw.jpg'
				),
				chg_img_09
			)
		else:
			pass

		if zoom is True and horizontal_flip is True:
			for i in range(10):
				shift_value = 0.4 + 0.06 * i
				chg_img_09 = horizontal_flip_(_image, True)
				_image = chg_img_09
				chg_img_06 = zoom_(_image, shift_value)
				cv2.imwrite(
					os.path.join(
						step_01_dir,
						class_name + output_name + 'horizontal_flip_zoom' + '_raw_' + str(i).zfill(2) + '.jpg'
					),
					chg_img_06
				)
		else:
			pass

		if shining is True and horizontal_flip is True:
			for _id in range(7):
				chg_img_09 = horizontal_flip_(_image, True)
				_image = chg_img_09
				chg_img_07 = _image/255*(0.6 + _id * 0.2)
				cv2.imwrite(
					os.path.join(
						step_01_dir,
						class_name + output_name + 'horizontal_flip_shine_' + str(_id) + '_raw.jpg'
					),
					chg_img_07
				)
		else:
			pass
  
		if clahe is True and horizontal_flip is True:
			chg_img_09 = horizontal_flip_(_image, True)
			_image = chg_img_09
			clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
			chg_img_08 = clahe_.apply(gray_image)
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'horizontal_flip_clahe_' + 'raw.jpg'
				),
				chg_img_08
			)
		else:
			pass

		if noising_white is True and horizontal_flip is True:
			chg_img_09 = horizontal_flip_(_image, True)
			_image = chg_img_09
			row, col, ch = _image.shape
			pts_x = np.random.randint(0, col-1, 1000)
			pts_y = np.random.randint(0, row-1, 1000)
			_image[(pts_y, pts_x)] = (255, 255, 255)
			chg_img_11 = _image
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'horizontal_flip_noize_white_' + 'raw.jpg'
				),
				chg_img_11
			)
		else:
			pass

		if noising_black is True and horizontal_flip is True:
			chg_img_09 = horizontal_flip_(_image, True)
			_image = chg_img_09
			row, col, ch = _image.shape
			pts_x = np.random.randint(0, col-1, 1000)
			pts_y = np.random.randint(0, row-1, 1000)
			_image[(pts_y, pts_x)] = (0, 0, 0)
			chg_img_12 = _image
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'horizontal_flip_noize_black_' + 'raw.jpg'
				),
				chg_img_12
			)
		else:
			pass


		# vertical flip
		_image = cv2.imread(item, cv2.IMREAD_UNCHANGED)

		if vertical_flip is True:
			chg_img_10 = vertical_flip_(_image, True)
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'vertical_flip_' + 'raw.jpg'
				),
				chg_img_10
			)
		else:
			pass

		if zoom is True and vertical_flip is True:
			for i in range(10):
				shift_value = 0.4 + 0.06 * i
				chg_img_10 = vertical_flip_(_image, True)
				_image = chg_img_10
				chg_img_13 = zoom_(_image, shift_value)
				cv2.imwrite(
					os.path.join(
						step_01_dir,
						class_name + output_name + 'vertical_flip_zoom' + '_raw_' + str(i).zfill(2) + '.jpg'
					),
					chg_img_13
				)
		else:
			pass

		if shining is True and vertical_flip is True:
			for _id in range(7):
				chg_img_10 = vertical_flip_(_image, True)
				_image = chg_img_10
				chg_img_14 = _image/255*(0.6 + _id * 0.2)
				cv2.imwrite(
					os.path.join(
						step_01_dir,
						class_name + output_name + 'vertical_flip_shine_' + str(_id) + '_raw.jpg'
					),
					chg_img_14
				)
		else:
			pass

		if clahe is True and vertical_flip is True:
			chg_img_10 = vertical_flip_(_image, True)
			_image = chg_img_10
			clahe_ = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
			gray_image = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
			chg_img_15 = clahe_.apply(gray_image)
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'vertical_flip_clahe_' + 'raw.jpg'
				),
				chg_img_15
			)
		else:
			pass

		if noising_white is True and vertical_flip is True:
			chg_img_10 = vertical_flip_(_image, True)
			_image = chg_img_10
			row, col, ch = _image.shape
			pts_x = np.random.randint(0, col-1, 1000)
			pts_y = np.random.randint(0, row-1, 1000)
			_image[(pts_y, pts_x)] = (255, 255, 255)
			chg_img_16 = _image
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'vertical_flip_noize_white_' + 'raw.jpg'
				),
				chg_img_16
			)
		else:
			pass

		if noising_black is True and vertical_flip is True:
			chg_img_10 = vertical_flip_(_image, True)
			_image = chg_img_10
			row, col, ch = _image.shape
			pts_x = np.random.randint(0, col-1, 1000)
			pts_y = np.random.randint(0, row-1, 1000)
			_image[(pts_y, pts_x)] = (0, 0, 0)
			chg_img_17 = _image
			cv2.imwrite(
				os.path.join(
					step_01_dir,
					class_name + output_name + 'vertical_flip_noize_black_' + 'raw.jpg'
				),
				chg_img_17
			)
		else:
			pass

		# rotation
		_image = cv2.imread(item, cv2.IMREAD_UNCHANGED)
		if rotation is True:
			for i in range(10):
				chg_img_18 = rotation_(_image, 36 * i)
				cv2.imwrite(
					os.path.join(
						step_01_dir, class_name + output_name + 'rotation' + '_raw_' + str(i).zfill(2) + '.jpg'
					),
					chg_img_18
				)
		else:
			pass

def count_files(directory):
	return len([1 for x in list(os.scandir(directory)) if x.is_file()])


# a step of data augmentation for train/test

train_and_test = ['train', 'test']

for target_train_test in train_and_test:
	if target_train_test == 'train':
		image_path = train_db_path
	else:
		image_path = test_db_path

	project_name = project_name + '_' + str(target_train_test)

	folder_path = os.path.join(current_dir, project_name + '_' + dataset_type_id + '_' + 'augmented_images')
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)
	step_01_dir = os.path.join(
		current_dir,
		project_name + '_' + dataset_type_id + '_' + 'augmented_images/step_01_augmentation_results'
	)
	if not os.path.exists(step_01_dir):
		os.makedirs(step_01_dir)
	step_02_dir = os.path.join(
		current_dir,
		project_name + '_' + dataset_type_id + '_' + 'augmented_images/step_02_cutting_results'
	)
	if not os.path.exists(step_02_dir):
		os.makedirs(step_02_dir)
	step_03_dir = os.path.join(
		current_dir,
		project_name + '_' + dataset_type_id + '_' + 'augmented_images/trash_files'
	)
	if not os.path.exists(step_03_dir):
		os.makedirs(step_03_dir)
	step_04_dir = os.path.join(step_02_dir, 'data')
	if not os.path.exists(step_04_dir):
		os.makedirs(step_04_dir)
	final_dir = os.path.join(
		current_dir,
		project_name + '_' + dataset_type_id + '_' + 'augmented_images/final'
	)
	if not os.path.exists(final_dir):
		os.makedirs(final_dir)
		os.makedirs(os.path.join(final_dir, 'class_01'))
		os.makedirs(os.path.join(final_dir, 'class_02'))
		os.makedirs(os.path.join(final_dir, 'class_03'))
		os.makedirs(os.path.join(final_dir, 'class_04'))
		if n_classes == 8:
			os.makedirs(os.path.join(final_dir, 'class_05'))
			os.makedirs(os.path.join(final_dir, 'class_06'))
			os.makedirs(os.path.join(final_dir, 'class_07'))
			os.makedirs(os.path.join(final_dir, 'class_08'))
		else:
			pass

	result_dir = os.path.join(current_dir, project_name + '_' + dataset_type_id + '_' + 'ResultDir')
	if not os.path.exists(result_dir):
		os.makedirs(result_dir)

	# data augmentation for rectangular image pieces

	data_augmentation(image_path)

	# cutting rectangular image pieces to square ones

	file_list_02 = glob.glob(os.path.join(step_01_dir, "**.jpg"), recursive=True)
	h_list = []
	w_list = []
	for idx in range(len(file_list_02)):
		_image_ = cv2.imread(file_list_02[idx], cv2.IMREAD_UNCHANGED)
		h_list.append(_image_.shape[0])
		w_list.append(_image_.shape[1])
    
	cutting_size = max(min(h_list), args.cutting_size)

	for file in tqdm(file_list_02):
		_image_ = Image.open(file)
		num_horizontal_pieces = int(_image_.width/cutting_size)
		num_vertical_pieces = int(_image_.height/cutting_size)
	
		if args.cutting_shift_type == 'standard':
			for d in range(num_horizontal_pieces):
				for c in range(num_vertical_pieces):
					img_crop = _image_.crop((cutting_size * d, cutting_size * c, cutting_size * (d + 1), cutting_size * (c + 1)))
					img_crop.save(
						os.path.join(
							step_02_dir,
							file.split('/')[-1].split('.')[-2] + '_' + str(d).zfill(2) + '_' + str(c).zfill(2) + '.jpg'
						),
						quality=95
					)
		elif args.cutting_shift_type == '10':
			n_images_width = int((10 + _image_.width - cutting_size)/10)
			n_images_height = int((10 + _image_.height - cutting_size)/10)
			for d in range(n_images_width):
				for c in range(n_images_height):
					img_crop = _image_.crop((10 * d, 10 * c, 10 * d + cutting_size, 10 * c + cutting_size))
					img_crop.save(
						os.path.join(
							step_02_dir,
							file.split('/')[-1].split('.')[-2] + '_' + str(d).zfill(2) + '_' + str(c).zfill(2) + '.jpg'
						),
						quality=95
					)
		elif args.cutting_shift_type == '20':
			n_images_width = int((20 + _image_.width - cutting_size)/20)
			n_images_height = int((20 + _image_.height - cutting_size)/20)
			for d in range(n_images_width):
				for c in range(n_images_height):
					img_crop = _image_.crop((20 * d, 20 * c, 20 * d + cutting_size, 20 * c + cutting_size))
					img_crop.save(
						os.path.join(
							step_02_dir,
							file.split('/')[-1].split('.')[-2] + '_' + str(d).zfill(2) + '_' + str(c).zfill(2) + '.jpg'
						),
						quality=95
					)
		else:
			pass

	# trashing (nearly) blank pieces
	
	file_list_mean = glob.glob(os.path.join(step_02_dir, '**.jpg'), recursive=True)
	file_size_mean_list = []
	for file in file_list_mean:
		file_size_mean_list.append(os.path.getsize(file))

	th_for_trash = mean(file_size_mean_list) * args.th_for_trash_ratio

	file_list_03 = glob.glob(os.path.join(step_02_dir, "**.jpg"), recursive=True)
	for file in file_list_03:
		if os.path.getsize(file) >= th_for_trash:
			pass
		else:
			shutil.move(file, step_03_dir)

	# saving ALL files to prescribed (sub)folders
	
	file_list_04 = glob.glob(os.path.join(step_02_dir, "**.jpg"), recursive=True)
	for file in file_list_04:
		shutil.move(file, step_04_dir)

	file_list_PreFinal = glob.glob(os.path.join(step_04_dir, "**.jpg"), recursive=True)

	for idy in range(len(file_list_PreFinal)):
		if file_list_PreFinal[idy].split('__')[-2] == 'class_01':
			shutil.move(file_list_PreFinal[idy], os.path.join(final_dir, 'class_01'))
		elif file_list_PreFinal[idy].split('__')[-2] == 'class_02':
			shutil.move(file_list_PreFinal[idy], os.path.join(final_dir, 'class_02'))
		elif file_list_PreFinal[idy].split('__')[-2] == 'class_03':
			shutil.move(file_list_PreFinal[idy], os.path.join(final_dir, 'class_03'))
		elif file_list_PreFinal[idy].split('__')[-2] == 'class_04':
			shutil.move(file_list_PreFinal[idy], os.path.join(final_dir, 'class_04'))
		elif file_list_PreFinal[idy].split('__')[-2] == 'class_05':
			shutil.move(file_list_PreFinal[idy], os.path.join(final_dir, 'class_05'))
		elif file_list_PreFinal[idy].split('__')[-2] == 'class_06':
			shutil.move(file_list_PreFinal[idy], os.path.join(final_dir, 'class_06'))
		elif file_list_PreFinal[idy].split('__')[-2] == 'class_07':
			shutil.move(file_list_PreFinal[idy], os.path.join(final_dir, 'class_07'))
		elif file_list_PreFinal[idy].split('__')[-2] == 'class_08':
			shutil.move(file_list_PreFinal[idy], os.path.join(final_dir, 'class_08'))

	final_dir_again = glob.glob(os.path.join(final_dir, "**/**.jpg"), recursive=True)

	for idy in range(len(final_dir_again)):
		if final_dir_again[idy].split('__')[-2] == 'class_01':
			if final_dir_again[idy].split('/')[-4].split('_')[-4] == 'train':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'train/class_01'))
			elif final_dir_again[idy].split('/')[-4].split('_')[-4] == 'test':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'test/class_01'))
		elif final_dir_again[idy].split('__')[-2] == 'class_02':
			if final_dir_again[idy].split('/')[-4].split('_')[-4] == 'train':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'train/class_02'))
			elif final_dir_again[idy].split('/')[-4].split('_')[-4] == 'test':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'test/class_02'))
		elif final_dir_again[idy].split('__')[-2] == 'class_03':
			if final_dir_again[idy].split('/')[-4].split('_')[-4] == 'train':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'train/class_03'))
			elif final_dir_again[idy].split('/')[-4].split('_')[-4] == 'test':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'test/class_03'))
		elif final_dir_again[idy].split('__')[-2] == 'class_04':
			if final_dir_again[idy].split('/')[-4].split('_')[-4] == 'train':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'train/class_04'))
			elif final_dir_again[idy].split('/')[-4].split('_')[-4] == 'test':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'test/class_04'))
		elif final_dir_again[idy].split('__')[-2] == 'class_05':
			if final_dir_again[idy].split('/')[-4].split('_')[-4] == 'train':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'train/class_05'))
			elif final_dir_again[idy].split('/')[-4].split('_')[-4] == 'test':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'test/class_05'))
		elif final_dir_again[idy].split('__')[-2] == 'class_06':
			if final_dir_again[idy].split('/')[-4].split('_')[-4] == 'train':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'train/class_06'))
			elif final_dir_again[idy].split('/')[-4].split('_')[-4] == 'test':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'test/class_06'))
		elif final_dir_again[idy].split('__')[-2] == 'class_07':
			if final_dir_again[idy].split('/')[-4].split('_')[-4] == 'train':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'train/class_07'))
			elif final_dir_again[idy].split('/')[-4].split('_')[-4] == 'test':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'test/class_07'))
		elif final_dir_again[idy].split('__')[-2] == 'class_08':
			if final_dir_again[idy].split('/')[-4].split('_')[-4] == 'train':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'train/class_08'))
			elif final_dir_again[idy].split('/')[-4].split('_')[-4] == 'test':
				shutil.move(final_dir_again[idy], os.path.join(output_folder, 'test/class_08'))

	# cleaning data aug dirs

	shutil.rmtree(folder_path)
	shutil.rmtree(result_dir)

# cleaning train/test tmp dirs

shutil.rmtree(train_db_path)
shutil.rmtree(test_db_path)

# displaying resulting-dirs

print('Output folder is: {}'.format(str(output_folder)))

dir_list = glob.glob(os.path.join(output_folder, '*/*'), recursive=True)

for id_ in range(len(dir_list)):
	print(dir_list[id_], count_files(dir_list[id_]))

sum_total = 0
for id_ in range(len(dir_list)):
	sum_total += count_files(dir_list[id_])
print('Total number of augmented square images is: {}'.format(sum_total))
