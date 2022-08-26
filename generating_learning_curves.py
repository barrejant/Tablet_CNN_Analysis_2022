import os
import glob
import argparse
import sys
import matplotlib.pyplot as plt

# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir_name', type=str, default='LearningCurves')
args = parser.parse_args()

# preparing an output directory

current_dir: str = os.getcwd()

output_dir_path = os.path.join(current_dir, args.output_dir_name)
if not os.path.isdir(output_dir_path): 
    os.mkdir(output_dir_path)
else:
    print('Specified directory has already exist.')
    sys.exit()

# making learning curves

LossRecords_List = glob.glob(os.path.join(os.path.join(current_dir, 'FineTunedModels'), '*_loss.txt'), recursive=True)

for Record in LossRecords_List:
    FileOpen = open(Record)
    Lines = FileOpen.readlines()
    LossValues = []
    for Line in Lines[1:]:
        LossValues.append(float(str(Line).split('\n')[-2]))

    ModelName = ''
    if Record.split('/')[-1].split('_')[3] == 'vgg19':
        ModelName = 'VGG19'
    elif Record.split('/')[-1].split('_')[3] == 'resnet50':
        ModelName = 'ResNet50'
    elif Record.split('/')[-1].split('_')[3] == 'inception':
        ModelName = 'InceptionV3'

    if ModelName == 'VGG19':
        fig, ax1 = plt.subplots(figsize=(5, 5), dpi=300)
        ax1.set_title(
            'Model: {},'.format(ModelName)
            + ' Dataset: {},'.format(Record.split('/')[-1].split('_')[13])
            + ' Seed: {},\n'.format(Record.split('/')[-1].split('_')[15])
            + 'Score: {}'.format(Record.split('/')[-1].split('_')[6]),
            size=10
        )
        ax1.plot(LossValues, 'b-')
        ax1.set_xlabel('Iterations', fontsize=10.0)
        ax1.set_ylabel('Loss', color='black', fontsize=10.0)
        ax1.tick_params('y', colors='black')

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir_path,
                str(ModelName) + '_' + str(Record.split('/')[-1].split('_')[13])
                + '_' + str(Record.split('/')[-1].split('_')[15])
                + '_' + str(Record.split('/')[-1].split('_')[6]) + ".jpg"
                ),
                dpi=300
        )
        plt.clf()
        plt.close()
    elif ModelName == 'ResNet50':
        fig, ax1 = plt.subplots(figsize=(5, 5), dpi=300)
        ax1.set_title(
            'Model: {},'.format(ModelName)
            + ' Dataset: {},'.format(Record.split('/')[-1].split('_')[13])
            + ' Seed: {},\n'.format(Record.split('/')[-1].split('_')[15])
            + 'Score: {}'.format(Record.split('/')[-1].split('_')[6]),
            size=10
        )
        ax1.plot(LossValues, 'b-')
        ax1.set_xlabel('Iterations', fontsize=10.0)
        ax1.set_ylabel('Loss', color='black', fontsize=10.0)
        ax1.tick_params('y', colors='black')

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir_path,
                str(ModelName) + '_' + str(Record.split('/')[-1].split('_')[13])
                + '_' + str(Record.split('/')[-1].split('_')[15]) + '_'
                + str(Record.split('/')[-1].split('_')[6]) + ".jpg"
                ),
                dpi=300
            )
        plt.clf()
        plt.close()
        
    elif ModelName == 'InceptionV3':
        fig, ax1 = plt.subplots(figsize=(5, 5), dpi=300)
        ax1.set_title(
            'Model: {},'.format(ModelName)
            + ' Dataset: {},'.format(Record.split('/')[-1].split('_')[14])
            + ' Seed: {},\n'.format(Record.split('/')[-1].split('_')[16])
            + 'Score: {}'.format(Record.split('/')[-1].split('_')[7]),
            size=10
        )
        ax1.plot(LossValues, 'b-')
        ax1.set_xlabel('Iterations', fontsize=10.0)
        ax1.set_ylabel('Loss', color='black', fontsize=10.0)
        ax1.tick_params('y', colors='black')

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                output_dir_path,
                str(ModelName) + '_' + str(Record.split('/')[-1].split('_')[14])
                + '_' + str(Record.split('/')[-1].split('_')[16]) + '_'
                + str(Record.split('/')[-1].split('_')[7]) + ".jpg"
                ),
                dpi=300
            )
        plt.clf()
        plt.close()
