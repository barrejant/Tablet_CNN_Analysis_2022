import sys
import argparse
import os
import urllib.error
import urllib.request
from argparse import Namespace
import requests
from bs4 import BeautifulSoup

# input parameters

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir_name', type=str, default='raw_images')
args: Namespace = parser.parse_args()

# setup

current_dir: str = os.getcwd()
output_dir_path = os.path.join(current_dir, args.output_dir_name)

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

# note: if the following method doesn't work well, please check the original website via keys 'bildnr'

base_url = 'https://www.hethport.adwmainz.de/temp_photos/'
url_01 = ('https://www.hethport.adwmainz.de/fotarch/Bildbetrachter.php?'
          'bildnr=BF00560&fundnr=709%2Fb+%2B+738%2Fb+%2B+2107%2Fc+%2B+2387%2Fc+%2B+AnAr+10286&pl=0&po=300&si=100&'
          'xy=8f60b3da376ef7b446d4e44d95125243&x=18&y=20')
url_02 = ('https://www.hethport.adwmainz.de/fotarch/Bildbetrachter.php?'
          'ori=&po=0&si=100&bildnr=BF00558&fundnr=AnAr%207003%20%2B%20AnAr%209155%20%2B%20AnAr%2010286&'
          'xy=ca62f0310e26931d5a38addec8e20dd7')
url_03 = ('https://www.hethport.adwmainz.de/fotarch/Bildbetrachter.php?'
          'ori=&po=0&si=100&bildnr=BF00593&fundnr=709%2Fb%20%2B%20738%2Fb%20%2B%20751%2Fb%20%2B%20756%'
          '2Fb%20%2B%201134%2Fc%20%2B%201169%2Fc%20%2B%201721%2Fc%20%2B%20AnAr%208348%20%2B%20AnAr%'
          '209139%20%2B%20AnAr%209155&xy=77de024223a28a91f4478cf5a91f0cc9')

r_01 = requests.get(url_01)
soup = BeautifulSoup(r_01.text, "html.parser")
download_image(
    os.path.join(base_url, soup.find_all("img")[-1]['src'].split('/')[-1].split('wo')[-1]),
    os.path.join(output_dir_path, '01.jpg')
)

r_02 = requests.get(url_02)
soup = BeautifulSoup(r_02.text, "html.parser")
download_image(
    os.path.join(base_url, soup.find_all("img")[-1]['src'].split('/')[-1].split('wo')[-1]),
    os.path.join(output_dir_path, '02.jpg')
)

r_03 = requests.get(url_03)
soup = BeautifulSoup(r_03.text, "html.parser")
download_image(
    os.path.join(base_url, soup.find_all("img")[-1]['src'].split('/')[-1].split('wo')[-1]),
    os.path.join(output_dir_path, '03.jpg')
)
