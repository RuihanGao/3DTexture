import os
import cv2
import sys
import glob
import torch
import shutil
import numpy as np
from PIL import Image
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json

MIVOS_PATH='/data/ruihan/projects/NeRF-Texture/thirdparty/MiVOS/' # 'PATH_TO_MIVOS' # https://github.com/hkchengrex/MiVOS
sys.path.append(MIVOS_PATH)
from interactive_invoke import seg_video
from colmap2nerf import colmap2nerf_invoke, optitrack2nerf_invoke


def Laplacian(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()


def cal_ambiguity(path):
    imgs = sorted(glob.glob(path + '/*.png'))
    laplace = np.zeros(len(imgs), np.float32)
    laplace_dict = {}
    for i in range(len(imgs)):
        laplace[i] = Laplacian(cv2.cvtColor(cv2.imread(imgs[i]), cv2.COLOR_BGR2GRAY))
        laplace_dict[imgs[i]] = laplace[i]
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.hist(laplace)
    fig.add_subplot(1, 2, 2)
    plt.plot(np.arange(len(laplace)), laplace)
    if not os.path.exists(path + '/../noise/'):
        os.makedirs(path + '/../noise/')
    elif os.path.exists(path + '../noise/'):
        return None, None
    else:
        return None, None
    plt.savefig(path+'/../noise/laplace.png')
    return laplace, laplace_dict


def select_blur_images(path, nb=10, threshold=0.8, mv_files=False):
    if mv_files and os.path.exists(path + '/../noise/'):
        print('No need to select. Already done.')
        return None, None
    def linear(x, a, b):
        return a * x + b
    laplace, laplace_dic = cal_ambiguity(path)
    if laplace is None:
        return None, None
    imgs = list(laplace_dic.keys())
    amb_img = []
    amb_lap = []
    for i in range(len(laplace)):
        i1 = max(0, int(i - nb / 2))
        i2 = min(len(laplace), int(i + nb / 2))
        lap = laplace[i1: i2]
        para, _ = optimize.curve_fit(linear, np.arange(i1, i2), lap)
        lapi_ = i * para[0] + para[1]
        if laplace[i] / lapi_ < threshold:
            amb_img.append(imgs[i])
            amb_lap.append(laplace[i])
            if mv_files:
                if not os.path.exists(path + '/../noise/'):
                    os.makedirs(path + '/../noise/')
                file_name = amb_img[-1].split('/')[-1].split('\\')[-1]
                shutil.move(amb_img[-1], path + '/../noise/' + file_name)
    return amb_img, amb_lap


def mask_images(img_path, msk_path, sv_path=None, no_mask=False):
    image_names = sorted(os.listdir(img_path))
    image_names = [img for img in image_names if img.endswith('.png') or img.endswith('.jpg')]
    msk_names = sorted(os.listdir(msk_path))
    msk_names = [img for img in msk_names if img.endswith('.png') or img.endswith('.jpg')]
    
    if sv_path is None:
        if img_path.endswith('/'):
            img_path = img_path[:-1]
        sv_path = '/'.join(img_path.split('/')[:-1]) + '/masked_images/'
    if not os.path.exists(sv_path) and not os.path.exists(sv_path + '../unmasked_images/'):
        os.makedirs(sv_path)
    else: 
        return sv_path

    for i in range(len(image_names)):
        image_name, msk_name = image_names[i], msk_names[i]
        mask = np.array(Image.open(msk_path + '/' + image_name))
        image = np.array(Image.open(img_path + '/' + image_name))
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        if no_mask:
            mask = np.ones_like(mask)
        if mask.max() == 1:
            mask = mask * 255
        image[mask==0] = 0
        masked_image = np.concatenate([image, mask[..., np.newaxis]], axis=-1)
        Image.fromarray(masked_image).save(sv_path + image_name)
    return sv_path


def extract_frames_mp4(path, gap=5, sv_path=None):
    if not os.path.exists(path):
        raise NotADirectoryError(path + ' does not exists.')
    if sv_path is None:
        sv_path = '/'.join(path.split('/')[:-1]) + '/images/'
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    else:
        return sv_path
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    cv2.imwrite(sv_path + "/%05d.png" % 0, image)
    count = 1
    image_count = 1
    while success: 
        success, image = vidcap.read()
        if count % gap == 0 and success:
            cv2.imwrite(sv_path + "/%05d.png" % image_count, image)
            image_count += 1
        count += 1
    return sv_path


def rename_images(path):
    image_names = sorted(os.listdir(path))
    org_image_names = [img for img in image_names if img.endswith('.png') or img.endswith('.jpg')]
    new_image_names = ['%05d.png' % i for i in range(len(org_image_names))]
    for i in range(len(org_image_names)):
        shutil.move(path + '/' + org_image_names[i], path + new_image_names[i])
    return org_image_names, new_image_names


if __name__ == '__main__':
    gap = 8 # default 15. change to 1 to debug my_purple_apple
    path_to_dataset = '/data/ruihan/projects/NeRF-Texture/data' # 'PARENT_FOLDER'
    dataset_name = 'dumbbell_20231207_obj_frame' # 'DATASET_NAME'
    input_video = False
    use_optitrack = True
    remove_blur = False
    no_mask = True
    process_poses = True
    
    # Step 1. Extract all images
    if input_video:
        video_path = f'{path_to_dataset}/{dataset_name}/{dataset_name}.mp4'
        if not os.path.exists(video_path):
            video_path = video_path[:-3] + 'mp4'
        print('Extracting frames from video: ', video_path, ' with gap: ', gap)
        img_path = extract_frames_mp4(video_path, gap=gap)
    else:
        img_path = f'{path_to_dataset}/{dataset_name}/images/'
    
    obj_dir = f'{path_to_dataset}/{dataset_name}/'
    
    # Step 2. Remove blurry images
    laplace = None
    if remove_blur:
        print('Removing Blurry Images')
        laplace, _ = select_blur_images(img_path, nb=10, threshold=0.8, mv_files=True)
        # Rename the images here so that the images match the mask names. See L338 in tools/interactive_invoke.py
        if laplace is not None:
            org_image_names, new_image_names = rename_images(img_path)
            filename_mapping = dict(zip(org_image_names, new_image_names))

    # Step 3. Segment images with MiVOS and mask images
    if not no_mask:
        print('Segmenting images with MiVOS ...')
        msk_path = seg_video(img_path=img_path, MIVOS_PATH=MIVOS_PATH)
        torch.cuda.empty_cache()
        print('Masking images with masks ...')
        msked_path = mask_images(img_path, msk_path, no_mask=no_mask)

    # Step 4. Process poses and output transforms.json, where the coordinates follow NeRF convention
    if process_poses:
        # RH: if you use optitrack data, we need to update transforms.py accordingly.  remove noisy frames, rename clean frames, and change file format from .jpg to .png
        if use_optitrack:
            json_path = "transforms_optitrack.json"
            if os.path.exists(os.path.join(obj_dir, json_path)):
                # delete the previous json file
                os.remove(os.path.join(obj_dir, json_path))

            if remove_blur:
                # we import the images_all.txt and process it to a clean txt file which only include non-noisy frames
                img_all_txt = f'{path_to_dataset}/{dataset_name}/images_all.txt'
                assert os.path.exists(img_all_txt), f'{img_all_txt} does not exist.'

                if laplace is not None:
                    # filter out noisy frames
                    print('Removing noisy frames from json data')
                    amb_imgs = [x.split('/')[-1] for x in laplace]
                else:
                    amb_imgs = []

                clean_lines = []

                with open(img_all_txt, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line[0] == '#':
                            # add the same text to clean_lines. The number of images could change, but it doesn't matter much
                            clean_lines.append(line)
                            continue
                        img_name = line.split(' ')[-1]
                        print(f"example img_name: {img_name}")
                        # remove the line if the img_name occurs in amb_imgs
                        if img_name in amb_imgs:
                            continue
                        # rename the frame file_path based on filename_mapping
                        new_img_name = filename_mapping[img_name]
                        new_line = line.replace(img_name, new_img_name)
                        clean_lines.append(new_line)
                # save to new txt file
                img_text = img_all_txt.replace('images_all', 'images')
                with open(img_text, 'w') as f:
                    for line in clean_lines:
                        f.write(line + '\n')
            print(f"Running optitrack2nerf_invoke")
            optitrack2nerf_invoke(img_path, obj_dir=obj_dir, img_txt_path="images.txt" if remove_blur else "images_all.txt", json_path=json_path)

        else:
            json_path = "transforms_colmap.json"
            if os.path.exists(os.path.join(obj_dir, json_path)):
                # delete the previous json file
                os.remove(os.path.join(obj_dir, json_path))
            print('Running COLMAP ...')
            colmap2nerf_invoke(img_path, img_txt_path="images.txt", json_path=json_path)

        # (Optionally) Step 5. Rename masked and unmasked pathes
        # if img_path.endswith('/'):
        #     img_path = img_path[:-1]
        # unmsk_path = '/'.join(img_path.split('/')[:-1]) + '/unmasked_images/'
        # print('Rename masked and unmasked pathes.')
        # if not no_mask:
        #     os.rename(img_path, unmsk_path)
        #     os.rename(msked_path, img_path)
