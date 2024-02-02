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
import os.path as osp

MIVOS_PATH='/data/ruihan/projects/NeRF-Texture/thirdparty/MiVOS/' # 'PATH_TO_MIVOS' # https://github.com/hkchengrex/MiVOS
sys.path.append(MIVOS_PATH)
from interactive_invoke import seg_video
from colmap2nerf import colmap2nerf_invoke, optitrack2nerf_invoke, create_gelsight_dict_from_txt_and_img_dict_params


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
    """
    Return directory of masked images and whether we need to rename the maksed image directory
    """
    image_names = sorted(os.listdir(img_path))
    image_names = [img for img in image_names if img.endswith('.png') or img.endswith('.jpg')]
    msk_names = sorted(os.listdir(msk_path))
    msk_names = [img for img in msk_names if img.endswith('.png') or img.endswith('.jpg')]
    
    if sv_path is None:
        if img_path.endswith('/'):
            img_path = img_path[:-1]
        sv_path = '/'.join(img_path.split('/')[:-1]) + '/masked_images/'
    
    if os.path.exists(sv_path) or os.path.exists('/'.join(img_path.split('/')[:-1]) + '/unmasked_images/'):
        # since if we use the masked images, we would rename the masked_images to images and rename images to unmasked_images
        print(f"Find existing masked images. Skip masking.")
        return sv_path, False
    
    else:
        os.makedirs(sv_path)
        # if os.path.exists('/'.join(img_path.split('/')[:-1]) + '/mask'):
        #     print(f"Find existing masks. Skip masking.")
        #     return sv_path, True

        # else:
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
        return sv_path, True


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
    dataset_name = 'onemarker_20240130_obj_frame' # 'woodbox_20240112_obj_frame' # 'dumbbell_20231207_obj_frame' # 'DATASET_NAME'
    input_video = False
    use_optitrack = True
    remove_blur = True # Note: if want to remove_blur, run use_optitrack=True first, then run use_optitrack=False for colmap processing
    no_mask = True
    process_poses = True # Option to run optitrack2nerf_invoke or colmap2nerf_invoke to get json file. Set to false for the second pass where we only need to remove inaccurate mask images
    remove_inaccurate_mask = False

    process_gelsight_poses = True # Option to process gelsight poses. Given camera poses and tf saved in img_dict_params.json, we can convert original gelsight poses stored in .txt file to .json format for NeRF training

    use_masked_images = False # use masked images for NeRF training. In that case, rename the folder "masked_imaes" to "images" and rename "images" to "unmasked_images"
    
    # 2024.01.04 In custom dataset, we rename the folder containing all images as "camera_images". 
    # When preparing the dataset, we make a copy from "camera_images" to "images" and then do the rest as normal. Therefore, after removing blurry images, we still have the orignal data in "camera_images" folder. Easier to delete the rest and restart.
    # Step 1. Extract all images
    if input_video:
        video_path = f'{path_to_dataset}/{dataset_name}/{dataset_name}.mp4'
        if not os.path.exists(video_path):
            video_path = video_path[:-3] + 'mp4'
        print('Extracting frames from video: ', video_path, ' with gap: ', gap)
        img_path = extract_frames_mp4(video_path, gap=gap)
    else:
        img_path = f'{path_to_dataset}/{dataset_name}/images/'

    org_img_path = f'{path_to_dataset}/{dataset_name}/camera_images/'
    if os.path.exists(org_img_path):
        if not os.path.exists(img_path):
            # copy the original images to img_path
            print(f"Copying all images from {org_img_path} to {img_path}")
            shutil.copytree(org_img_path, img_path)
    
    obj_dir = f'{path_to_dataset}/{dataset_name}/'
    filename_mapping = None
    
    # Step 2. Remove blurry images
    laplace = None
    if remove_blur:
        print('Removing Blurry Images')
        laplace, _ = select_blur_images(img_path, nb=10, threshold=0.8, mv_files=True)
        # Rename the images here so that the images match the mask names. See L338 in tools/interactive_invoke.py
        if laplace is not None:
            org_image_names, new_image_names = rename_images(img_path)
            filename_mapping = dict(zip(org_image_names, new_image_names))

        if use_optitrack:
            # we import the images_all.txt and process it to a clean txt file which only include non-noisy frames
            img_all_txt = f'{path_to_dataset}/{dataset_name}/images_all.txt'
            assert os.path.exists(img_all_txt), f'{img_all_txt} does not exist.'

            if laplace is not None:
                # filter out noisy frames
                print('Removing noisy frames from json data')
                amb_imgs = [x.split('/')[-1] for x in laplace]

                clean_lines = []
                # create a clean txt file only if remove_blur is performed
                with open(img_all_txt, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line[0] == '#':
                            # add the same text to clean_lines. The number of images could change, but it doesn't matter much
                            clean_lines.append(line)
                            continue
                        img_name = line.split(' ')[-1]
                        # remove the line if the img_name occurs in amb_imgs
                        if img_name in amb_imgs:
                            continue
                        # rename the frame file_path based on filename_mapping
                        new_img_name = filename_mapping[img_name]
                        new_line = line.replace(img_name, new_img_name)
                        clean_lines.append(new_line)
                # save to new txt file
                img_text = img_all_txt.replace('images_all', 'images_optitrack')
                with open(img_text, 'w') as f:
                    for line in clean_lines:
                        f.write(line + '\n')

    # Step 3. Segment images with MiVOS and mask images
    if not no_mask:
        print('Segmenting images with MiVOS ...')
        msk_path = seg_video(img_path=img_path, MIVOS_PATH=MIVOS_PATH)
        torch.cuda.empty_cache()
        print('Masking images with masks ...')
        msked_path, rename_maskimages = mask_images(img_path, msk_path, no_mask=no_mask)

    
    # Step 4. Process poses and output transforms.json, where the coordinates follow NeRF convention
    json_path = "transforms_optitrack.json" if use_optitrack else "transforms_colmap.json"
    img_dict_params_path="img_dict_params.json"
    if process_poses:
        if os.path.exists(os.path.join(obj_dir, json_path)):
            # delete the previous json file
            os.remove(os.path.join(obj_dir, json_path))

        # RH: if you use optitrack data, we need to update transforms.py accordingly.  remove noisy frames, rename clean frames, and change file format from .jpg to .png
        if use_optitrack:
            print(f"Running optitrack2nerf_invoke")
            optitrack2nerf_invoke(img_path, obj_dir=obj_dir, img_txt_path="images_optitrack.txt" if remove_blur else "images_all.txt", json_path=json_path, img_dict_params_path=img_dict_params_path)

        else:
            print('Running COLMAP ...')
            colmap2nerf_invoke(img_path, img_txt_path="images.txt", json_path=json_path, img_dict_params_path=img_dict_params_path)
        
        if process_gelsight_poses:
            print(f"process_gelesight_poses is set to True. Processing gelsight poses ...")
            gelsight_dict = create_gelsight_dict_from_txt_and_img_dict_params(obj_dir, gelsight_txt_path="gelsight_images_all.txt", img_dict_params_path=img_dict_params_path)
            # Note: Unlike camera poses, we don't have camera intrinsics for gelsight. Therefore, we only save the poses in the .json file.
            gelsight_json_path = "transforms_gelsight.json" if use_optitrack else "transforms_gelsight_colmap.json"
            with open(os.path.join(obj_dir, gelsight_json_path), 'w') as f:
                json.dump(gelsight_dict, f, indent=4)


    # (Optionally) Step 5. Rename masked and unmasked pathes
    if use_masked_images and rename_maskimages:
        if img_path.endswith('/'):
            img_path = img_path[:-1]
        unmsk_path = '/'.join(img_path.split('/')[:-1]) + '/unmasked_images/'
        print('Rename masked and unmasked pathes.')
        if not no_mask:
            os.rename(img_path, unmsk_path)
            os.rename(msked_path, img_path)

    # Step 6. after the first round of running this script,
    # manually filter out the images that have inaccurate mask in the second pass (manually pick the index by visually inspecting the mask folder)
    inaccurate_mask_index_list = [] # [24, 26, 27, 32, 33, 34, 137, 191, 192, 226, 231, 232, 233, 234, 235, 243, 244, 245, 246, 247, 248, 260] # [ 60, 62, 63, 64, 84, 86] 
    if len(inaccurate_mask_index_list) > 0 and remove_inaccurate_mask:
        print(f'Removing inaccurate mask images for frames {inaccurate_mask_index_list} ...')
        inaccurate_mask_dir = os.path.join(obj_dir, 'inaccurate_mask_images')
        inaccurate_mask_image_dir = os.path.join(inaccurate_mask_dir, 'images')
        inaccurate_mask_mask_dir = os.path.join(inaccurate_mask_dir, 'mask')
        inaccurate_mask_overlay_dir = os.path.join(inaccurate_mask_dir, 'overlay')
        for inaccurate_dir in [inaccurate_mask_dir, inaccurate_mask_image_dir, inaccurate_mask_mask_dir, inaccurate_mask_overlay_dir]:
            if not os.path.exists(inaccurate_dir):
                os.makedirs(inaccurate_dir)
            else:
                shutil.rmtree(inaccurate_dir)
                os.makedirs(inaccurate_dir)

        mask_path = os.path.join(obj_dir, 'mask')
        overlay_path = os.path.join(obj_dir, 'overlay')
        for inaccurate_mask_index in inaccurate_mask_index_list:
            shutil.move(os.path.join(img_path, f'{inaccurate_mask_index:05d}.png'), inaccurate_mask_image_dir)
            shutil.move(os.path.join(mask_path, f'{inaccurate_mask_index:05d}.png'), inaccurate_mask_mask_dir)
            shutil.move(os.path.join(overlay_path, f'{inaccurate_mask_index:05d}.png'), inaccurate_mask_overlay_dir)
        # TODO: update .json file too
        # load current json file as a dict
        current_json = json.load(open(os.path.join(obj_dir, json_path)))
        filtered_json_path = os.path.join(obj_dir, 'transforms.json')
        current_frames = current_json['frames']
        filtered_frames = []
        for frame in current_frames:
            if frame['file_path'].split('/')[-1] not in [f'{inaccurate_mask_index:05d}.png' for inaccurate_mask_index in inaccurate_mask_index_list]:
                filtered_frames.append(frame)
        current_json['frames'] = filtered_frames
        json.dump(current_json, open(filtered_json_path, 'w'), indent=4)
        print(f'Filtered json file saved to {filtered_json_path}')

