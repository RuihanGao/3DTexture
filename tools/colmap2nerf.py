#!/usr/bin/env python3

# origin: https://github.com/NVlabs/instant-ngp/blob/master/scripts/colmap2nerf.py

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
import copy

def parse_args():
	parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

	parser.add_argument("--video_in", default="", help="run ffmpeg first to convert a provided video file into a set of images. uses the video_fps parameter also")
	parser.add_argument("--video_fps", default=2)
	parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")
	parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
	parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
	parser.add_argument("--images", default="images", help="input path to the images")
	parser.add_argument("--text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
	parser.add_argument("--aabb_scale", default=16, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
	parser.add_argument("--skip_early", default=0, help="skip this many images from the start")
	parser.add_argument("--img_txt_path", default="images.txt", help="path to the images txt file")
	parser.add_argument("--json_path", default="transforms.json", help="path to the json file")
	args = parser.parse_args()
	return args

def do_system(arg):
	print(f"==== running: {arg}")
	err=os.system(arg)
	if err:
		print("FATAL: command failed")
		sys.exit(err)


def run_ffmpeg(args):
	if not os.path.isabs(args.images):
		args.images = os.path.join(os.path.dirname(args.video_in), args.images)
	images=args.images
	video=args.video_in
	fps=float(args.video_fps) or 1.0
	print(f"running ffmpeg with input video file={video}, output image folder={images}, fps={fps}.")
	if (input(f"warning! folder '{images}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
		sys.exit(1)
	try:
		shutil.rmtree(images)
	except:
		pass
	do_system(f"mkdir {images}")
	do_system(f"ffmpeg -i {video} -qscale:v 1 -qmin 1 -vf \"fps={fps}\" {images}/%04d.jpg")

def run_colmap(args, warning=True):
	db=args.colmap_db
	images=args.images
	db_noext=str(Path(db).with_suffix(""))

	if args.text=="text":
		args.text=db_noext+"_text"
	text=args.text
	sparse=db_noext+"_sparse"
	print(f"running colmap with:\n\tdb={db}\n\timages={images}\n\tsparse={sparse}\n\ttext={text}")
	if warning:
		if (input(f"warning! folders '{sparse}' and '{text}' will be deleted/replaced. continue? (Y/n)").lower().strip()+"y")[:1] != "y":
			sys.exit(1)
	if os.path.exists(db):
		os.remove(db)
	do_system(f"colmap feature_extractor --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 --database_path {db} --image_path {images}")
	do_system(f"colmap {args.colmap_matcher}_matcher --database_path {db}")
	try:
		shutil.rmtree(sparse)
	except:
		pass
	do_system(f"mkdir {sparse}")
	do_system(f"colmap mapper --database_path {db} --image_path {images} --output_path {sparse}")
	do_system(f"colmap bundle_adjuster --input_path {sparse}/0 --output_path {sparse}/0 --BundleAdjustment.refine_principal_point 1")
	try:
		shutil.rmtree(text)
	except:
		pass
	do_system(f"mkdir {text}")
	do_system(f"colmap model_converter --input_path {sparse}/0 --output_path {text} --output_type TXT")

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da=da/np.linalg.norm(da)
	db=db/np.linalg.norm(db)
	c=np.cross(da,db)
	denom=(np.linalg.norm(c)**2)
	t=ob-oa
	ta=np.linalg.det([t,db,c])/(denom+1e-10)
	tb=np.linalg.det([t,da,c])/(denom+1e-10)
	if ta>0:
		ta=0
	if tb>0:
		tb=0
	return (oa+ta*da+ob+tb*db)*0.5,denom


def obtain_camera_intrinsics_from_txt(cam_txt_folder, cam_txt_path="cameras.txt", aabb_scale=16):
	"""
	Process cameras.txt to obtain camera intrinsics
	"""
	with open(os.path.join(cam_txt_folder, cam_txt_path), "r") as f:
		angle_x=math.pi/2
		cam_count = 0
		for line in f:
			# 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
			# 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
			# 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
			if line[0]=="#":
				continue
			els=line.split(" ")
			w = float(els[2])
			h = float(els[3])
			fl_x = float(els[4])
			fl_y = float(els[4])
			k1 = 0
			k2 = 0
			p1 = 0
			p2 = 0
			cx = w/2
			cy = h/2
			if (els[1]=="SIMPLE_RADIAL"):
				cx = float(els[5])
				cy = float(els[6])
				k1 = float(els[7])
			elif (els[1]=="RADIAL"):
				cx = float(els[5])
				cy = float(els[6])
				k1 = float(els[7])
				k2 = float(els[8])
			elif (els[1]=="OPENCV"):
				fl_y = float(els[5])
				cx = float(els[6])
				cy = float(els[7])
				k1 = float(els[8])
				k2 = float(els[9])
				p1 = float(els[10])
				p2 = float(els[11])
			elif (els[1]=="REALSENSE"):
				# Added for RealSense camera data
				fl_y = float(els[5])
				cx = float(els[6])
				cy = float(els[7])
			else:
				print("unknown camera model ", els[1])
			cam_count += 1
			# fl = 0.5 * w / tan(0.5 * angle_x);
			angle_x= math.atan(w/(fl_x*2))*2
			angle_y= math.atan(h/(fl_y*2))*2
			fovx=angle_x*180/math.pi
			fovy=angle_y*180/math.pi
	assert cam_count == 1, "Only one camera is supported for now."
	print(f"Find {cam_count} cameras.\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")
	cam_intrinsics={
		"camera_angle_x":angle_x,
		"camera_angle_y":angle_y,
		"fl_x":fl_x,
		"fl_y":fl_y,
		"k1":k1,
		"k2":k2,
		"p1":p1,
		"p2":p2,
		"cx":cx,
		"cy":cy,
		"w":w,
		"h":h,
		"aabb_scale":aabb_scale,
	}
	return cam_intrinsics


def obtain_img_dict_from_txt(img_path, img_txt_folder, img_txt_path="images.txt", num_line_per_frame=2, skip_early=0, return_img_dict_params=False):
	"""
	Process images.txt to obtain camera poses
	
	Args:
		img_txt_folder (str): path to the folder containing images.txt
		img_txt_path (str): path to images.txt
		num_line_per_frame (int): number of lines per frame in images.txt. Set to 2 for colmap output and 1 for optitrack output. Colmap has an additional line for 3D feature points.
	
	Returns:
		cam_poses (list): list of camera poses
	
	"""
	img_dict = {'frames': []}

	# compute the transformation matrix
	with open(os.path.join(img_txt_folder, img_txt_path), "r") as f:
		i=0
		bottom = np.array([0,0,0,1.]).reshape([1,4])
		up=np.zeros(3)
		for line in f:
			line=line.strip()
			if line[0]=="#":
				continue
			i=i+1
			if i < skip_early*num_line_per_frame:
				continue
			if  (i-1)%num_line_per_frame==0 :
				elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9 is filename
				filename = elems[9].split('/')[-1]
				#name = str(PurePosixPath(Path(img_path, elems[9])))
				# why is this requireing a relitive path while using ^
				image_rel = os.path.relpath(img_path)
				name = str(f"./images/{filename}")
				print(filename)
				b=sharpness(os.path.join(image_rel, filename))
				print(name, "sharpness=",b)
				image_id = int(elems[0])
				qvec = np.array(tuple(map(float, elems[1:5])))
				tvec = np.array(tuple(map(float, elems[5:8])))
				R = qvec2rotmat(-qvec)
				t = tvec.reshape([3,1])
				m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
				c2w = np.linalg.inv(m)
				# flip the y and z axis
				c2w[0:3,2] *= -1 
				c2w[0:3,1] *= -1
				# swap y and z
				c2w=c2w[[1,0,2,3],:]
				c2w[2,:] *= -1 # flip whole world upside down

				up += c2w[0:3,1]

				frame={"file_path":name,"sharpness":b,"transform_matrix": c2w}
				img_dict["frames"].append(frame)
	
	nframes = len(img_dict["frames"])
	img_dict_params = {} # save necessary parameters to reproduce the transforms for gelsight poses

	# rotate up vector to be z-axis [0,0,1]
	up = up / np.linalg.norm(up)
	print("find the original up vector was: ", up)
	R=rotmat(up,[0,0,1]) 
	R=np.pad(R,[0,1])
	R[-1,-1]=1
	print(f"rotate up vector to be z-axis [0,0,1]...")
	img_dict_params["R"] = R.tolist()
	for f in img_dict["frames"]:
		f["transform_matrix"]=np.matmul(R,f["transform_matrix"])

	# find a central point the cameras are all looking at and center all camera poses around it
	print("computing center of attention...")
	totw=0
	totp=[0,0,0]
	for f in img_dict["frames"]:
		mf=f["transform_matrix"][0:3,:]
		for g in img_dict["frames"]:
			mg=g["transform_matrix"][0:3,:]
			p,w=closest_point_2_lines(mf[:,3],mf[:,2],mg[:,3],mg[:,2])
			if w>0.01:
				totp+=p*w
				totw+=w
	totp/=totw
	print("find the center of attention: ", totp)
	print(f"center all camera poses to the center of attention...")
	img_dict_params["totp"] = totp.tolist()
	for f in img_dict["frames"]:
		f["transform_matrix"][0:3,3]-=totp

	# scale all camera poses to "nerf sized"
	avglen=0.
	for f in img_dict["frames"]:
		avglen+=np.linalg.norm(f["transform_matrix"][0:3,3])
	avglen/=nframes
	print("find avg camera distance from origin ", avglen)
	print(f"scale all camera poses to 'nerf sized'...")
	img_dict_params["avglen"] = avglen
	for f in img_dict["frames"]:
		f["transform_matrix"][0:3,3]*=4./avglen

	# convert the transform matrix to list
	for f in img_dict["frames"]:
		f["transform_matrix"]=f["transform_matrix"].tolist()
	print(f"Finish processing {nframes} frames in total in the img_dict.")

	if return_img_dict_params:
		return img_dict, img_dict_params
	return img_dict


def create_gelsight_dict_from_txt_and_img_dict_params(gelsight_txt_folder, gelsight_txt_path="gelsight_images_all.txt", num_line_per_frame=1, img_dict_params_path=None):
	"""
	Given gelsight poses in txt file and transforms done for img_dict, apply the same transforms to gelsight poses and return a gelsight_dict.

	Required parameters to reproduce the transforms (stored in img_dict_params):
	* rotate up vector to be z-axis [0,0,1] - R
	* center of attention - totp
	* scale all camera poses to "nerf sized" - avglen

	"""
	# load img_dict_params
	assert img_dict_params_path is not None, "img_dict_params_path is required to reproduce the transforms."
	img_dict_params_path = os.path.join(gelsight_txt_folder, img_dict_params_path)
	print(f"check img_dict_params_path {img_dict_params_path}")
	with open(img_dict_params_path, "r") as f:
		img_dict_params = json.load(f)
	for k, v in img_dict_params.items():
		if isinstance(v, list):
			img_dict_params[k] = np.array(v)
	
	# create gelsight_dict
	gelsight_dict = {'frames': []}
	# compute the transformation matrix to get the c2w matrix
	verbose = False
	with open(os.path.join(gelsight_txt_folder, gelsight_txt_path), "r") as f:
		i = 0
		bottom = np.array([0,0,0,1.]).reshape([1,4])
		for line in f:
			line = line.strip()
			if line[0] == "#":
				continue
			i = i + 1
			if (i-1) % num_line_per_frame == 0:
				elems = line.split(" ") # 1-4 is quat, 5-7 is trans, 9 is filename
				filename = elems[9].split('/')[-1]
				# image_rel = os.path.relpath(gelsight_img_path)
				name = str(f"./gelsight_images/{filename}")
				qvec = np.array(tuple(map(float, elems[1:5])))
				tvec = np.array(tuple(map(float, elems[5:8])))
				R = qvec2rotmat(-qvec)
				t = tvec.reshape([3,1])
				m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
				c2w = np.linalg.inv(m)
				# flip the y and z axis
				c2w[0:3,2] *= -1 
				c2w[0:3,1] *= -1
				# swap y and z
				c2w=c2w[[1,0,2,3],:]
				c2w[2,:] *= -1 # flip whole world upside down
				frame={"file_path": name, "transform_matrix": c2w}
				gelsight_dict["frames"].append(frame)




	nframes = len(gelsight_dict["frames"])

	# reproduce the transforms done for "nerf sized" camera poses

	# rotate up vector to be z-axis [0,0,1]
	for f in gelsight_dict["frames"]:
		f["transform_matrix"]=np.matmul(img_dict_params["R"], f["transform_matrix"])

	# find a central point the cameras are all looking at and center all camera poses around it
	for f in gelsight_dict["frames"]:
		f["transform_matrix"][0:3,3] -= img_dict_params["totp"]

	# scale all camera poses to "nerf sized"
	for f in gelsight_dict["frames"]:
		f["transform_matrix"][0:3,3] *= 4./img_dict_params["avglen"]
	
	# convert the transform matrix to list
	for f in gelsight_dict["frames"]:
		f["transform_matrix"]=f["transform_matrix"].tolist()
	print(f"Finish processing {nframes} frames in total in the gelsight_dict.")
	return gelsight_dict


def colmap2nerf_invoke(img_path, img_txt_path="images.txt", json_path="transforms.json", img_dict_params_path="img_dict_params.json"):
	args = parse_args()
	img_path = img_path[:-1] if img_path.endswith('/') else img_path
	sv_path = '/'.join(img_path.split('/')[:-1])
	args.images = img_path
	args.colmap_matcher = "exhaustive"
	args.run_colmap = True
	args.text = sv_path + '/' + args.text
	args.colmap_db = sv_path + '/' + args.colmap_db


	img_path=args.images
	TEXT_FOLDER=sv_path + '/colmap_text'
	OUT_PATH= os.path.join(sv_path, json_path)

	# Generate colmap_sparse/, colmap_text/ and colmap.db
	if os.path.exists(OUT_PATH):
		return
	if args.video_in != "":
		run_ffmpeg(args)
	if args.run_colmap and not os.path.exists(os.path.join(TEXT_FOLDER,"cameras.txt")):
		run_colmap(args, warning=False)
	
	# Generate transforms.json
	print(f"outputting to {OUT_PATH}...")
	#Obtain camera intrinsics
	cam_intrinsics = obtain_camera_intrinsics_from_txt(TEXT_FOLDER, aabb_scale=int(args.aabb_scale))
	# Create output dictionary
	out = copy.deepcopy(cam_intrinsics)
	# Obtain image dictionary
	img_dict, img_dict_params = obtain_img_dict_from_txt(img_path, TEXT_FOLDER, img_txt_path, num_line_per_frame=2, skip_early=int(args.skip_early), return_img_dict_params=True)
	out["frames"] = img_dict["frames"]
	# Write json file
	print(f"Writing json file to {OUT_PATH}")
	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)
	
	# Save img_dict_params
	img_dict_params_path = os.path.join(sv_path, img_dict_params_path)
	with open(img_dict_params_path, "w") as outfile:
		json.dump(img_dict_params, outfile, indent=2)


def optitrack2nerf_invoke(img_path, obj_dir, img_txt_path="images.txt", json_path="transforms.json", img_dict_params_path="img_dict_params.json"):
    
	args = parse_args()
	OUT_PATH = os.path.join(obj_dir, json_path)
	img_path = os.path.join(obj_dir, 'images')

	print(f"outputting to {OUT_PATH}...")
	# Obtain camera intrinsics
	cam_intrinsics = obtain_camera_intrinsics_from_txt(obj_dir, aabb_scale=int(args.aabb_scale))
	# Create output dictionary
	out = copy.deepcopy(cam_intrinsics)
	# Obtain image dictionary
	img_dict, img_dict_params = obtain_img_dict_from_txt(img_path, obj_dir, img_txt_path, num_line_per_frame=1, skip_early=int(args.skip_early), return_img_dict_params=True)
	out["frames"] = img_dict["frames"]
	# Write json file
	print(f"Writing json file to {OUT_PATH}")
	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)
	
	# Save img_dict_params
	img_dict_params_path = os.path.join(obj_dir, img_dict_params_path)
	with open(img_dict_params_path, "w") as outfile:
		json.dump(img_dict_params, outfile, indent=2)




if __name__ == "__main__":
	args = parse_args()
	if args.video_in != "":
		run_ffmpeg(args)
	if args.run_colmap:
		run_colmap(args)
	AABB_SCALE=int(args.aabb_scale)
	SKIP_EARLY=int(args.skip_early)
	img_path=args.images
	TEXT_FOLDER=args.text
	OUT_PATH=args.json_path
	print(f"outputting to {OUT_PATH}...")
	with open(os.path.join(TEXT_FOLDER,"cameras.txt"), "r") as f:
		angle_x=math.pi/2
		for line in f:
			# 1 SIMPLE_RADIAL 2048 1536 1580.46 1024 768 0.0045691
			# 1 OPENCV 3840 2160 3178.27 3182.09 1920 1080 0.159668 -0.231286 -0.00123982 0.00272224
			# 1 RADIAL 1920 1080 1665.1 960 540 0.0672856 -0.0761443
			if line[0]=="#":
				continue
			els=line.split(" ")
			w = float(els[2])
			h = float(els[3])
			fl_x = float(els[4])
			fl_y = float(els[4])
			k1 = 0
			k2 = 0
			p1 = 0
			p2 = 0
			cx = w/2
			cy = h/2
			if (els[1]=="SIMPLE_RADIAL"):
				cx = float(els[5])
				cy = float(els[6])
				k1 = float(els[7])
			elif (els[1]=="RADIAL"):
				cx = float(els[5])
				cy = float(els[6])
				k1 = float(els[7])
				k2 = float(els[8])
			elif (els[1]=="OPENCV"):
				fl_y = float(els[5])
				cx = float(els[6])
				cy = float(els[7])
				k1 = float(els[8])
				k2 = float(els[9])
				p1 = float(els[10])
				p2 = float(els[11])
			else:
				print("unknown camera model ", els[1])
			# fl = 0.5 * w / tan(0.5 * angle_x);
			angle_x= math.atan(w/(fl_x*2))*2
			angle_y= math.atan(h/(fl_y*2))*2
			fovx=angle_x*180/math.pi
			fovy=angle_y*180/math.pi

	print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

	with open(os.path.join(TEXT_FOLDER, args.img_txt_path), "r") as f:
		i=0
		bottom = np.array([0,0,0,1.]).reshape([1,4])
		out={
			"camera_angle_x":angle_x,
			"camera_angle_y":angle_y,
			"fl_x":fl_x,
			"fl_y":fl_y,
			"k1":k1,
			"k2":k2,
			"p1":p1,
			"p2":p2,
			"cx":cx,
			"cy":cy,
			"w":w,
			"h":h,
			"aabb_scale":AABB_SCALE,"frames":[]
		}

		up=np.zeros(3)
		for line in f:
			line=line.strip()
			if line[0]=="#":
				continue
			i=i+1
			if i < SKIP_EARLY*2:
				continue
			if  i%2==1 :
				elems=line.split(" ") # 1-4 is quat, 5-7 is trans, 9 is filename
				#name = str(PurePosixPath(Path(img_path, elems[9])))
				# why is this requireing a relitive path while using ^
				image_rel = os.path.relpath(img_path)
				name = str(f"./{image_rel}/{elems[9]}")
				b=sharpness(name)
				print(name, "sharpness=",b)
				image_id = int(elems[0])
				qvec = np.array(tuple(map(float, elems[1:5])))
				tvec = np.array(tuple(map(float, elems[5:8])))
				R = qvec2rotmat(-qvec)
				t = tvec.reshape([3,1])
				m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
				c2w = np.linalg.inv(m)
				c2w[0:3,2] *= -1 # flip the y and z axis
				c2w[0:3,1] *= -1
				c2w=c2w[[1,0,2,3],:] # swap y and z
				c2w[2,:] *= -1 # flip whole world upside down

				up += c2w[0:3,1]

				frame={"file_path":name,"sharpness":b,"transform_matrix": c2w}
				out["frames"].append(frame)
	nframes = len(out["frames"])
	up = up / np.linalg.norm(up)
	print("up vector was ", up)
	R=rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
	R=np.pad(R,[0,1])
	R[-1,-1]=1


	for f in out["frames"]:
		f["transform_matrix"]=np.matmul(R,f["transform_matrix"]) # rotate up to be the z axis

	# find a central point they are all looking at
	print("computing center of attention...")
	totw=0
	totp=[0,0,0]
	for f in out["frames"]:
		mf=f["transform_matrix"][0:3,:]
		for g in out["frames"]:
			mg=g["transform_matrix"][0:3,:]
			p,w=closest_point_2_lines(mf[:,3],mf[:,2],mg[:,3],mg[:,2])
			if w>0.01:
				totp+=p*w
				totw+=w
	totp/=totw
	print(totp) # the cameras are looking at totp
	for f in out["frames"]:
		f["transform_matrix"][0:3,3]-=totp

	avglen=0.
	for f in out["frames"]:
		avglen+=np.linalg.norm(f["transform_matrix"][0:3,3])
	avglen/=nframes
	print("avg camera distance from origin ", avglen)
	for f in out["frames"]:
		f["transform_matrix"][0:3,3]*=4./avglen     # scale to "nerf sized"

	for f in out["frames"]:
		f["transform_matrix"]=f["transform_matrix"].tolist()
	print(nframes,"frames")
	print(f"writing {OUT_PATH}")
	with open(OUT_PATH, "w") as outfile:
		json.dump(out, outfile, indent=2)