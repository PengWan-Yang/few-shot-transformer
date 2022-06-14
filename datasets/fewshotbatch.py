# --------------------------------------------------------
# R-C3D
# Copyright (c) 2017 Boston University
# Licensed under The MIT License [see LICENSE for details]
# Written by Huijuan Xu
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Shiguang Wang
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import cv2
import os
from datasets.config import cfg
from datasets.blob import prep_im_for_blob, video_list_to_blob
import math
import glob


DEBUG = False


def read_images(path, videoname, fid, num=25, fps=25):
    """
    Load images from disk for middel frame fid with given num and fps

    return:
        a list of array with shape (num, H,W,C)
    """

    images = []
    img_path = os.path.join(path, videoname + '/{:05d}/'.format(int(fid)))
    images.extend(_load_images(img_path, num=num, fps=fps, direction='forward'))

    return images


def base_transform(images, size=400, mean=(0, 0, 0)):
    T = len(images)
    w, h, c = images[0].shape
    resized_images = np.zeros((T, size, size, c), dtype=images[0].dtype)
    for i in range(T):
        resized_images[i] = cv2.resize(images[i], (size, size)).astype(np.float32)
    resized_images = resized_images.astype(np.float32)
    resized_images -= mean
    return resized_images.astype(np.float32)

def _load_images(path, num, fps=12, direction='forward'):
    """
    Load images in a folder wiht given num and fps, direction can be either 'forward' or 'backward'
    """

    img_names = glob.glob(os.path.join(path, '*.jpg'))
    if len(img_names) == 0:
        raise ValueError("Image path {} not Found".format(path))
    img_names = sorted(img_names)

    # resampling according to fps
    index = np.linspace(0, len(img_names), fps, endpoint=False, dtype=np.int)
    if direction == 'forward':
        index = index[:num]
    elif direction == 'backward':
        index = index[-num:][::-1]
    else:
        raise ValueError("Not recognized direction", direction)

    images = []
    for idx in index:
        img_name = img_names[idx]
        if os.path.isfile(img_name):
            img = cv2.imread(img_name)
            images.append(img)
        else:
            raise ValueError("Image not found!", img_name)

    return images


def get_minibatch(roidb, crop_size=112, phase='train', step_frame=1, length_support=768):
    """Given a roidb, construct a minibatch sampled from it."""
    num_videos = len(roidb)
    assert num_videos == 1, "Single batch only"
    # Sample random scales to use for each video in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.LENGTH),
                                    size=num_videos)

    # Get the input video blob, formatted for caffe
    video_blob = _get_video_blob(roidb, random_scale_inds, phase=phase, step_frame=step_frame,
                                 length_support=length_support, crop_size=crop_size)
    blobs = {'data': video_blob}

    # gt windows: (x1, x2, cls)
    gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    gt_windows = np.empty((len(gt_inds), 3), dtype=np.float32)
    gt_windows[:, 0:2] = roidb[0]['wins'][gt_inds, :]
    gt_windows[:, -1] = roidb[0]['gt_classes'][gt_inds]
    blobs['gt_windows'] = gt_windows

    return blobs

def _get_video_blob(roidb, scale_inds, crop_size=112, phase='train', step_frame=1, length_support=768):
    """Builds an input blob from the videos in the roidb at the specified
    scales.
    """
    processed_videos = []

    for i, item in enumerate(roidb):
        # just one scale implementated
        video_length = length_support
        video = np.zeros((video_length, crop_size,
                          crop_size, 3))
        j = 0

        if phase == 'train':
            random_idx = [np.random.randint(cfg.TRAIN.FRAME_SIZE[1] - crop_size),
                          np.random.randint(cfg.TRAIN.FRAME_SIZE[0] - crop_size)]
            # TODO: data argumentation
            # image_w, image_h, crop_w, crop_h = cfg.TRAIN.FRAME_SIZE[1], cfg.TRAIN.FRAME_SIZE[0], crop_size, crop_size
            # offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
            # random_idx = offsets[ npr.choice(len(offsets)) ]
        else:
            random_idx = [int((cfg.TRAIN.FRAME_SIZE[1] - crop_size) / 2),
                          int((cfg.TRAIN.FRAME_SIZE[0] - crop_size) / 2)]

        if DEBUG:
            print("offsets: {}, random_idx: {}".format(offsets, random_idx))

        video_info = item['frames'][0]  # for video_info in item['frames']:
        step = step_frame
        prefix = item['fg_name'] if video_info[0] else item['bg_name']

        if cfg.TEMP_SPARSE_SAMPLING:
            if phase == 'train':
                segment_offsets = npr.randint(step, size=len(range(video_info[1], video_info[2], step)))
            else:
                segment_offsets = np.zeros(len(range(video_info[1], video_info[2], step))) + step // 2
        else:
            segment_offsets = np.zeros(len(range(video_info[1], video_info[2], step)))

        times = math.ceil((video_info[2] - video_info[1]) / length_support)
        for i, idx in enumerate(range(video_info[1], video_info[2], times * step)):
            frame_idx = int(segment_offsets[i] + idx + 1)
            frame_path = os.path.join(prefix, 'image_' + str(frame_idx).zfill(5) + '.jpg')
            frame = cv2.imread(frame_path)
            # process the boundary frame
            if frame is None:
                frames = sorted(os.listdir(prefix))
                frame_path = os.path.join(prefix, frames[-1])
                frame = cv2.imread(frame_path)
                # crop to 112 with a random offset
            frame = prep_im_for_blob(frame, cfg.PIXEL_MEANS, tuple(cfg.TRAIN.FRAME_SIZE[::-1]), crop_size,
                                     random_idx)

            if item['flipped']:
                frame = frame[:, ::-1, :]

            if DEBUG:
                cv2.imshow('frame', frame / 255.0)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            video[j] = frame
            j = j + 1

        video[j:video_length] = video[j - 1]

    processed_videos.append(video)
    # Create a blob to hold the input images, dimension trans CLHW
    blob = video_list_to_blob(processed_videos)

    return blob
