import os
import copy
import json
import pickle
import numpy as np

FPS = 3
LENGTH = 768
WINS = [LENGTH * 8]
# LENGTH = 192
# WINS = [LENGTH * 32]
min_length = 3
overlap_thresh = 0.7
STEP = LENGTH / 4
META_FILE = './activity_net.v1-3.min.json'
data = json.load(open(META_FILE))
FRAME_DIR = '/home/tao/dataset/v1-3/mkv_train_val_frames_3'
NUM_CLS = 20


def generate_roi(rois, video, start, end, stride, split):
    tmp = {}
    tmp['wins'] = (rois[:, :2] - start) / stride
    tmp['durations'] = tmp['wins'][:, 1] - tmp['wins'][:, 0]
    tmp['gt_classes'] = rois[:, 2]
    tmp['max_classes'] = rois[:, 2]
    tmp['max_overlaps'] = np.ones(len(rois))
    tmp['flipped'] = False
    tmp['video_id'] = video
    tmp['frames'] = np.array([[0, start, end, stride]])
    tmp['bg_name'] = os.path.join('datasets/activitynet13', split, video)
    tmp['fg_name'] = os.path.join('datasets/activitynet13', split, video)
    if not os.path.isfile(os.path.join(FRAME_DIR, split, video, 'image_' + str(end - 1).zfill(5) + '.jpg')):
        print(os.path.join(FRAME_DIR, split, video, 'image_' + str(end - 1).zfill(5) + '.jpg'))
        raise
    return tmp


def generate_roidb(split, segment):
    VIDEO_PATH = os.path.join(FRAME_DIR, split)
    video_list = set(os.listdir(VIDEO_PATH))
    duration = []
    roidb = []
    for vid in segment:
        if vid in video_list:
            length = len(os.listdir(os.path.join(VIDEO_PATH, vid)))
            if length > 768:
                continue
            db = np.array(segment[vid])
            if len(db) == 0:
                continue
            db[:, :2] = db[:, :2] * FPS

            for win in WINS:
                stride = int(win / LENGTH)
                step = int(stride * STEP)

                # Forward Direction
                for start in range(0, max(1, length - win + 1), step):
                    end = min(start + win, length)
                    assert end <= length
                    # No overlap between gt and dt
                    rois = db[np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))]

                    # Remove duration less than min_length
                    if len(rois) > 0:
                        duration = rois[:, 1] - rois[:, 0]
                        rois = rois[duration >= min_length]

                    # Remove overlap(for gt) less than overlap_thresh
                    if len(rois) > 0:
                        time_in_wins = (np.minimum(end, rois[:, 1]) - np.maximum(start, rois[:, 0])) * 1.0
                        overlap = time_in_wins / (rois[:, 1] - rois[:, 0])
                        assert min(overlap) >= 0
                        assert max(overlap) <= 1
                        rois = rois[overlap >= overlap_thresh]

                    # Append data
                    if len(rois) > 0:
                        rois[:, 0] = np.maximum(start, rois[:, 0])
                        rois[:, 1] = np.minimum(end, rois[:, 1])
                        tmp = generate_roi(rois, vid, start, end, stride, split)
                        roidb.append(tmp)
                        if USE_FLIPPED:
                            flipped_tmp = copy.deepcopy(tmp)
                            flipped_tmp['flipped'] = True
                            roidb.append(flipped_tmp)

                # Backward Direction
                for end in range(length, win - 1, - step):
                    start = end - win
                    assert start >= 0
                    rois = db[np.logical_not(np.logical_or(db[:, 0] >= end, db[:, 1] <= start))]

                    # Remove duration less than min_length
                    if len(rois) > 0:
                        duration = rois[:, 1] - rois[:, 0]
                        rois = rois[duration > min_length]

                    # Remove overlap less than overlap_thresh
                    if len(rois) > 0:
                        time_in_wins = (np.minimum(end, rois[:, 1]) - np.maximum(start, rois[:, 0])) * 1.0
                        overlap = time_in_wins / (rois[:, 1] - rois[:, 0])
                        assert min(overlap) >= 0
                        assert max(overlap) <= 1
                        rois = rois[overlap > overlap_thresh]

                    # Append data
                    if len(rois) > 0:
                        rois[:, 0] = np.maximum(start, rois[:, 0])
                        rois[:, 1] = np.minimum(end, rois[:, 1])
                        tmp = generate_roi(rois, vid, start, end, stride, split)
                        roidb.append(tmp)
                    if USE_FLIPPED:
                        flipped_tmp = copy.deepcopy(tmp)
                        flipped_tmp['flipped'] = True
                        roidb.append(flipped_tmp)

    return roidb

def generate_classes(data):
    class_list = []
    for vid, vinfo in data['database'].items():
        for item in vinfo['annotations']:
            class_list.append(item['label'])

    class_list = list(set(class_list))
    class_list = sorted(class_list)
    classes = {'Background': 0}
    for i,cls in enumerate(class_list):
        classes[cls] = i + 1
    return classes

def generate_segment(split, data, classes, frame_dir):
    segment = {}
    VIDEO_PATH = os.path.join(frame_dir, split)
    video_list = set(os.listdir(VIDEO_PATH))
    # get time windows based on video key
    for vid, vinfo in data['database'].items():
        vid_name = [v for v in video_list if vid in v]
        if len(vid_name) == 1:
            if vinfo['subset'] == split:
                # get time windows
                segment[vid] = []
                is_del = False
                for anno in vinfo['annotations']:
                    start_time = anno['segment'][0]
                    end_time = anno['segment'][1]
                    label = classes[anno['label']]
                    if label > NUM_CLS:
                        is_del = True
                    segment[vid].append([start_time, end_time, label])
                if is_del:
                    del segment[vid]
    # sort segments by start_time
    for vid in segment:
        segment[vid].sort(key=lambda x: x[0])

    return segment

# data_pkl = pickle.load(open('./train_data_3fps_flipped_1.pkl', 'rb'))

USE_FLIPPED = True
print('Generate Classes')
classes = generate_classes(data)

print('Generate Training Segments')
train_segment = generate_segment('training', data, classes, FRAME_DIR)

train_roidb = generate_roidb('training', train_segment)
print(len(train_roidb))
print("Save dictionary")
t = 0
pickle.dump(train_roidb, open('train_data_cls{}_flipped.pkl'.format(NUM_CLS), 'wb'), pickle.HIGHEST_PROTOCOL)
