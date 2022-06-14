

import pickle

# modify validation data

_few_shot_pickle_file = 'few_shot_test_data.pkl'
_few_shot_file = open(_few_shot_pickle_file, 'rb')
data_few_shot = pickle.load(_few_shot_file)

_few_shot_pickle_file = 'few_shot_val_data.pkl'
_few_shot_file = open(_few_shot_pickle_file, 'rb')
data_val = pickle.load(_few_shot_file)

_few_shot_pickle_file = 'few_shot_train_data.pkl'
_few_shot_file = open(_few_shot_pickle_file, 'rb')
data_train = pickle.load(_few_shot_file)

raise 1

for _list in data_few_shot:
    for _video in _list:
        _video['fg_name'] = _video['fg_name'].replace('/home/tao/dataset/v1-3/train_val_frames_3',
                                                      'datasets/activitynet13')
        _video['bg_name'] = _video['bg_name'].replace('/home/tao/dataset/v1-3/train_val_frames_3',
                                                      'datasets/activitynet13')
pickle.dump(data_few_shot, open(_few_shot_pickle_file, "wb"))
print("done")

# modify testing data

_few_shot_pickle_file = 'few_shot_test_data.pkl'
_few_shot_file = open(_few_shot_pickle_file, 'rb')
data_few_shot = pickle.load(_few_shot_file)

for _list in data_few_shot:
    for _video in _list:
        _video['fg_name'] = _video['fg_name'].replace('/home/tao/dataset/v1-3/train_val_frames_3',
                                                      'datasets/activitynet13')
        _video['bg_name'] = _video['bg_name'].replace('/home/tao/dataset/v1-3/train_val_frames_3',
                                                      'datasets/activitynet13')
pickle.dump(data_few_shot, open(_few_shot_pickle_file, "wb"))
print("done")

# modify training data

_few_shot_pickle_file = 'few_shot_train_data.pkl'
_few_shot_file = open(_few_shot_pickle_file, 'rb')
data_few_shot = pickle.load(_few_shot_file)

index = 0
for k, _list in data_few_shot.items():
    for _video in _list:
        _video['video_id'] = "query_{:0>5d}".format(index)
        _video['fg_name'] = _video['fg_name'].replace('dataset/activitynet13/train_val_frames_3',
                                                      'datasets/activitynet13')
        _video['bg_name'] = _video['bg_name'].replace('dataset/activitynet13/train_val_frames_3',
                                                      'datasets/activitynet13')
        index = index + 1

pickle.dump(data_few_shot, open(_few_shot_pickle_file, "wb"))
print("done")
