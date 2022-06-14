
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import torch

from datasets.config import cfg
from datasets.fewshotbatch import read_images, base_transform, get_minibatch

import numpy as np

class roibatchLoader(data.Dataset):
    def __init__(self, roidb, normalize=None, crop_size=400, phase='train', len_dataset=2000, shot=0, step_frame=1):
        self._roidb = roidb
        self.normalize = normalize
        self.phase = phase
        self.step_frame = step_frame
        self.len_dataset = len_dataset
        self.shot = shot
        self.crop_size = crop_size

    def __getitem__(self, index):
        # get the anchor index for current sample index
        if self.phase == 'train':
            len_train = len(self._roidb[2])
            index_query = np.random.randint(0, high=len_train)
            class_index = self._roidb[2][index_query]
            segments = self._roidb[1][class_index]
            len_segment = len(segments)

            query_id = 'query_{:07d}'.format(index)
            index_rand = np.random.randint(0, high=len_segment, size=(1+self.shot))
            query_index = index_rand[0]
            query_segment = segments[query_index]
            query_video = query_segment[0]
            query_subject = query_segment[1]
            query_frame = str(query_segment[2][0])
            query_gt = self._roidb[0][(query_video, query_frame, query_subject)]
            query_box = ((query_gt[0] + query_gt[2]) / 2, (query_gt[1] + query_gt[3]) / 2,
                         query_gt[2] - query_gt[0], query_gt[3] - query_gt[1])
            assert query_box[2] > 0 and query_box[3] > 0
            target = {}
            target['boxes'] = torch.tensor([query_box], dtype=torch.float32)
            target['labels'] = torch.tensor([1], dtype=torch.int64)
            target['gt'] = torch.tensor([query_gt], dtype=torch.float32)

            data_query = read_images('./datasets/ava', query_video, query_frame, num=25, fps=25)
            data_query = base_transform(data_query, size=self.crop_size)
            data_query = torch.from_numpy(data_query).permute(3, 0, 1, 2)

            data_support = []
            if self.shot > 0:
                for i in range(self.shot):
                    support_index = index_rand[i+1]
                    support_segment = segments[support_index]
                    support_video = support_segment[0]
                    support_frame = str(support_segment[2][0])
                    data_tmp = read_images('./datasets/ava', support_video, support_frame, num=25, fps=25)
                    data_tmp = base_transform(data_tmp, size=self.crop_size)
                    data_tmp = torch.from_numpy(data_tmp).permute(3, 0, 1, 2)
                    data_support.append(data_tmp)
            return data_query, data_support, target, query_id
        else:
            segments = self._roidb[index]
            query_id = 'query_{:07d}'.format(index)

            query_segment = segments[0]
            query_video = query_segment['id']
            query_frame = str(query_segment['frames'][0])
            query_gt = query_segment['boxes'][0]
            query_box = ((query_gt[0] + query_gt[2]) / 2, (query_gt[1] + query_gt[3]) / 2,
                         query_gt[2] - query_gt[0], query_gt[3] - query_gt[1])
            assert query_box[2] > 0 and query_box[3] > 0
            target = {}
            target['boxes'] = torch.tensor([query_box], dtype=torch.float32)
            target['labels'] = torch.tensor([1], dtype=torch.int64)
            target['gt'] = torch.tensor([query_gt], dtype=torch.float32)

            data_query = read_images('./datasets/ava', query_video, query_frame, num=25, fps=25)
            data_query = base_transform(data_query, size=self.crop_size)
            data_query = torch.from_numpy(data_query).permute(3, 0, 1, 2)

            data_support = []
            if self.shot > 0:
                for i in range(self.shot):
                    support_segment = segments[i+1]
                    support_video = support_segment['id']
                    support_frame = str(support_segment['frames'][0])
                    data_tmp = read_images('./datasets/ava', support_video, support_frame, num=25, fps=25)
                    data_tmp = base_transform(data_tmp, size=self.crop_size)
                    data_tmp = torch.from_numpy(data_tmp).permute(3, 0, 1, 2)
                    data_support.append(data_tmp)
            return data_query, data_support, target, query_id

    def __len__(self):
        if self.phase == 'train':
            return self.len_dataset
        else:
            if self.len_dataset is None:
                return len(self._roidb)
            else:
                return self.len_dataset



