import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList # , SequenceDepth
from pytracking.utils.load_text import load_text
import os


class CDTBDataset(BaseDataset):
    """ CDTB dataset.

    Publication:
        CDTB: A Color and Depth Visual Object Tracking and Benchmark
        Alan Lukezic, Ugur Kart, Jani Kapyla, Ahmed Durmush, Joni-Kristian Kamarainen, Jiri Matas, Matej Kristan.
        http://prints.vicos.si/publications/374

    Download dataset from VOT-RGBD2020 challenge
    """
    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        if split == 'test' or split == 'val':
            self.base_path = os.path.join(self.env_settings.cdtb_path, split)
        else:
            self.base_path = os.path.join(self.env_settings.cdtb_path)

        self.sequence_list = self._get_sequence_list(split)
        self.split = split

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}/color/'.format(self.base_path, sequence_name)
        depths_path = '{}/{}/depth/'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        depth_list = [depth for depth in os.listdir(depths_path) if depth.endswith(".png")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        depth_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]
        depths_list = [os.path.join(depths_path, depth) for depth in depth_list]

        # return SequenceDepth(sequence_name, frames_list, depths_list, 'cdtb', ground_truth_rect.reshape(-1, 4))
        return Sequence(sequence_name, frames_list, 'cdtb', ground_truth_rect.reshape(-1, 4), depths=depths_list)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        print(self.base_path)
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/val.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list
