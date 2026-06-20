import numpy as np
import os
import pandas as pd
import torch

from .base import MotionDataset


class FreeManDataset(MotionDataset):
    def __init__(self, *args, actions='all', annotations_folder=None, **kwargs): 
                # annotations_folder, split,
                # precomputed_folder, obs_length, pred_length, skeleton: FreeManSkeleton, 
                # stride=1, augmentation=0, segments_path=None,
                # if_consider_hip=False, dtype='float32', 
                # da_mirroring=0.0, da_rotations=0.0, **kwargs
                
        self.annotations_folder = annotations_folder
        self.FPS = 30
        self.actions = actions
        
        self.dict_indices = {} # dict_indices[seq_name] indicated idx where subject-action annotations start.
        self.mm_indces = None
        self.metadata_class_idx = 0 # 0: action, 1: seq_name --> action is the class used for metrics computation
        # following variables will be created in the next function calls
        # self.idx_to_class = []
        # self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}
        # self.mean_motion_per_class = [0.004533339312024582, 0.005071772030221925, 0.003968115058494981, 0.00592599384929542, 0.003590651675618232, 0.004194935839372698, 0.005625120976387903, 0.0024796492124910586, 0.0035406092427418797, 0.003602172245980421, 0.004347639393585013, 0.004222595821256223, 0.007537553520400006, 0.007066049169369122, 0.006754175094952483]

        super().__init__(*args, actions=actions, **kwargs)
        assert (self.actions is not None) or self.segments_path is not None

    def extract_action_label(self, extra):
        return extra['metadata'][0]    

    
    def _prepare_data(self, num_workers=8):
        if self.segments_path:
            self.segments, self.segment_idx_to_metadata = self._load_annotations_and_segments(self.segments_path, num_workers=num_workers)
            self.stride = 1
            self.augmentation = 0
        else:
            print("During traiing we should not have access to the annotation folder. Everything we need during training sould be in the precomputed folder. OR given as an extra path like the segments files.")
            with open(os.path.join(self.annotations_folder, f'{self.split}.txt')) as f:
                split_seqs = [line.strip() for line in f] #f.readlines()
                assert len(split_seqs) == len(set(split_seqs)), "Some sequences are repeated in the split file."
            self.annotations = self._read_all_annotations(self.actions, split_seqs)
            self.segments, self.segment_idx_to_metadata = self._generate_segments()
            
    def _read_all_annotations(self, actions, seqs):
        # Note that seqs names do not include 'the slice subfix'
        preprocessed_path = os.path.join(self.precomputed_folder, 'data_3d_freeman.npz')
            
        data_o = np.load(preprocessed_path, allow_pickle=True)['positions_3d'].item()
        
        # filter sequences by name: remove sequences not in seqs
        data_f = {key: data_o[key] for key in seqs}
        
        file2action = {}
        with open(os.path.join(self.annotations_folder, 'seq_actions_labels.txt')) as f:
            for line in f:
                name, label = line.strip().split(',')
                # if name in seqs and (label in actions or actions == 'all'):
                file2action[name] = label
        
        # filter actions dict to keep only the seqs in data_f        
        file2action = {key: file2action[key] for key in file2action if key in data_f}
        
        # if we do not want to keep all actions
        if actions != 'all':
            assert isinstance(actions, list), "actions must be a list"
            # filter actions
            file2action = {key: file2action[key] for key in file2action if file2action[key] in actions}
            data_f = {key: data_f[key] for key in data_f if key in file2action}
            
        action_list = {action: None for action in list(file2action.values())}

        data = {}
        self.seq2action = file2action
        self.idx_to_class = list(dict.fromkeys(action_list))
        self.countclassoccurrences = {a: 0 for a in action_list}
        counter = 0
        # we build the feature vectors for each participant and action
        for seq_name in list(data_f.keys()):
            action = self.seq2action[seq_name]
            self.dict_indices[seq_name] = counter
            counter += 1
            kpts = data_f[seq_name]
            data[seq_name] = kpts
            self.countclassoccurrences[action] += 1
            assert not np.isnan(kpts).any()

        self.data = data
        self.class_to_idx = {v: k for k, v in enumerate(self.idx_to_class)}

        anns_all = []
        self.clip_idx_to_metadata = []
        for seq_name in self.data:
            self.clip_idx_to_metadata.append((self.seq2action[seq_name], seq_name))
            anns_all.append(self.data[seq_name].astype(self.dtype)) # participants axis expanded
        
        self._generate_statistics_full(anns_all)

        return anns_all
    
    def _load_annotations_and_segments(self, segments_path, num_workers=8):
        assert os.path.exists(segments_path), "The path specified for segments does not exist: %s" % segments_path
        df = pd.read_csv(segments_path)
        seqs = list(df["name"].unique())
        self.annotations = self._read_all_annotations(self.actions, seqs)
        segments = [(self.dict_indices[row["name"]], 
                    int(row["init"]),
                    int(row["pred_end"])) 
                        for i, row in df.iterrows()]

        segment_idx_to_metadata = [(self.seq2action[row["name"]], row["name"]) for i, row in df.iterrows()]
                        
        #print(segments)
        #print(self.dict_indices)
        return segments, segment_idx_to_metadata
           
