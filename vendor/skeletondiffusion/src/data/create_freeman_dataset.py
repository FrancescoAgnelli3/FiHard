import os
import numpy as np
from tqdm import tqdm
import json
import ast
from functools import partial

from .loaders.base.create_dataset_utils import compute_stats, compute_multimodal_gt
from .loaders import FreeManDataset
from .skeleton import get_skeleton_class

IF_NDEBUG = False

OUTPUT_3D = 'data_3d_freeman'

def remove_illposedframes(seq_name, kpts, illposed_at_seq_startend):

    if seq_name in illposed_at_seq_startend:
        s = illposed_at_seq_startend[seq_name]
        if len(s) > 1:
            new_kpts = []
            for i in range(len(s)):
                slice = kpts[s[i][0]:s[i][1]]
                assert np.isnan(slice).sum() == 0
                new_kpts.append(slice)
        else: 
            new_kpts = kpts[s[0][0]:s[0][1]]
            assert np.isnan(new_kpts).sum() == 0
        return new_kpts, [s[i][0] for i in range(len(s))]
    return kpts, 0

def preprocess_kpts(positions):
    assert not np.isnan(positions).any(), "Sequence has nan!"
    # invert joint order such thta hip is first
    # JOINTS_DICT_FREEMAN_17 = {0: "Nose",
    #               1: "LEye", 2: "REye", 3: "LEar", 4: "REar",
    #               5: "LShoulder", 6: "RShoulder", 7: "LElbow", 8: "RElbow",
    #               9: "LWrist", 10: "RWrist",
    #               11: "LHip", 12: "RHip", 13: "LKnee",
    #               14: "RKnee", 15: "LAnkle", 16: "RAnkle"}
    positions = np.concatenate([positions[..., 11:, :], positions[..., :11, :]], axis=-2)
    # add center hip
    positions = np.concatenate([positions[..., 0:1, :] + (positions[..., 1:2, :] - positions[..., 0:1, :])/2, positions], axis=-2)
    positions /= 100 # Meters instead of centimeters
    positions[..., 2] *= -1 # Invert y axis
    return positions

def preprocess_dataset(dataset_folder, annotation_folder, output_path=OUTPUT_3D, task="hmp"):
    dataset_folder = os.path.abspath(dataset_folder)
    if os.path.exists(f"{output_path}.npz"):
        # print('The dataset already exists at', output_path)
        raise Exception(f'The dataset already exists at {output_path}')
        
    print('Converting original FreeMan dataset from', dataset_folder, 'into', output_path, '...')

    with open(os.path.join(dataset_folder, 'ignore_list.txt')) as f:
        ignore_files = set([line.strip() for line in f]) #f.readlines()
    with open(os.path.join(annotation_folder, f"bad_sequences.json"), 'r') as outfile:
        sequences_with_no_good_frames = json.load(outfile)
        ignore_files = ignore_files.union(sequences_with_no_good_frames)
    
    with open(os.path.join(annotation_folder, f"illlposed_slices_idxs.json"), 'r') as outfile:
        illposed_at_seq_startend = ast.literal_eval(json.load(outfile))
        
    sequences = [file.strip('.npy') for file in os.listdir(os.path.join(dataset_folder, 'keypoints3d')) if (file.endswith('.npy') and file.strip('.npy') not in ignore_files)] # file = 'keypoints3d/20220818_857ffafc02_subj22.npy' pattern
    file2action = {}
    with open(os.path.join(annotation_folder, 'seq_actions_labels.txt')) as f:
        for line in f:
            name, label = line.strip().split(',')
            file2action[name] = label
            canonical_name = name.split('_slice')[0]
            if canonical_name not in file2action:
                file2action[canonical_name] = label

    output = {}
    count = 0
    subseq_count = 0
    n_frames = 0
    for seq in tqdm(sequences):
        f = os.path.join(dataset_folder, 'keypoints3d', seq+ '.npy')
        kpts3d = np.load(f, allow_pickle=True)

        if 'keypoints3d_smoothnet32' in kpts3d[0].keys():
            positions = kpts3d[0]['keypoints3d_smoothnet32']
        elif 'keypoints3d_smoothnet' in kpts3d[0].keys():
            positions = kpts3d[0]['keypoints3d_smoothnet'] # change here if you want to load keypoints3d_smoothnet, keypoints3d_smoothnet32 or keypoints3d_optim
        else:
            positions = kpts3d[0]['keypoints3d_optim'] 
        
        clean_seq, s = remove_illposedframes(seq, positions, illposed_at_seq_startend)
        if isinstance(clean_seq, list):
            for i, kpts in enumerate(clean_seq):
                slice_name = f"{seq}_slice{i+1}"
                if seq in file2action:
                    # Too short discarded slices sequences will not be considered
                    kpts = preprocess_kpts(kpts)
                    output[slice_name] = kpts.astype('float32')
                    subseq_count += 1
                    n_frames += kpts.shape[0]
        else:
            kpts = preprocess_kpts(clean_seq)
            output[seq] = kpts.astype('float32')
            count += 1
            n_frames += kpts.shape[0]
            if seq not in file2action:
                print(f"Warning: {seq} not in file2action")

    print(f"Processed total {count+subseq_count} sequences. {count} original sequences or sequences sclied at beginning or end and {subseq_count} sub-sequences obtained by slicing.")
    print(f"Total number of frames: {n_frames}")
    print(f'Saving into "{output_path}"...')
    np.savez_compressed(output_path, positions_3d=output)
    print('Done.')
    return sequences
     
import argparse


# python -m data_loader.create_freeman_dataset
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='FreeMan Process Raw Data')

    parser.add_argument('-gpu', '--gpu', action='store_true', help='Use GPU for processing')

    args = parser.parse_args()

    np.random.seed(0)
    
    actions = "all" 
    
    comp_device = 'cpu' if not args.gpu else 'cuda'
    print("Using device:", comp_device)
    
    batch_size = 128
    original_folder = "../datasets/raw/FreeMan"
    annotations_folder = "../datasets/annotations/FreeMan"
    stride = 1
    dataset_name = 'FreeMan'

    augmentation = 0
    multimodal_threshold = 0.5
    FPS = 30
    
    # precompute data and statistics
    for task in ["hmp"]:
        if_consider_hip = False if task == "hmp" else True
        precomputed_folder_task = f"../datasets/processed/FreeMan/{task}/"
        annotations_folder_task = os.path.join(annotations_folder, task)
        os.makedirs(precomputed_folder_task, exist_ok=True)
        
        if task == "hmp":
            obs_length = 15
            pred_length = 60
        else:
            obs_length = 30
            pred_length = 120

        output_npz_path = os.path.join(precomputed_folder_task, f"{OUTPUT_3D}.npz")

        if os.path.exists(output_npz_path):
            print('The npz dataset already (partially) exists at', output_npz_path)        
        else: 
            sequences = preprocess_dataset(original_folder, annotations_folder_task, output_path=output_npz_path, task=task)            

        FreeManSkeletonVanilla = get_skeleton_class(motion_repr_type="SkeletonVanilla", dataset_name=dataset_name.lower())
        FreeManSkeletonCenterPose = get_skeleton_class(motion_repr_type="SkeletonCenterPose", dataset_name=dataset_name.lower() )
        segments_path = os.path.join(annotations_folder_task, "segments_test.csv")
        print("="*50)
        if os.path.exists(os.path.join(precomputed_folder_task, f"mean_motion_test.txt")):
            print('Values for CMD already (partially) exists at', os.path.join(precomputed_folder_task, f"mean_motion_test.txt"))        
        else: 
            print("Computing values for CMD.")
        # below to compute the average motion per class for the TEST set, for the CMD computation
        def return_skeletons_for_stats_funct(current_task):
            def partialize_skel_class(skel_class):
                return partial(skel_class, obs_length=obs_length, pred_length=pred_length)
            if current_task == "hmp":
                skeletons = [partialize_skel_class(FreeManSkeletonCenterPose), None]
            else: 
                skeletons = [partialize_skel_class(FreeManSkeletonVanilla), partialize_skel_class(H36MSkeletonCenterPose)]
            return skeletons
            return skeletons
        
        compute_stats(precomputed_folder=precomputed_folder_task, 
                        Dataset=partial(FreeManDataset,annotations_folder=annotations_folder_task, actions=actions, stride=stride,  segments_path=segments_path, obs_length=obs_length, pred_length=pred_length),
                        Skeleton=return_skeletons_for_stats_funct(task),
                        if_consider_hip=if_consider_hip, 
                        batch_size=batch_size)
        print("="*50)
        if os.path.exists(os.path.join(precomputed_folder_task, f"mmgt_test.txt")):
            print('Multimodal Ground Truth already (partially) exists at', os.path.join(precomputed_folder_task, f"mmgt_test.txt"))        
        else: 
            print(f"Computing Multimodal Ground Truth for task {task}.")

        skeleton= FreeManSkeletonCenterPose( if_consider_hip=False, obs_length=obs_length, pred_length=pred_length, actions=actions)
        compute_multimodal_gt( Dataset=partial(FreeManDataset, skeleton=skeleton, obs_length=obs_length, pred_length=pred_length, if_consider_hip=False,
                                            actions=actions, stride=stride),
                            multimodal_threshold=multimodal_threshold, segment_path= segments_path, split='test',
                            annotations_folder=annotations_folder_task, precomputed_folder=precomputed_folder_task,
                                )
        
    
    
