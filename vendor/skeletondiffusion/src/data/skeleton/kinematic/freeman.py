import torch
import numpy as np
from .base import Kinematic

class FreeManKinematic(Kinematic):
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        self.joint_dict_orig  = {0: 'GlobalRoot', 1: 'LHip', 2: 'RHip', 
                                    3: 'LKnee', 4: 'RKnee', 5: 'LAnkle', 6: 'RAnkle', 
                                    7: 'Nose', 8: 'LEye', 9: 'REye', 10: 'LEar', 11: 'REar', 
                                    12: 'LShoulder', 13: 'RShoulder', 14: 'LElbow', 15: 'RElbow', 16: 'LWrist', 17: 'RWrist'}
        self.limbseq = [[0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6], 
                        [0, 7], [7, 8], [7, 9], [8, 10], [9, 11], 
                        [7, 12], [7, 13], [12, 14], [13, 15], [14, 16], [15, 17]]
        assert self.num_joints ==18

        self.left_right_limb_list = [False if joint[0] == "L" and joint[1].isupper() else True for joint in list(self.joint_dict_orig.values())]


        if not self.if_consider_hip:
            self.node_dict = self.joint_dict_orig.copy()
            self.node_dict.pop(0)
            self.node_dict = {i:v for i,v in enumerate(list(self.node_dict.values()))}
            node_dict_reversed = {v:i for i,v in self.node_dict.items()}
            self.node_limbseq = [[node_dict_reversed['RHip'], node_dict_reversed['LHip']], 
                                    [node_dict_reversed['RHip'], node_dict_reversed['Nose']], 
                                    [node_dict_reversed['LHip'], node_dict_reversed['Nose']],
                                    *[[limb[0] -1,  limb[1] -1] for limb in self.limbseq if limb[1] != 0 and limb[0] != 0]]
            self.limb_angles_idx = [[0,1,7,9],  [0,4,6], [1,8,10], [3,5], [2,11,13, 15], [1,12,14,16]]
            self.kinchain = [[0,6,7,9,10,8], # hip to head 
                             [6,11,13,15], # left arm
                             [6,12,14,16], # right arm
                            [0,2,4], # left leg
                            [1,3,5], # right leg
                            [0,1], [7,8], # hip triangle
                            ]

        else: 
            self.node_dict = {k:v for k,v in enumerate(list(self.node_hip.values())+list(self.joint_dict_orig.values())[1:])}
            self.node_limbseq = self.limbseq.copy()
        self.limbseq = np.array(self.limbseq)
