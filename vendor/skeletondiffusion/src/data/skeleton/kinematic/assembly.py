import numpy as np

from .base import Kinematic


class AssemblyKinematic(Kinematic):
    """
    Assembly hand kinematics with a synthetic root at index 0.
    Real hand joints are indices 1..21 (matching SplineEqNet local hand ordering + 1).
    """

    def __init__(self, num_joints=22, **kwargs):
        super().__init__(**kwargs)
        assert num_joints == 22, "AssemblyKinematic expects 22 joints (synthetic root + 21 hand joints)."

        # Synthetic root + 21 local hand joints from Assembly preprocessing.
        self.joint_dict_orig = {
            0: "GlobalRoot",
            1: "ThumbTip",
            2: "IndexTip",
            3: "MiddleTip",
            4: "RingTip",
            5: "PinkyTip",
            6: "Wrist",
            7: "ThumbBase2",
            8: "ThumbBase1",
            9: "IndexBase3",
            10: "IndexBase2",
            11: "IndexBase1",
            12: "MiddleBase3",
            13: "MiddleBase2",
            14: "MiddleBase1",
            15: "RingBase3",
            16: "RingBase2",
            17: "RingBase1",
            18: "PinkyBase3",
            19: "PinkyBase2",
            20: "PinkyBase1",
            21: "PalmAux",
        }

        # SplineEqNet assembly hand links (shifted by +1 for synthetic root).
        hand_links = [
            (4, 19),
            (3, 16),
            (2, 13),
            (1, 10),
            (19, 18),
            (16, 15),
            (13, 12),
            (10, 9),
            (18, 17),
            (15, 14),
            (12, 11),
            (9, 8),
            (17, 5),
            (14, 5),
            (11, 5),
            (8, 5),
            (0, 7),
            (7, 6),
            (6, 5),
            (20, 5),
            (17, 14),
            (14, 11),
            (11, 8),
        ]
        shifted = [(a + 1, b + 1) for a, b in hand_links]
        shifted.append((0, 6))  # synthetic root to wrist
        self.limbseq = np.array([[min(a, b), max(a, b)] for a, b in shifted], dtype=int)

        self.left_right_limb_list = [True for _ in self.joint_dict_orig]

        if not self.if_consider_hip:
            self.node_dict = self.joint_dict_orig.copy()
            self.node_dict.pop(0)
            self.node_dict = {i: v for i, v in enumerate(list(self.node_dict.values()))}
            self.node_limbseq = np.array(
                [[a - 1, b - 1] for a, b in self.limbseq.tolist() if a != 0 and b != 0],
                dtype=int,
            )
            # Finger chains for angle-based metrics.
            self.limb_angles_idx = [
                [0, 7, 6, 5],
                [1, 10, 9, 8, 5],
                [2, 13, 12, 11, 5],
                [3, 16, 15, 14, 5],
                [4, 19, 18, 17, 5],
            ]
            self.kinchain = [
                [5, 6, 7, 0],      # thumb
                [5, 8, 9, 10, 1],  # index
                [5, 11, 12, 13, 2],# middle
                [5, 14, 15, 16, 3],# ring
                [5, 17, 18, 19, 4],# pinky
            ]
        else:
            self.node_dict = {k: v for k, v in enumerate(list(self.node_hip.values()) + list(self.joint_dict_orig.values())[1:])}
            self.node_limbseq = self.limbseq.copy()
