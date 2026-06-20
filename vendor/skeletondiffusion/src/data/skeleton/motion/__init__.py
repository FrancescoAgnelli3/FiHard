from .utils import center_kpts_around_hip_and_drop_root
from .centerpose import SkeletonCenterPose
from .rescalepose import  SkeletonRescalePose
from .dct import SkeletonDiscreteCosineTransform
from .base import Skeleton as SkeletonVanilla


def get_motion_representation_objclass(motion_repr_type):
    return globals()[motion_repr_type]