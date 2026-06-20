from .motion import get_motion_representation_objclass
from .kinematic import get_kinematic_objclass


def get_skeleton_class(**kwargs):
    """
    Returns a skeleton class based on the provided parameters.
    
    Parameters:
        **kwargs: Dictionary containing the parameters to determine the skeleton class.
            - motion_repr_type: Type of motion representation (e.g., 'CenterPose', 'Vanilla').
            - dataset_name: Name of the dataset (e.g., 'Assembly', 'FreeMan').
    
    Returns:
        CompoundSkeletonClass: A class that combines kinematic and motion representation functionalities.
    """
    motion_repr_class = get_motion_representation_objclass(kwargs["motion_repr_type"])
    kinematic_class, dataset_name = get_kinematic_objclass(kwargs["dataset_name"])
    new_class_name = dataset_name + kwargs["motion_repr_type"]
    def init_both_classes(self, *args, **kwargs):
        # Initialize the kinematic class
        kinematic_class.__init__(self, *args, **kwargs)
        # Initialize the motion representation class
        motion_repr_class.__init__(self, *args, **kwargs)

    CompoundSkeletonClass = type(new_class_name, (kinematic_class, motion_repr_class), {"__init__": init_both_classes,})
    #### THe previous line is creating a class similar to the following and returning the constructor/Class name
        # class AssemblySkeletonCenterPose(AssemblyKinematic, SkeletonCenterPose):
        # def __init__(self, **kwargs):
        #     init_both_classes(self, **kwargs) 

    return CompoundSkeletonClass

def create_skeleton(**kwargs):
    CompoundSkeletonClass = get_skeleton_class(**kwargs)
   
    return CompoundSkeletonClass(**kwargs)
