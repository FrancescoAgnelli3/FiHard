from .utils import get_adj_matrix
from .freeman import FreeManKinematic
from .assembly import AssemblyKinematic

def get_kinematic_objclass(dataset_name):
    dataset_type = {'freeman': 'FreeMan', 'assembly': 'Assembly'}[dataset_name.lower()]
    return globals()[dataset_type+"Kinematic"], dataset_type
