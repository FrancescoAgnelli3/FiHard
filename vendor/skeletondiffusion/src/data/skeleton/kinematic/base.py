import torch
    
from .utils import plot_matrix, get_adj_matrix


class Kinematic():
    """
    It is a abstract class, needs to be subclassed.
    """
    def __init__(self, if_consider_hip=False, **kwargs):
        self.if_consider_hip = if_consider_hip


        # TO DO define following variables
        # self.joint_dict_orig  = {}
        # self.limbseq = [[0, 1], ...
        #                 ]

        # if if_consider_hip:
        #     self.node_dict = {**Skeleton.node_hip, **{i +len(Skeleton.node_hip): v for i,v in self.joint_dict_orig.items()}}
        #    self.node_limbseq = [[limb[0] -1,  limb[1] -1] for limb in self.limbseq if limb[1] != 0 and limb[0] != 0]
        #       # rememeber to add hip connections!
        # else: 
        #     self.node_dict =  self.joint_dict_orig
        #    self.node_limbseq = self.limbseq
        # self.left_right_limb_list = [False if joint[0] == "L" and joint[1].isupper() else True for joint in list(self.joint_dict_orig.values())]

    # @property
    def parents(self, mode='original'):
        num_joints = self.num_joints if mode == 'original' else self.num_nodes
        parents = [None] * num_joints
        parents[0] = -1
        limbseq = self.limbseq if mode == 'original' else self.node_limbseq
        for tup in limbseq:
            assert tup[0] < tup[1]
            parents[tup[1]] = tup[0]
        return parents
    
    def limb_names(self, mode='original'):
        limbseq = self.limbseq if mode == 'original' else self.node_limbseq
        nlimbs = self.num_joints if mode == 'original' else self.num_nodes
        joint_dict = self.joint_dict_orig if mode == 'original' else self.node_dict
        parents = self.parents(mode=mode)
        return [f"Limb {i}: {joint_dict[i]}-{joint_dict[parents[i]]}" for i in range(1, nlimbs)] 
    
    @property
    def num_joints(self):
        return len(self.joint_dict_orig)

    @property
    def num_nodes(self):
        return len(self.node_dict)
    
    @property
    def left_right_limb(self):
        return self.left_right_limb_list.copy()
    
    @property
    def nodes_type_id(self):
        joint_id_string_wo = []
        for joint_id_string in list(self.node_dict.values()):
            if joint_id_string[0] == 'L' and joint_id_string[1].isupper():
                joint_id_string_wo.append(joint_id_string[1:])
            elif joint_id_string[0] == 'R' and joint_id_string[1].isupper():
                joint_id_string_wo.append(joint_id_string[1:])
            else:
                joint_id_string_wo.append(joint_id_string)
        unique_strings = list(dict.fromkeys(joint_id_string_wo))
        joint_ids = [unique_strings.index(s) for s in joint_id_string_wo]
        return torch.tensor(joint_ids)

    @property
    def adj_matrix(self):
        return get_adj_matrix(limbseq=self.node_limbseq, num_nodes=self.num_nodes)

    
    @property
    def plot_adj(self):
        plot_matrix(self.adj_matrix, list(self.node_dict.values()))
    
    def get_limbseq(self):
        limbseq = self.limbseq if self.if_consider_hip else self.node_limbseq
        return limbseq
    
    def reachability_matrix(self, factor=0.5, stop_at='hips'):
        # stop_at=0
        adj = self.adj_matrix
        reach = torch.zeros_like(adj)
        if stop_at is not None:
            if stop_at == 'hips':
                stop_at = [k  for  k, v in self.node_dict.items() if 'hip' in v.lower()]
            elif stop_at == 'bmn':
                stop_at =  [k  for  k, v in self.node_dict.items() if 'bmn' in v.lower()]
            else: 
                assert 0, "Not implemented"
        def is_reachable(i, j, list_visited_nodes=[]):
            def last_node_reached(k):
                if stop_at is None:
                    return False
                elif isinstance(stop_at, list):
                    return k in stop_at
                return k == stop_at
            
            if adj[i,j] == 1:
                return 1
            else:
                reachable_paths = [0]
                reachable_nodes = ['hip']
                for k in range(self.num_nodes):
                    if adj[i,k] == 1:
                        if last_node_reached(k):
                            return 0
                        if k not in list_visited_nodes:
                            reached = is_reachable(k, j, list_visited_nodes+[k])
                            if reached > 0:
                                if 0 in reachable_paths:
                                    reachable_paths.pop(0)
                                    reachable_nodes = reachable_nodes[1:]
                                reachable_paths.append(reached + 1)
                                reachable_nodes.append(self.node_dict[k])
                return min(reachable_paths)
        
        for i in range(0, reach.shape[0]):
            for j in range(i+1, reach.shape[1]):
                reach[i, j] = factor**(is_reachable(i, j)-1) if is_reachable(i, j) > 0 else 0
                reach[j, i] = reach[i, j]
        return reach
    
    
    def extract_limb_length(self, kpts, mode="metric"):
        limbdist = []
        limbseq = self.limbseq if mode == "metric" else self.node_limbseq
        for l1,l2 in limbseq:
            limbdist.append( (kpts[..., l1, :] - kpts[..., l2, :]).norm(dim=-1))
        return torch.stack(limbdist, dim=-1)
    
    def get_node_num(self, NODE_NAME):
        NODE_NUM = { v: k for k, v in self.joint_dict_orig.items()}[NODE_NAME]
        return NODE_NUM