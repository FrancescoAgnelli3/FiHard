

def center_kpts_around_hip(kpts, hip_idx=0):
    # hip is index 0 in our canonical layout
    center = kpts[..., hip_idx, :].unsqueeze(-2) #(n_frames, n_joints, 3)
    centered_kpts = kpts - center
    # centered_kpts = centered_kpts[..., 1:, :] # keep hip joint for compatibility with rest of code
    return centered_kpts, center

def center_kpts_around_hip_and_drop_root(kpts, hip_idx=0):
    centered_kpts, center = center_kpts_around_hip(kpts.clone(), hip_idx)
    centered_kpts = centered_kpts[..., 1:, :]
    return centered_kpts
