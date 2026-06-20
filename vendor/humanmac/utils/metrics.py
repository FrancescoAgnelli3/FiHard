from scipy.spatial.distance import pdist
import numpy as np
import torch
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from common.evaluation import compute_all_metrics_single as _common_compute_all_metrics_single  # type: ignore

"""metrics"""


def compute_all_metrics(pred, gt, gt_multi):
    """
    calculate all metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        gt: ground truth, shape as [1, t_pred, 3 * joints_num]
        gt_multi: multi-modal ground truth, shape as [multi_modal, t_pred, 3 * joints_num]

    Returns:
        diversity, ade, fde, mmade, mmfde
    """
    return _common_compute_all_metrics_single(pred, gt, gt_multi)
