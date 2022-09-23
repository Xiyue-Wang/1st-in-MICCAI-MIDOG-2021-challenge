"""
        Non maxima suppression on WSI results

        Uses a kdTree to improve speed. This will only work reasonably for same-sized objects.

        Marc Aubreville, Pattern Recognition Lab, FAU Erlangen-Nürnberg, 2019
"""

import numpy as np
import torch
from sklearn.neighbors import KDTree

from util.object_detection_helper import IoU_values


def non_max_suppression_by_distance(boxes, scores, radius: float = 25, det_thres=None):
    if det_thres is not None:  # perform thresholding
        to_keep = scores > det_thres
        boxes = boxes[to_keep]
        scores = scores[to_keep]

    if boxes.shape[-1] == 6:  # BBOXES
        center_x = boxes[:, 0] + (boxes[:, 2] - boxes[:, 0]) / 2
        center_y = boxes[:, 1] + (boxes[:, 3] - boxes[:, 1]) / 2
    else:
        center_x = boxes[:, 0]
        center_y = boxes[:, 1]

    X = np.dstack((center_x, center_y))[0]
    tree = KDTree(X)

    sorted_ids = np.argsort(scores)[::-1]

    ids_to_keep = []
    ind = tree.query_radius(X, r=radius)

    while len(sorted_ids) > 0:
        ids = sorted_ids[0]
        ids_to_keep.append(ids)
        sorted_ids = np.delete(sorted_ids, np.in1d(sorted_ids, ind[ids]).nonzero()[0])

    return boxes[ids_to_keep]


def nms(result_boxes, det_thres=None):
    arr = np.array(result_boxes)
    if arr is not None and isinstance(arr, np.ndarray) and (arr.shape[0] == 0):
        return result_boxes
    if det_thres is not None:
        before = np.sum(arr[:, -1] > det_thres)
    if arr.shape[0] > 0:
        try:
            arr = non_max_suppression_by_distance(arr, arr[:, -1], 25, det_thres)
        except:
            pass

    result_boxes = arr

    return result_boxes


def nms_patch(boxes, scores, thresh=0.5):
    idx_sort = scores.argsort(descending=True)
    boxes, scores = boxes[idx_sort], scores[idx_sort]
    to_keep, indexes = [], torch.LongTensor(np.arange(len(scores)))

    while len(scores) > 0:
        to_keep.append(idx_sort[indexes[0]])
        iou_vals = IoU_values(boxes, boxes[:1]).squeeze()
        mask_keep = iou_vals <= thresh
        if len(mask_keep.nonzero()) == 0: break
        boxes, scores, indexes = boxes[mask_keep], scores[mask_keep], indexes[mask_keep]
    return torch.LongTensor(to_keep)
