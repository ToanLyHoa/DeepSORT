from __future__ import absolute_import
import numpy as np
from . import linear_assignment 

def iou(bbox, candidates):
    """Computer intersection over union.
    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`. (N, 4)
    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    """

    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    
    intersect_tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0]),
                            np.maximum(bbox_tl[1], candidates_tl[:, 1])]
    intersect_br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0]),
                            np.minimum(bbox_br[1], candidates_br[:, 1])]
    
    # no intersect have negative width and height
    intersect_wd = np.maximum(0, np.subtract(intersect_br, intersect_tl))

    # area = w*h, we use width and height available 
    intersect_area = np.prod(intersect_wd, axis=1)
    candidates_area = np.prod(candidates[:, 2:], axis=1)
    bbox_area = np.prod(bbox[2:])

    # iou = intersect/union = intersect/(A + B - AB)
    iou_meansure = intersect_area/(bbox_area + candidates_area - intersect_area)

    return iou_meansure

def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.
    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.
    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.
    """

    if track_indices == None:
        track_indices = np.arange(len(tracks))
    
    if detection_indices == None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))

    for row, indice in enumerate(track_indices):
        
        # if object outside camera (mean we not update) we dont find iou for this object
        if(tracks[indice].time_since_update > 1):
            cost_matrix[row, :] = linear_assignment.INFTY_COST
            continue

        bbox = tracks[indice].to_tlwh()
        candidates = np.asarray([detections[indice].tlwh for indice in detection_indices])
        cost_matrix[row, :] = 1 - iou(bbox, candidates)
        
    
    return cost_matrix
        
