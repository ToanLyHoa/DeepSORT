# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import enum
from cv2 import Mat
import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment
from . import kalman_filter

INFTY_COST = 1e+5


def min_cost_matching(
        distance_metric, max_distance, tracks, detections, track_indices=None,
        detection_indices=None):
    """Solve linear assignment problem.
    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.
    """

    if track_indices == None:
        track_indices = np.arange(len(tracks))

    if detection_indices == None:
        detection_indices = np.arange(len(detections))

    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices #nothing to match

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)

    # why we do that? I guess for improve perfomance of hungary algorithm
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5

    # implement Jonker-Volgenant algorithm with no initialization
    # and sum of cost_matrix[row_indice, col_indice] is min
    row_indices, col_indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    # check indx is not in row_indices and append in unmatched_tracks
    for row, indx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(indx)

    # check indx is not in col_indices and append in unmatched_tracks
    for col, indx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(indx)

    print('unmatched_tracks', unmatched_tracks)
    print('unmatched_detections', unmatched_detections)

    # check if distance > max_distance, we add this indx in unmatched
    # else we add in matched
    for row_indice, col_indice in zip(row_indices, col_indices):
        
        # remember row_indice is indice of array track_indices
        # value of track_indices is real indice
        track_incide = track_indices[row_indice]
        detection_indice = detection_indices[col_indice]

        if cost_matrix[row_indice, col_indice] > max_distance:
            unmatched_tracks.append(track_incide)
            unmatched_detections.append(track_indices[detection_indice])
        else:
            matches.append((track_incide, detection_indice))

    return matches, unmatched_tracks, unmatched_detections