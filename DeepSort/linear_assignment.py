# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import enum
from turtle import distance
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


def matching_cascade(
        distance_metric, max_distance, cascade_depth, tracks, detections,
        track_indices=None, detection_indices=None):
    """Run matching cascade.
    Parameters
    ----------
    distance_metric : Callable(List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.
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

    unmatched_detections = detection_indices
    matches = []

    for level in range(1, cascade_depth + 1):
        
        # if matches all in detection, we break for loop
        if len(unmatched_detections) == 0:
            break

        # we get tracks in this depth level (mean time object outside the camera)
        track_indices_level = [
            indice for indice in track_indices if tracks[indice].time_since_update == level
        ]   

        if len(track_indices_level) == 0:
            continue # nothing to match in this level
        
        matches_level, _, unmatched_detections = \
            min_cost_matching(distance_metric, tracks, detections, track_indices_level, detection_indices)
    
        matches += matches_level

    # set will delete duplicate element
    unmatched_tracks = list(set(track_indices) - set(indice for indice, _ in matches))

    return matches, unmatched_tracks, unmatched_detections


def gate_cost_matrix(
        kf, cost_matrix, tracks, detections, track_indices, detection_indices,
        gated_cost=INFTY_COST, only_position=False):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.
    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
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
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.
    Returns
    -------
    ndarray
        Returns the modified cost matrix.
    """

    # observation of kalman fliter is (center x, center y, a, h)
    gating_dim = 2 if only_position else 4

    # degree of freedom chisquare-distribution and get threshold with 95%
    gating_threshold = kalman_filter.chi2inv95[gating_dim]

    meansurements = np.asarray([
        detections[indice].to_xyah() for indice in detection_indices
    ])

    for row, indice in enumerate(track_indices):
        track = tracks[indice]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, meansurements, only_position)
        
        # if observation too far from mean => we assume two object are different
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost

    return cost_matrix