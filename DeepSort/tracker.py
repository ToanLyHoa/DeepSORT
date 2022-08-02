from __future__ import absolute_import
from statistics import covariance
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1


    def predict(self):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self.match(detections)

        # update previous state of object (Kalman filter) are detected
        for match_track, match_detection in zip(matches):
            self.tracks[match_track].update(self.kf, detections[match_detection])
        
        # update state of missed track
        for unmatched_track in unmatched_tracks:
            self.tracks[unmatched_track].mark_missed()

        # create track for unmatched_detections
        for unmatched_detection in unmatched_detections:
            self.initiate_track(detections[unmatched_detection])

        # update tracks after match with detections
        self.tracks = [track for track in self.tracks if not track.is_deleted()]

        # update matrix distance of comfirm track.feature 
        # + delete deleted track and 
        # + add new track
        # + add newfeature to comfirm track
        confirm_trackids = np.asarray([track.track_id for track in self.tracks if track.is_confirmed()])

        feature_confirmed_tracks, feature_confirmed_trackids = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            
            # remember track.features of tentative track maybe a list 
            feature_confirmed_tracks += track.features

            # id of each feature, it's convenient in for loop
            feature_confirmed_trackids += [track.id for _ in track.features]

            # we reset features of track here, avoid duplicate feature 
            track.features = []

        self.metric.partial_fit(
            np.asarray(feature_confirmed_tracks), np.asarray(feature_confirmed_trackids), confirm_trackids)
        pass

    def match(self, detections):
        """
        Use cosine or euclide distance between two feature extracted from Convolution neural network 
        which assign comfirmed track to dectections
        Use iou distance between bbox of track predict and bbox of detection which assign
        remaining tracks to remaining detections 
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        """

        def distance_metric(tracks, detections, track_indices, detection_indices):
            # distance_metric : Callable(List[Track], List[Detection], List[int], List[int])
            detection_features = np.asarray([detections[indice].feature for indice in detection_indices])
            trackids = np.asarray([tracks[indice].id for indice in track_indices])
            distance_matrix = self.metric.distance(detection_features, trackids)
            # apply gating distance to avoid 2 bbox of track and detection too far
            distance_matrix = linear_assignment.gate_cost_matrix(
                self.kf, distance_matrix, tracks,detections, track_indices, 
                detection_indices)
            return distance_matrix

        confirmed_track_indices = [i for i, track in enumerate(self.tracks) if track.is_confirmed()]
        unconfirmed_track_indices = [i for i, track in enumerate(self.tracks) if not track.is_confirmed()]

        matches, unmatched_confirmed_track_indices, unmatched_detection_track_indices = \
            linear_assignment.matching_cascade(
                distance_metric, self.metric.matching_threshold, self.max_age, 
                self.tracks, detections, confirmed_track_indices)


        # Logic: 
        #   we call this A:
        #   + when object go outside then track.time_since_update > 1
        #   and we matching by cosine distance between feature object 
        #   and detections to re-tracking (person-reid) 
        # 
        #   We call this B:
        #   + when object appear constantly then track.time_since_update == 1
        #   (remeber we call predict before update so time_since_update of new track always == 1)
        #   and if we can not match by consine distance then we will find it by iou_matching 
        #   
        #   + we also use iou_matching to matching unconfirmed_track object 
        #   (author assume that we need 3,4 feature of object is tracked
        #   to implement cosine distance or person reid task)

        # object outside
        track_indices_A = [
            indice for indice in unmatched_confirmed_track_indices 
            if self.tracks[indice].time_since_update != 1
        ]

        # object maybe inside but cannot recognize
        track_indices_B = unconfirmed_track_indices + [
            indice for indice in unmatched_confirmed_track_indices 
            if self.tracks[indice].time_since_update == 1
        ]

        # note: we use unmatched_detection bbox to match
        iou_matches, unmatch_track_indices_B, unmatch_detection_indices_B = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, track_indices_B, unmatched_detection_track_indices)

        # matches of cosine distance and iou distance
        matches = matches + iou_matches

        # unmatch is object outside + 
        unmatched_tracks = list(set(unmatch_track_indices_B + track_indices_A))

        return matches, unmatched_tracks, unmatch_detection_indices_B


    def initiate_track(self, detection):
        # when a detection doest math with any track
        mean, covariance = self.kf.initiate(detection.detection.to_xyah())
        track = Track(
            mean, covariance, self._next_id, 
            self.n_init, self.max_age, detection.feature)
        self.tracks.append(track)
        self._next_id += 1
        pass