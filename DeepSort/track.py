
class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.
    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    """
    def __init__(self, mean, covariance, track_id, n_init, max_age, 
                feature = None) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 1
        self.state = TrackState.Tentative
        self.feature = []

        if feature is not None:
            self.feature.append(feature)

        self.n_init = n_init
        self.max_age = max_age
    
    def to_tlwh(self):
        """
        This function convert bounding box (center x, center y, aspect ratio a, height h) 
        to -> (top left x, top left y, width, height) 

        Return:
            The bounding box: (top left x, top left y, width, height) 
        """
        to_tlwh = self.mean[:4].copy()
        # aspect ratio a = w/h => w = a*h
        to_tlwh[2] *= to_tlwh[3]
        to_tlwh[:2] -= to_tlwh[2:]/2

        return to_tlwh

    def to_tlbr(self):
        """
        This function convert bounding box (center x, center y, aspect ratio a, height h) 
        to -> (top left x, top left y, bot right left x, bot right y) <+> (min x, min y, max x, max y)

        Return:
            The bounding box: (min x, min y, max x, max y)
        """
        to_tlbr = self.to_tlwh()
        to_tlbr[2:] += to_tlbr[:2]

        return to_tlbr

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)

        # increase age and time_since_update when predict
        self.age +=1
        self.time_since_update +=1

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.
        """
        measurement = detection.to_xyah()
        self.mean, self.covariance = kf.predict(self.mean, self.covariance, measurement)

        self.feature.append(detection.feature)

        # increase when we update
        self.hits += 1
        # set 0 when we update
        self.time_since_update = 0

        # when we update with a sufficient amount of times, we set to confirmed state
        if self.state == TrackState.Tentative and self.hits >= self.n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """

        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        # when we predict a sufficient amount of times without update, we set to deleted state
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
    
    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

