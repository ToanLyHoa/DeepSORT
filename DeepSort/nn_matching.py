import numpy as np

# this file produce function to compute cost matrix about tracks and detection

def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """

    a, b = np.asarray(a), np.asarray(b) 

    N, _= a.shape
    L, _= b.shape

    # we want to calculate (a - b)^2 = a^2 + b^2 - 2ab
    a2, b2 = np.sum(np.square(a), axis=1), np.sum(np.square(b), axis=1)

    # We can use a2[:, None] to expand second dimension (N,) => (N, 1)
    # We can use b2[None, :] to expand first dimension (L,) => (1, L)
    a2, b2 = np.tile(a2, (L, 1)), np.tile(b2, (N, 1))
    _2ab  = np.dot(np.matmul(a, b.T), 2)

    # Diff way: use point wise + - array in numpy 
    distance_matrix = a2.T + b2 - _2ab

    return distance_matrix

def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.
    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.
    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.
    """
    
    a, b = np.asarray(a), np.asarray(b) 

    if not data_is_normalized:
        a = np.divide(a, np.linalg.norm(a, axis=1, keepdims= True)) 
        b = np.divide(b, np.linalg.norm(b, axis=1, keepdims= True)) 

    # cos(0) = 1 => we substract 1 for cosine_distance
    return 1 - np.matmul(a, b.T)

def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.
    """
    distance = _pdist(x, y)

    return np.maximum(0, np.min(distance, axis = 0))

def _nn_cosine_distance(x, y):
    """ Helper function for nearest neighbor distance metric (cosine).
    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).
    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.
    """

    distance = _cosine_distance(x, y)

    return np.min(distance, axis = 0)


class NearestNeighborDistanceMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.
    Parameters
    ----------
    metric : str
        Either "euclidean" or "cosine".
    matching_threshold: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.
    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.
    """
    def __init__(self, metric, matching_threshold, budget = None) -> None:

        if metric == "euclidean":
            self.metric = _nn_euclidean_distance
        elif metric == "cosine":
            self.metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine' ")

        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}


        pass
    def partial_fit(self, features, targets, active_targets):
        """Update the distance metric with new data.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        """

        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)

            if self.budget is not None:
                # remove oldest feature of track
                self.samples[target] = self.samples[target][-self.budget:]
        
        # remove feature of deleted track
        self.samples[target] = {i: self.samples[i] for i in active_targets}


    def distance(self, features, targets):
        """Compute distance between features and targets.
        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.
        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """

        # compute distance of features of dectections and target of track
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            # it will get minimum distance of each target
            # remember: each target track (trackid) have a list of distance
            cost_matrix[i, :] = self.metric(self.samples[target], features)

        return cost_matrix