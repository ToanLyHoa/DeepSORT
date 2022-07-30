import numpy as np

class Detection:
    """
    This class represents a bounding box detection in a single image.
    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.
    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.
    """

    def __init__(self, tlwh, confidence, feature) -> None:
        self.tlwh = np.asarray(tlwh, dtype = np.float)
        self.confidence = confidence
        self.feature = np.asarray(feature, dtype = np.float32)

    def to_tlbr(self):
        """
        This function convert bounding box (top left x, top left y, width, height) 
        to -> (top left x, top left y, bot right left x, bot right y) <+> (min x, min y, max x, max y)
        """

        tlbr = np.copy(self.tlwh)
        tlbr[2:] += tlbr[:2]

        return tlbr
    
    def to_xyah(self):
        """
        This function convert bounding box (top left x, top left y, width, height) 
        to -> (center x, center y, aspect ratio a, height h) 
        """
        xyah = np.copy(self.tlwh)
        
        xyah[:2] = (xyah[:2] + xyah[2:])/2

        # aspect ratio = w/h
        xyah[2] /= xyah[3]

        return xyah


