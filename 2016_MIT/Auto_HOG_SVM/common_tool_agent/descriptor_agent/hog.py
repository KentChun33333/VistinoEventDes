
# Use skimage feature 
from skimage import feature
import numpy as np

class HOG:
	def __init__(self, orientations=12, pixelsPerCell=(4, 4), cellsPerBlock=(2, 2), normalize=True):
		# store the number of orientations, pixels per cell, cells per block, and
		# whether normalization should be applied to the image
		self.orientations = orientations
		self.pixelsPerCell = pixelsPerCell
		self.cellsPerBlock = cellsPerBlock
		self.normalize = normalize

	def describe(self, image):
		# compute Histogram of Oriented Gradients features
		hist = feature.hog(image, orientations=self.orientations, pixels_per_cell=self.pixelsPerCell,
			cells_per_block=self.cellsPerBlock, normalise=self.normalize)
		hist[hist < 0] = 0
		# return the histogram
		return hist

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius    

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
        self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)        

        # return the histogram of Local Binary Patterns
        return hist


