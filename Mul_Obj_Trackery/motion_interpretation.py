from math import sqrt
from sys import maxsize
from scipy.stats.mstats import mode

################################
# [*] Motion Interpretation    #
################################
#   up-left    up   up-right   #
#          7   8   9           #
#     left 4   5   6 right     #
#          1   2   3           #
# down-left   down  down-right #
################################



DOWNLEFT   = 1
DOWN       = 2
DOWNRIGHT  = 3
LEFT       = 4
STATIC     = 5
RIGHT      = 6
UPLEFT     = 7
UP         = 8
UPRIGHT    = 9


def interpreter(coord1, coord2, noiseDistance=20):
    # Return the integer of one of the 8 directions this line is going in.
    # coord1 and coord2 are (x, y) integers coordinates.

    x1, y1 = coord1
    x2, y2 = coord2
    distance_of_vect = sqrt((x1-x2)**2+(y1-y2)**2)
    if x1 == x2 and y1 == y2:
        return STATIC  # two coordinates are the same.
    elif  distance_of_vect < noiseDistance:
    # [*] the magnitude is too small
    	return STATIC
    elif x1 == x2 and y1 > y2:
        return UP
    elif x1 == x2 and y1 < y2:
        return DOWN
    elif x1 > x2 and y1 == y2:
        return LEFT
    elif x1 < x2 and y1 == y2:
        return RIGHT

    slope = float(y2 - y1) / float(x2 - x1)

    # Figure out which quadrant the line is going in, and then
    # determine the closest direction by calculating the slope
    if x2 > x1 and y2 < y1: # up right quadrant
        if slope > -0.4142:
            return RIGHT # slope is between 0 and 22.5 degrees
        elif slope < -2.4142:
            return UP # slope is between 67.5 and 90 degrees
        else:
            return UPRIGHT # slope is between 22.5 and 67.5 degrees
    elif x2 > x1 and y2 > y1: # down right quadrant
        if slope > 2.4142:
            return DOWN
        elif slope < 0.4142:
            return RIGHT
        else:
            return DOWNRIGHT
    elif x2 < x1 and y2 < y1: # up left quadrant
        if slope < 0.4142:
            return LEFT
        elif slope > 2.4142:
            return UP
        else:
            return UPLEFT
    elif x2 < x1 and y2 > y1: # down left quadrant
        if slope < -2.4142:
            return DOWN
        elif slope > -0.4142:
            return LEFT
        else:
            return DOWNLEFT

