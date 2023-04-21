import numpy as np


def calculate_transformation(H, src_pt):
    x, y = src_pt
    new_coords = H @ np.array([x, y, 1])
    new_coords /= new_coords[-1]  # divide by last coordinate to normalize
    new_coords = new_coords[:-1]  # remove last index
    if sum(new_coords < 0) == 0:  # check that both x, y coordinates are positive!
        return new_coords
    return None
