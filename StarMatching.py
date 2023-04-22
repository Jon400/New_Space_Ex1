import random
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from collections import namedtuple
from skimage.measure import ransac, LineModelND
from skimage.transform import AffineTransform
from ImageLoader import load_image
from StarDetector import find_stars

Line = namedtuple('Line', 'm b')


def __calculate_transformation(M: np.ndarray, src_pt):
    x, y = src_pt
    new_coords = M @ np.array([x, y, 1])
    new_coords /= new_coords[-1]  # divide by last coordinate to normalize
    new_coords = new_coords[:-1]  # remove last index
    if sum(new_coords < 0) == 0:  # check that both x, y coordinates are positive!
        return new_coords
    return None


def __get_line_points(L):
    m, b = L.m, L.b
    return np.array([0, b]), np.array([-b / m, 0])


# https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
def __calc_dist(p1, p2, p3):
    """
    Calculate the point's distance from a given line.
    """
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


# https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
def __least_squares(points):
    x, y = points[:, 0], points[:, 1]
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return Line(m, b)


def __estimate_line(points: list, return_n_first=15) -> [Line, np.ndarray]:
    L = __least_squares(np.array(points))
    p1, p2 = __get_line_points(L)
    ret_points = sorted(points, key=lambda p: __calc_dist(p, p1, p2))
    return L, np.array(ret_points)[:return_n_first]


def __estimate_line_ransac(points: list):
    points_arr = np.array(points)
    model, inliers = ransac(points_arr, LineModelND, min_samples=2, residual_threshold=5)
    return model, points_arr[inliers]


def estimate_transformation(points1: list, points2: list, max_iterations=500):
    try:
        L1, inliers1 = __estimate_line(points1)
        L2, inliers2 = __estimate_line(points2)
        size = min(len(inliers1), len(inliers2))
        if size < 3:
            return None
        best_correct = 0
        best_model = None
        for i in range(max_iterations):
            # robustly estimate affine transform model with RANSAC
            model, _ = ransac((inliers1, inliers2), AffineTransform, min_samples=2,
                              residual_threshold=2, max_trials=100)
            # train model on inliers, compute matches on all feature points
            matched_points = get_star_matches(model, points1, points2)
            n_correct = len(matched_points)
            if best_correct < n_correct:
                best_correct = n_correct
                best_model = model
        return best_model
    except Exception as e:
        print(e)
        return None


def __validate_matching(matched_points: list) -> np.ndarray:
    """
    :param matched_points: List of potential matchings (assumes matches are in ascending order according to distance).
    :return: Array without duplicated matched points.
    """
    col_names = ['i', 'j']
    df = pd.DataFrame(data=matched_points, columns=col_names, dtype='int').astype(pd.Int64Dtype())
    df.drop_duplicates(subset=col_names[0], keep='first', inplace=True)
    df.drop_duplicates(subset=col_names[1], keep='first', inplace=True)
    reduced_points = df.to_numpy()  # take only best points!
    return reduced_points


def get_star_matches(model, points1: list, points2: list, dist_thresh=10) -> np.ndarray:
    """
    Calculate the transformed points and look for a match in the other image.
    :param model: Trained RANSAC model.
    :param points1: Feature points from first image.
    :param points2: Feature points from second image.
    :param dist_thresh: Set a distance threshold for matching points (default=100)
    :return: List of points after validation.
    """
    try:
        M = model.params  # Get transformation matrix
        matched_points = []
        for i, p1 in enumerate(points1):
            p1_transformed = __calculate_transformation(M, p1)
            if p1_transformed is not None:
                for j, p2 in enumerate(points2):
                    curr_dist = np.linalg.norm(p1_transformed - p2)
                    if curr_dist < dist_thresh:
                        matched_points.append((i, j, curr_dist))
        sorted_points = sorted(matched_points, key=lambda val: val[2])  # sort by lowest to highest distance
        mapped_points = list(map(lambda v: (v[0], v[1]), sorted_points))  # save only the indices
        return __validate_matching(mapped_points)
    except Exception as e:
        print(e)
        return np.ndarray([])


if __name__ == '__main__':
    im1_path = "Ex1_test_101/fr1.jpg"
    im2_path = "Ex1_test_101/fr2.jpg"
    im1 = load_image(im1_path)
    im2 = load_image(im2_path)

    keypoints1, im1_data = find_stars(im1)
    keypoints2, im2_data = find_stars(im2)

    points1 = [pt for pt in im1_data[:, :2]]
    points2 = [pt for pt in im2_data[:, :2]]

    model = estimate_transformation(points1, points2)
    matched_points = get_star_matches(model, points1, points2)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
    fig.suptitle("Detected Matchings", size=15)
    ax[0].imshow(im1, cmap='gray')
    ax[1].imshow(im2, cmap='gray')

    for num, (i, j) in enumerate(matched_points, 1):
        x1, y1, r1, b1 = im1_data[i]
        ax[0].text(x1, y1, f"{num}", color='b', fontsize=12, horizontalalignment='left', verticalalignment='baseline')
        ax[0].add_patch(plt.Circle((x1, y1), radius=r1 + 25, edgecolor='r', facecolor='none'))

        x2, y2, r2, b2 = im2_data[j]
        ax[1].text(x2, y2, f"{num}", color='b', fontsize=12, horizontalalignment='left', verticalalignment='baseline')
        ax[1].add_patch(plt.Circle((x2, y2), radius=r2 + 25, edgecolor='r', facecolor='none'))

    plt.tight_layout()
    plt.show()
