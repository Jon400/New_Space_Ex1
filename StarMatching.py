import random
import math
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


def __get_points_on_line(L, min_x, max_x):
    m, b = L.m, L.b
    y1 = m * min_x + b
    y2 = m * max_x + b
    return np.array([min_x, y1]), np.array([max_x, y2])


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


def __estimate_line_lstsq(points: list) -> [Line, np.ndarray]:
    L = __least_squares(np.array(points))
    p1, p2 = __get_points_on_line(L, 0, 1)
    ret_points = sorted(points, key=lambda p: __calc_dist(p, p1, p2))
    return L, np.array(ret_points)


def __estimate_line_ransac(points: list):
    points_arr = np.array(points)
    model, inliers = ransac(points_arr, LineModelND, min_samples=2, residual_threshold=15)
    p1, p2 = model.params
    return __get_line_from_points(p1, p2), points_arr[inliers]


def __get_line_from_points(p1: np.ndarray, p2: np.ndarray) -> Line:
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:  # avoid division by zero!
        return Line(0, 0)
    m = (y2 - y1) / (x2 - x1)  # calculate the slope
    b = y1 - m * x1  # calculate the y-intercept
    return Line(m, b)


def __estimate_line(sample1, sample2, method):
    if method == 'lstsq':
        L1, inliers1 = __estimate_line_lstsq(sample1)
        L2, inliers2 = __estimate_line_lstsq(sample2)
    else:
        L1, inliers1 = __estimate_line_ransac(sample1)
        L2, inliers2 = __estimate_line_ransac(sample2)
    return L1, inliers1, L2, inliers2


def estimate_transformation(points1: list, points2: list, max_iterations=200, method='ransac'):
    """
    :param method: 'lstsq' (Least Squares) or 'ransac' (default is 'ransac')
    :return:
    """
    try:
        best_correct = 0
        best_model = None
        best_L1, best_L2 = None, None
        # Check if satisfies minimum length
        min_len = min(len(points1), len(points2))
        if min_len < 3:
            print("estimate_transformation(): Not enough parameters!")
            return best_model, best_L1, best_L2

        # Sort points by distance from the origin
        points1 = sorted(points1, key=lambda p: math.dist(p, [0, 0]))
        points2 = sorted(points2, key=lambda p: math.dist(p, [0, 0]))

        for i in range(max_iterations):
            L1, inliers1, L2, inliers2 = __estimate_line(points1, points2, method)
            size = min(len(inliers1), len(inliers2))
            if size < 3:  # Not enough points for estimation!
                continue
            indices = random.sample(range(size), 3)  # Choose 3 random indices
            # Robustly estimate affine transform model with RANSAC
            model, _ = ransac((inliers1[indices], inliers2[indices]), AffineTransform, min_samples=2,
                              residual_threshold=1, max_trials=100)
            # Count number of correct matches (inliers only!)
            matched_points = get_star_matches(model, inliers1, inliers2)
            n_correct = len(matched_points)
            if best_correct < n_correct:
                best_correct = n_correct
                best_model = model
                best_L1, best_L2 = L1, L2
        return best_model, best_L1, best_L2
    except Exception as e:
        print(e)
        return None, None, None


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


def get_star_matches(model, points1: list, points2: list, dist_thresh=15) -> np.ndarray:
    """
    Calculate the transformed points and look for a match in the other image.
    :param model: Trained RANSAC model.
    :param points1: Feature points from first image.
    :param points2: Feature points from second image.
    :param dist_thresh: Set a distance threshold for matching points (default = 15)
    :return: List of index pairs of the point in each list (points1_idx, points2_idx), after validation.
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


def plot_matches(matched_points: np.ndarray, im1: np.ndarray, im2: np.ndarray,
                 im1_data: np.ndarray, im2_data: np.ndarray,
                 L1: Line, L2: Line, n_first=20):
    p1, p2 = __get_points_on_line(L1, 0, im1.shape[1])
    p3, p4 = __get_points_on_line(L2, 0, im2.shape[1])

    fig, ax = plt.subplots(ncols=2, figsize=(10, 10))
    ax[0].imshow(im1, cmap='gray')
    ax[1].imshow(im2, cmap='gray')

    # Plot fitted lines
    ax[0].plot((p1[0], p2[0]), (p1[1], p2[1]), 'g')
    ax[1].plot((p3[0], p4[0]), (p3[1], p4[1]), 'g')

    for num, (i, j) in enumerate(matched_points, 1):
        x1, y1, r1, b1 = im1_data[i]
        ax[0].text(x1, y1, f"{num}", color='b', fontsize=12, horizontalalignment='left', verticalalignment='baseline')
        ax[0].add_patch(plt.Circle((x1, y1), radius=r1 * 4, edgecolor='r', facecolor='none'))

        x2, y2, r2, b2 = im2_data[j]
        ax[1].text(x2, y2, f"{num}", color='b', fontsize=12, horizontalalignment='left', verticalalignment='baseline')
        ax[1].add_patch(plt.Circle((x2, y2), radius=r2 * 4, edgecolor='r', facecolor='none'))

        if num == n_first:  # Otherwise is crowded!
            break

    # Plot all detected stars
    for star in im1_data:
        x1, y1, r1, b1 = star
        ax[0].add_patch(plt.Circle((x1, y1), radius=r1 * 4, edgecolor='r', facecolor='none', alpha=0.2))
    for star in im2_data:
        x1, y1, r1, b1 = star
        ax[1].add_patch(plt.Circle((x1, y1), radius=r1 * 4, edgecolor='r', facecolor='none', alpha=0.2))

    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    im1_path = "Ex1_test_101/fr1.jpg"
    im2_path = "Ex1_test_101/fr2.jpg"
    im1 = load_image(im1_path)
    im2 = load_image(im2_path)

    points1, im1_data = find_stars(im1, method='hough')
    points2, im2_data = find_stars(im2, method='hough')
    print(f"\tImage1 number of feature points found: {len(points1)}\n"
          f"\tImage2 number of feature points found: {len(points2)}")

    model, L1, L2 = estimate_transformation(points1, points2, method='ransac')
    matched_points = get_star_matches(model, points1, points2)

    print(f"Number of matches: {len(matched_points)}")

    plot_matches(matched_points, im1, im2, im1_data, im2_data, L1, L2, n_first=20)
