import random
import cv2
import numpy as np
from itertools import product
from collections import namedtuple
from skimage.measure import ransac, LineModelND
from skimage.transform import AffineTransform
from ImageLoader import load_image
from StarDetector import get_blobs

Line = namedtuple('Line', 'a b c')
Point = namedtuple('Point', 'x y')


def get_line_pts(L, x_min=0, x_max=500):
    # Compute two points on the line
    x1 = -x_min
    y1 = (-L.a * x1 - L.c) / L.b

    x2 = x_max
    y2 = (-L.a * x2 - L.c) / L.b
    return Point(x1, y1), Point(x2, y2)


def calculate_transformation(M, src_pt):
    x, y = src_pt
    new_coords = M @ np.array([x, y, 1])
    new_coords /= new_coords[-1]  # divide by last coordinate to normalize
    new_coords = new_coords[:-1]  # remove last index
    if sum(new_coords < 0) == 0:  # check that both x, y coordinates are positive!
        return new_coords
    return None


def __get_line(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    # Compute the line parameters (y = a*x + b + c)
    if x2 - x1 == 0:
        return None  # avoid division by zero
    a = (y2 - y2) / (x2 - x1)
    b = y1 - a * x1
    c = x2 * y1 - x1 * y2
    return Line(a, b, c)


def __calc_dist(pt, L):
    """
    Calculate the point's distance from a given line.
    :param pt:
    :param L:
    :return:
    """
    x, y = pt
    return abs((L.a * x - y + L.b + L.c) / np.sqrt(L.a ** 2 + L.b ** 2))


def estimate_line(points, threshold=100, max_iterations=3000):
    """
    Detects a line using the RANSAC algorithm.
    :param points: A list of 2D points in the form [(x1, y1), (x2, y2), ...].
    :param threshold: The maximum distance allowed between a point and the fitted line.
    :param max_iterations: The maximum number of iterations to run the RANSAC algorithm.
    :return: A tuple (a, b, inliers), where a and b are the slope and intercept of the fitted line
             and inliers is a list of the inlier points.
    """
    best_inliers = []
    best_line = None

    for i in range(max_iterations):
        # Randomly select two points
        p1, p2 = random.sample(points, 2)

        L = __get_line(p1, p2)
        if L is None:
            continue

        # Find the inliers (points that are within the threshold distance from the line)
        inliers = [pt for pt in points if __calc_dist(pt, L) < threshold]
        inliers = sorted(inliers, key=lambda pt: __calc_dist(pt, L))

        # Update the best line if we found more inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_line = L

    return best_line, best_inliers


def estimate_transformation(points1, points2, sample_size=4, max_iterations=1000):
    # estimate line model from keypoints
    line1, inliers1 = estimate_line(points1)
    line2, inliers2 = estimate_line(points2)
    best_inliers = 0
    best_model = None
    # iterate the transformed points and look for a match in the original image
    for _ in range(max_iterations):
        sample1 = random.sample(inliers1, sample_size)
        sample2 = random.sample(inliers2, sample_size)
        # robustly estimate affine transform model with RANSAC
        model, inliers = ransac((np.array(sample1), np.array(sample2)), AffineTransform, min_samples=3,
                                residual_threshold=2, max_trials=100)
        if inliers is None:
            continue
        n_inliers = sum(inliers == True)
        if best_inliers < n_inliers:
            best_inliers = n_inliers
            best_model = model
            print("HERE")
    return best_model


def get_star_matches(model, points1, points2):
    M = model.params
    dist_thresh = 100  # Set a distance threshold for matching points
    matched_points = []
    for i, p1 in enumerate(points1):
        p1_transformed = calculate_transformation(M, p1)
        if p1_transformed is not None:
            for j, p2 in enumerate(points2):
                curr_dist = np.linalg.norm(p1_transformed - p2)
                if curr_dist < dist_thresh:
                    matched_points.append((i, j, curr_dist))
    return matched_points


if __name__ == '__main__':
    im1_path = "Ex1_test_101/fr1.jpg"
    im2_path = "Ex1_test_101/fr2.jpg"

    img1 = load_image(im1_path)
    img2 = load_image(im2_path)

    kp1 = get_blobs(img1)
    kp2 = get_blobs(img2)

    # Convert the keypoints to numpy arrays
    pts1 = [val.pt for val in kp1]
    pts2 = [val.pt for val in kp2]

    line, inlier_pts = estimate_line(pts1)

    print(get_line_pts(line))
