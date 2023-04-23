import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ImageLoader import load_image
from os import path


def __threshold_image(img, thresh=150, max_value=255):
    """
    Apply threshold for better blob detection (darkens the background and keeps the stars).
    """
    _, img_bin = cv2.threshold(img, thresh, max_value, cv2.THRESH_BINARY)
    return img_bin


# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
# https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
def __get_hough_circles(img):
    img_bin = __threshold_image(img)
    img_bin = cv2.GaussianBlur(img_bin, (7, 7), 0)
    circles = cv2.HoughCircles(img_bin, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=200, param2=0.8, minRadius=3, maxRadius=6)
    return circles


def __find_hough(img, as_pandas=False):
    coords, stars_data = [], []
    circles = __get_hough_circles(img)
    if circles is not None:
        for x, y, r in circles[0]:
            x, y, r = round(x, 5), round(y, 5), round(r, 2)
            b = img[int(y), int(x)] / np.max(img)
            stars_data.append((x, y, r, b))
            coords.append([x, y])
    return __handle_data_return(coords, stars_data, as_pandas)


def __get_blobs(img):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 250

    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = False
    params.blobColor = 255
    params.minRepeatability = 2

    detector = cv2.SimpleBlobDetector_create(params)

    img_bin = __threshold_image(img)
    kp = detector.detect(img_bin)
    return kp


def __find_blobs(img, as_pandas=False):
    keypoints = __get_blobs(img)
    coords, stars_data = [], []
    for kp in keypoints:
        x, y = kp.pt
        x, y = round(x, 5), round(y, 5)
        b = img[int(y), int(x)] / np.max(img)
        stars_data.append([x, y, round(kp.size, 2), b])
        coords.append([round(x, 5), round(y, 5)])
    return __handle_data_return(coords, stars_data, as_pandas)


def __handle_data_return(coords, stars_data, as_pandas):
    # empty coords_arr raises an error in pd.DataFrame
    stars_data_arr = np.array(stars_data) if len(stars_data) > 0 else stars_data
    coords_arr = np.array(coords) if len(coords) > 0 else coords
    if as_pandas:
        return coords_arr, pd.DataFrame(stars_data_arr, columns=['x', 'y', 'r', 'b'])
    return coords_arr, stars_data_arr


def find_stars(img, as_pandas=False, method='hough'):
    """
    :param method: 'blob' or 'hough' (default 'hough')
    """
    if method == 'blob':
        return __find_blobs(img, as_pandas)
    else:
        return __find_hough(img, as_pandas)


def save_as_text_file(stars_data: [np.ndarray, pd.DataFrame], filename: str):
    """
    :param stars_data: Data returned from get_blobs_data.
    :param filename: Path to save file
    """
    try:
        with open(filename, 'w') as file:
            np.savetxt(filename, stars_data, delimiter='\t', fmt='%f')
    except Exception as e:
        print(e)


def plot_detected_stars(img, stars_data):
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    for (x, y, r, b) in stars_data:
        ax.add_patch(plt.Circle((x, y), radius=r * 4, edgecolor='g', facecolor='none'))
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    im1_path = "Ex1_test_101/fr1.jpg"
    points1, im1_data = find_stars(load_image(im1_path))

    im2_path = "Ex1_test_101/ST_db2.png"
    im2 = load_image(im2_path)
    points2, im2_data = find_stars(im2)

    plot_detected_stars(im2, im2_data[:20])
