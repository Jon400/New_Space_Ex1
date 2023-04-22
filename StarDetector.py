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


def __get_blobs(img):
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 10
    params.maxArea = 600

    params.filterByCircularity = True
    params.minCircularity = 0.1

    params.filterByConvexity = False
    params.blobColor = 255
    params.minRepeatability = 2

    detector = cv2.SimpleBlobDetector_create(params)

    img_bin = __threshold_image(img)
    kp = detector.detect(img_bin)
    return kp


def find_stars(img, as_pandas=False):
    keypoints = __get_blobs(img)
    coords = []
    for kp in keypoints:
        x, y = kp.pt
        b = img[int(y), int(x)] / np.max(img)
        coords.append([round(x, 5), round(y, 5), round(kp.size, 2), b])
    coords_arr = np.array(coords) if len(coords) > 0 else coords
    # empty coords_arr raises an error in pd.DataFrame
    if as_pandas:
        return keypoints, pd.DataFrame(coords_arr, columns=['x', 'y', 'r', 'b'])
    return keypoints, coords_arr


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


def plot_detected_stars(img, kps):
    img_with_kps = cv2.drawKeypoints(img, kps, np.array([]),
                                     (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Display the result using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_kps)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    im1_path = "Ex1_test_101/fr1.jpg"
    keypoints1, im1_data = find_stars(load_image(im1_path))
