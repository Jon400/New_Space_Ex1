import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ImageLoader import load_image


# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d
# https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html
def get_hough_circles(img):
    img = cv2.GaussianBlur(img, (7, 7), 0)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=300, param2=0.8, minRadius=3, maxRadius=6)
    return circles


def plot_detected_stars(img, circles):
    if circles is not None:  # make sure circles were found!
        fig, ax = plt.subplots()
        ax.imshow(img, cmap='gray')
        for x, y, r in circles[0]:
            circle = plt.Circle((x, y), r, color='g', fill=False, linewidth=2)
            ax.add_artist(circle)
        plt.tight_layout()
        plt.show()


def get_stars_data(img, as_pandas=False):
    coords = []
    circles = get_hough_circles(img)
    if circles is not None:
        for x, y, r in circles[0]:
            x, y, r = int(x), int(y), int(r)
            coords.append((x, y, r, img[y, x]))
    coords_arr = np.array(coords, dtype=np.uint32) if len(coords) > 0 else coords  # empty np.array raises an error in pd.DataFrame
    if as_pandas:
        return pd.DataFrame(coords_arr, columns=['x', 'y', 'r', 'b'], dtype=np.uint32)
    return coords_arr


if __name__ == '__main__':
    im1_path = r'Stars/IMG_3046.HEIC'
    im2_path = r'Stars/IMG_3047.HEIC'

    im1 = load_image(im1_path)
    im2 = load_image(im2_path)

    im1_circles = get_hough_circles(im1)
    im2_circles = get_hough_circles(im2)

    im1_data = get_stars_data(im1)
    # print(im1_data)

    plot_detected_stars(im1, im1_circles)
    # plot_detected_stars(im2, im2_circles)