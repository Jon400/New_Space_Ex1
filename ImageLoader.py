import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pillow_heif
from PIL import Image


def __validate_path(img_path):
    """
    https://stackoverflow.com/questions/63866180/how-to-convert-from-heic-to-jpg-in-python-on-windows
    :param img_path:
    :return:
    """
    try:
        # Check if image format is .heic
        parsed_path = img_path.split('.')
        if parsed_path[-1] in ['HEIC', 'heic']:
            heif_file = pillow_heif.read_heif(img_path)
            im = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
            )
            new_path = f"{parsed_path[0]}.png"
            if not os.path.isfile(new_path):  # check if image was already saved
                im.save(new_path, format('png'))
            return new_path
        return img_path
    except Exception as e:
        print(e)


def load_image(img_path):
    """
    https://stackabuse.com/opencv-edge-detection-in-python-with-cv2canny/
    :param img_path:
    :return:
    """
    try:
        valid_path = __validate_path(img_path)
        im = cv2.cvtColor(cv2.imread(valid_path), cv2.COLOR_BGR2GRAY)
        im = im.astype(np.uint8)
        return im
    except Exception as e:
        print(e)


def plot_loaded_images(img1, img2):
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    fig.suptitle("Loaded Images", size=15)
    ax[0].imshow(img1, cmap='gray')
    ax[1].imshow(img2, cmap='gray')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    im1_path = r'Stars/IMG_3046.HEIC'
    im2_path = r'Stars/IMG_3047.HEIC'

    im1 = load_image(im1_path)
    im2 = load_image(im2_path)

    # plot_loaded_images(im1, im2)
