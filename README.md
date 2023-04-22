# New-Space-Ex1

## Part 1: Algorithm

## Part 2: Detecting Stars

The main functions we used to detect stars and get the data:

### ImageLoader.py:

Functions that load and display images (such as png, jpf, and heic).

* `load_image`
* `plot_loaded_images`

### StarDetector.py

Functions that detect stars in a loaded image.

* `find_stars`: Uses blob detection with binary thresholding to detect the stars in an image.
* `save_as_text_file`: Saves the stars data in a file **(x, y, r, b)**.

## Part 3: Matching Stars in 2 Images

### StarMatching.py

* `estimate_transformation`: Uses RANSAC to estimate the Affine transformation from Image1 to Image2.
* `get_star_matches`:

## Part 4: Results