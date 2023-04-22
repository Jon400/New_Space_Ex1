# New-Space-Ex1

## Part 1: Algorithm

For each image:

1. Load the image.
2. Detect the stars in the image.
3. Determine the slope (m) and y-intercept (b) of the line that goes through the stars which have the shortest distance
   from the given points, using the least squares method. 
4. Sort the stars by their distance from the line.
5. Take the first 15 stars from the sorted list and label them as inliers.

For each pair of images:

1. Take the inliers from each image and match them using the RANSAC algorithm.
2. Calculate the Affine transformation from the first image to the second image.
3. Apply the transformation to the first image and choose the point from the second image that is closest to the
   transformed point within a threshold.
4. Count the number of correct matches.
5. Repeat steps 1-4 until reaching some fixed maximum number of iterations.
6. Choose the transformation that gave the maximum number of correct matches.

## Part 2: Detecting Stars

The main functions we used to detect stars and get the data:

### ImageLoader.py:

Functions that load and display images (such as png, jpf, and heic).

* `load_image`
* `plot_loaded_images`

### StarDetector.py

Functions that detect stars in a loaded image.

We used or experimented with some methods including: thresholding, blurring, hough circles, detecting blobs.

* `find_stars`: Detect the stars in an image.
* `save_as_text_file`: Saves the stars' data in a file **(x, y, r, b)**.

## Part 3: Matching Stars in 2 Images

### StarMatching.py

* `estimate_transformation`: Uses RANSAC to estimate the Affine transformation from Image1 to Image2.
* `get_star_matches`:

## Part 4: Results