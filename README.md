# New-Space-Ex1

## Usage

See full example in [Demo](Demo.ipynb).

```python
##### Load the images #####
im1 = load_image(im1_path)
im2 = load_image(im2_path)

##### Find feature points in each image #####
points1, im1_data = find_stars(im1, method='hough')
points2, im2_data = find_stars(im2, method='hough')

##### Estimate Affine transformation #####
model, L1, L2 = estimate_transformation(points1, points2, method='ransac')
##### Detect matches using the estimated transformation #####
matched_points = get_star_matches(model, points1, points2)

##### Plot and save results #####
plot_matches(matched_points, im1, im2, im1_data, im2_data, L1, L2)
save_as_text_file(im1_data, f"StarsData/{im1_path.split('/')[-1].split('.')[0]}.txt", verbose=True)
save_as_text_file(im2_data, f"StarsData/{im2_path.split('/')[-1].split('.')[0]}.txt", verbose=True)
```

## Methods and Approaches Explored

* In our experimentation, we extensively tried out various libraries and their functions including opencv, astropy, and
  others.
* We carefully evaluated the results and retained the methods that worked best for our project.
* We also tried out other approaches such as blob detection but ultimately decided to use hough circles with Gaussian
  blurring and image thresholding for our specific task.
* Similarly, we initially attempted to use least squares to detect the line with the most inliers, but found that using
  RANSAC for estimating the line provided better results.
* We decided to make blob detection and least squares available as optional methods that can be called when running the
  functions. You can experiment with these methods by changing the parameters in the function call.

## Part 1: Algorithm

1. Sort the detected feature points (stars) coordinates by their distance from the origin point for each image.

The following steps are repeated for a fixed number of iterations:

2. Estimate a line for each image and return the line and inlier points for the one with the most inliers.
3. Sample three indices from the inlier points of each line to estimate the Affine transformation from image 1 to image
   2 using RANSAC.
4. Check for matching correspondence of the transformed inlier points from image 1 in the inliers from the second image,
   and count the number of correct matches.
5. If the number of correct matches exceeds the previous best number of matches, save the current Affine matrix as the
   best model.
6. After completing all iterations, return the best Affine matrix (i.e. the RANSAC model) and the computed lines from
   the function.

## Part 2: Detecting Stars

To detect stars and extract their relevant data, we primarily relied on the following key functions:

### ImageLoader.py:

The file ImageLoader.py consists of functions that are designed to load and display images in various formats such as
png, jpg, and heic.

One such function is `load_image`.

### StarDetector.py

Contains functions that are designed to detect stars within an image. We explored and experimented with various methods
such as thresholding, blurring, hough circles, and blob detection techniques to achieve our goal.

* `find_stars`: Identifies and locates the stars present in an image.
* `save_as_text_file`: Stores the relevant data of the detected stars, which includes their x and y coordinates, radius,
  and brightness, into a file. The file format is **(x, y, r, b)**.
* `plot_detected_stars`:  Generates a plot that shows the original image and its corresponding thresholded version side
  by side. Additionally, it highlights the stars that were detected on the thresholded binary image.

## Part 3: Matching Stars in 2 Images

### StarMatching.py

The purpose of this script is to offer functions that can be used to compare and match stars that are present in two
different images. In addition to comparing and matching stars, this script also provides functions to visualize the
detected matches between the two images.

* `estimate_transformation`: Utilizes RANSAC algorithm to estimate the Affine transformation required to map Image1 onto
  Image2. RANSAC is also employed for the detection of lines in the images.
* `get_star_matches`: Computes the transformed coordinates of the stars in one image and searches for corresponding
  matches in the other image.
* `plot_matches`: Produces a visualization of the matches and estimated line that were detected during the matching
  process.

## Part 4: Results

You can refer to the [Demo](Demo.ipynb) file or run it yourself to view the results.

In the `StarsData` directory, there are text files that store the data of the detected stars in the test images.