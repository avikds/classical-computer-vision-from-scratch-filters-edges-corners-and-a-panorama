# Classical Computer Vision from Scratch: Filters, Edges, Corners and a Panorama

Build the classical image-stitching pipeline in plain NumPy, one piece at a time, and use it to turn two overlapping photographs into a single panorama. A seeded generator renders the photographs from a flat world of layered shapes, so you always know the true homography and the true scene. That lets you measure every stage instead of judging it by eye. Write convolution and separable Gaussian blur, then Sobel gradients and the Canny edge detector with non-maximum suppression and hysteresis. Find Harris corners from the structure tensor, refine them to sub-pixel accuracy, and measure how many survive a change of viewpoint. Describe each corner with a normalized patch and match the patches with Lowe's ratio test. Estimate the homography with the normalized DLT, then make it robust with RANSAC and adaptive stopping. Finally, warp with inverse mapping and bilinear sampling, match the two exposures and feather the seam. The full run stitches eight scenes to within about half a pixel of the true homography. It also measures what each design choice is worth: sub-pixel refinement, Hartley normalization, the ratio threshold, the RANSAC iteration formula, exposure matching and blending.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** convolve2d
- [x] **2.** gaussian_kernel1d
- [x] **3.** gaussian_blur
- [x] **4.** sobel_gradients
- [x] **5.** non_max_suppression
- [x] **6.** hysteresis_threshold
- [x] **7.** canny
- [x] **8.** structure_tensor
- [x] **9.** corner_response
- [x] **10.** local_maxima
- [x] **11.** refine_subpixel
- [x] **12.** harris_corners
- [x] **13.** apply_homography
- [x] **14.** normalize_points
- [x] **15.** dlt_homography
- [x] **16.** transfer_error
- [x] **17.** describe_patches
- [x] **18.** match_descriptors
- [x] **19.** corner_repeatability
- [x] **20.** ransac_num_iterations

---

Built on Deep-ML.
