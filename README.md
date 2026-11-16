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

---

Built on Deep-ML.
