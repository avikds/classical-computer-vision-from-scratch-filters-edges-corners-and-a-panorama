```
Eight generated scenes: two 160 x 220 views each, noise 0.01, view 2 darker by a hidden gain.

== Part 1. Convolution and Gaussian blur ==
  sigma 3 on 256 x 256: two 1-D passes 1 ms, one 2-D pass 9 ms (19 vs 361 taps), max difference 1.6e-15
  noise std after blur (measured / predicted sum g^2): {0.5: (0.643, 0.641), 1.0: (0.285, 0.282), 2.0: (0.142, 0.141), 4.0: (0.07, 0.071)}

== Part 2. Canny ==
  noise-free view 1: 3114 edge pixels
  noise 0.01: sigma 0.5: 3565 px, 89% on a clean edge | sigma 1.0: 3126 px, 99% on a clean edge | sigma 2.0: 2253 px, 99% on a clean edge
  noise 0.03: sigma 0.5: 3739 px, 86% on a clean edge | sigma 1.0: 3185 px, 97% on a clean edge | sigma 2.0: 2199 px, 99% on a clean edge
  noise 0.06: sigma 0.5: 9428 px, 36% on a clean edge | sigma 1.0: 3286 px, 93% on a clean edge | sigma 2.0: 2086 px, 99% on a clean edge

== Part 3. Harris corners ==
  noise 0.01: repeatability Harris 0.80, Shi-Tomasi 0.77 (4 scenes)
  noise 0.03: repeatability Harris 0.76, Shi-Tomasi 0.59 (4 scenes)
  noise 0.06: repeatability Harris 0.61, Shi-Tomasi 0.42 (4 scenes)
  sub-pixel refinement off: median error of correct matches under the true H 0.55 px
  sub-pixel refinement on : median error of correct matches under the true H 0.16 px

== Part 4. Homographies and the DLT ==
  coordinates +    0, 30 points, 0.5 px noise: corner error median / max: plain DLT 0.46 / 0.9 px, normalized 0.47 / 0.92 px
  coordinates + 1000, 30 points, 0.5 px noise: corner error median / max: plain DLT 1.62 / 337.6 px, normalized 0.47 / 0.92 px

== Part 5. Describing and matching ==
  ratio 0.60:  45.2 matches per pair,  93% correct,  42.1 correct
  ratio 0.70:  52.1 matches per pair,  85% correct,  44.3 correct
  ratio 0.80:  64.0 matches per pair,  73% correct,  46.6 correct
  ratio 0.90:  81.6 matches per pair,  58% correct,  47.6 correct
  ratio 1.01: 110.1 matches per pair,  44% correct,  47.9 correct
  descriptor blur sigma 0.0: 41.1 correct matches, 83% precision
  descriptor blur sigma 2.0: 46.6 correct matches, 73% precision

== Part 6. RANSAC ==
  DLT on all matches: median corner error 198 px
  RANSAC: corner errors [0.31, 0.58, 0.69, 0.7, 0.35, 0.35, 0.09, 0.23] px, inlier fraction 0.70, samples drawn [19, 27, 23, 15, 25, 24, 26, 12]
  50% outliers,  18 samples (formula says 72 for 99%): within 1 px in 39% of 100 runs, formula predicts 69%
  50% outliers,  72 samples (formula says 72 for 99%): within 1 px in 91% of 100 runs, formula predicts 99%

== Part 7. The panorama ==
  exposure matching off, overwrite: PSNR vs true scene 30.02 dB, seam jump 0.043
  exposure matching off, average  : PSNR vs true scene 29.26 dB, seam jump 0.025
  exposure matching off, feather  : PSNR vs true scene 29.03 dB, seam jump 0.016
  exposure matching on , overwrite: PSNR vs true scene 34.97 dB, seam jump 0.014
  exposure matching on , average  : PSNR vs true scene 34.92 dB, seam jump 0.013
  exposure matching on , feather  : PSNR vs true scene 34.80 dB, seam jump 0.015
  homography corner error over the 8 pairs: median 0.43 px, worst 0.69 px

total 8.5s
```
