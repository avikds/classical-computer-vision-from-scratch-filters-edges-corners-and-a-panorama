"""
Classical Computer Vision from Scratch: Filters, Edges, Corners and a Panorama

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - convolve2d
import numpy as np

def convolve2d(img, kernel, pad="reflect"):
    """
    Perform true 2-D convolution on a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale input image.
    kernel : np.ndarray
        2-D convolution kernel with odd height and width.
    pad : str, default="reflect"
        Border handling method:
        - "reflect": Reflect image values at the borders.
        - "edge": Repeat border pixel values.
        - "zero": Pad the image with zeros.

    Returns
    -------
    np.ndarray
        Float array containing the convolved image with the same
        dimensions as the input image.
    """

    # Convert inputs into NumPy arrays.
    img = np.asarray(img)
    kernel = np.asarray(kernel)

    # Validate that the input image is 2-D.
    if img.ndim != 2:
        raise ValueError("Image must be a 2-D grayscale array.")

    # Validate that the kernel is 2-D with odd dimensions.
    if kernel.ndim != 2 or kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError("Kernel must be a 2-D array with odd height and width.")

    # Validate the padding mode.
    if pad not in ("reflect", "edge", "zero"):
        raise ValueError("Padding must be 'reflect', 'edge', or 'zero'.")

    # Convert image and kernel to floating-point arrays.
    img = img.astype(float, copy=False)
    kernel = kernel.astype(float, copy=False)

    # Get image and kernel dimensions.
    h, w = img.shape
    kh, kw = kernel.shape

    # Calculate the padding radii.
    rh = kh // 2
    rw = kw // 2

    # Flip the kernel horizontally and vertically.
    # This ensures true convolution rather than cross-correlation.
    flipped_kernel = kernel[::-1, ::-1]

    # Apply the requested border padding.
    if pad == "reflect":
        padded_img = np.pad(
            img,
            ((rh, rh), (rw, rw)),
            mode="reflect"
        )

    elif pad == "edge":
        padded_img = np.pad(
            img,
            ((rh, rh), (rw, rw)),
            mode="edge"
        )

    else:  # pad == "zero"
        padded_img = np.pad(
            img,
            ((rh, rh), (rw, rw)),
            mode="constant",
            constant_values=0
        )

    # Initialize the output image as a floating-point array.
    output = np.zeros((h, w), dtype=float)

    # Accumulate contributions from each kernel entry.
    # Vectorized image operations avoid looping over individual pixels.
    for i in range(kh):
        for j in range(kw):

            # Extract the image region corresponding to this kernel entry.
            region = padded_img[i:i + h, j:j + w]

            # Multiply the region by the corresponding flipped kernel value.
            output += flipped_kernel[i, j] * region

    return output

# Step 2 - gaussian_kernel1d
def gaussian_kernel1d(sigma, radius=None):
    """
    Generate a normalized 1-D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian distribution.
        Must be strictly positive.

    radius : int, optional
        Radius of the kernel. The resulting kernel has length
        2 * radius + 1.
        If None, the default radius is ceil(3 * sigma).

    Returns
    -------
    np.ndarray
        A normalized 1-D Gaussian kernel whose entries sum to 1.
    """

    # Validate the standard deviation.
    if sigma <= 0:
        raise ValueError("Sigma must be greater than zero.")

    # Use ceil(3 * sigma) as the default radius.
    # This captures approximately 99.7% of the Gaussian mass.
    if radius is None:
        radius = int(np.ceil(3 * sigma))

    # Generate coordinates from -radius to +radius.
    x = np.arange(-radius, radius + 1, dtype=float)

    # Evaluate the Gaussian function at each coordinate.
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))

    # Normalize the kernel so that all entries sum to 1.
    kernel /= np.sum(kernel)

    return kernel

# Step 3 - gaussian_blur
def gaussian_blur(img, sigma):
    """
    Apply Gaussian blur to a grayscale image using separable convolution.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale input image.

    sigma : float
        Standard deviation of the Gaussian kernel.
        A value of zero returns an unchanged floating-point copy.

    Returns
    -------
    np.ndarray
        Blurred image as a floating-point array with the same
        dimensions as the input image.
    """

    # Convert the input image to floating point.
    img = np.asarray(img, dtype=float)

    # If sigma is zero, return a new floating-point copy without blurring.
    if sigma == 0:
        return img.copy()

    # Generate the normalized 1-D Gaussian kernel.
    g = gaussian_kernel1d(sigma)

    # First pass: convolve horizontally using a row kernel.
    horizontal_blur = convolve2d(
        img,
        g[None, :],
        pad="reflect"
    )

    # Second pass: convolve vertically using a column kernel.
    blurred_img = convolve2d(
        horizontal_blur,
        g[:, None],
        pad="reflect"
    )

    return blurred_img

# Step 4 - sobel_gradients
def sobel_gradients(img):
    """
    Compute image gradients using the normalized Sobel operator.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale input image.

    Returns
    -------
    gx : np.ndarray
        Horizontal gradient (positive when brightness increases to the right).

    gy : np.ndarray
        Vertical gradient (positive when brightness increases downward).

    mag : np.ndarray
        Gradient magnitude.

    theta : np.ndarray
        Gradient direction in radians, ranging from -pi to pi.
    """

    # Convert the input image to a floating-point NumPy array.
    img = np.asarray(img, dtype=float)

    # Define the normalized horizontal Sobel kernel.
    # The factor 1/8 ensures that a unit-slope ramp produces a gradient of 1.
    kx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ], dtype=float) / 8.0

    # The vertical Sobel kernel is the transpose of the horizontal kernel.
    ky = kx.T

    # Compute horizontal and vertical gradients.
    # convolve2d uses reflect padding by default and flips the kernels.
    gx = convolve2d(img, kx)
    gy = convolve2d(img, ky)

    # Calculate the gradient magnitude.
    mag = np.sqrt(gx ** 2 + gy ** 2)

    # Calculate the gradient direction in radians.
    theta = np.arctan2(gy, gx)

    return gx, gy, mag, theta

# Step 5 - non_max_suppression
def non_max_suppression(mag, theta):
    """
    Apply Non-Maximum Suppression to thin gradient magnitudes
    into approximately one-pixel-wide edges.

    Parameters
    ----------
    mag : np.ndarray
        2-D gradient magnitude array.

    theta : np.ndarray
        2-D gradient direction array in radians.

    Returns
    -------
    np.ndarray
        Gradient magnitude array containing only local maxima
        along their quantized gradient directions. Other pixels
        are set to zero.
    """

    # Convert inputs to floating-point NumPy arrays.
    mag = np.asarray(mag, dtype=float)
    theta = np.asarray(theta, dtype=float)

    # Validate input dimensions.
    if mag.ndim != 2 or theta.ndim != 2:
        raise ValueError("mag and theta must be 2-D arrays.")

    # Ensure both arrays have identical dimensions.
    if mag.shape != theta.shape:
        raise ValueError("mag and theta must have the same shape.")

    h, w = mag.shape

    # Convert gradient directions from radians to degrees.
    # Modulo 180 makes opposite gradient directions equivalent.
    angle = np.mod(np.rad2deg(theta), 180.0)

    # Quantize each gradient direction into one of four bins:
    # 0, 45, 90, or 135 degrees.
    bins = np.floor((angle + 22.5) / 45.0).astype(int) % 4

    # Define the neighbor direction (dr, dc) for each bin.
    # These directions point toward the neighbor ahead.
    directions = (
        (0, 1),    # 0 degrees
        (1, 1),    # 45 degrees
        (1, 0),    # 90 degrees
        (1, -1)    # 135 degrees
    )

    # Pad the magnitude array with zeros.
    # This ensures that out-of-image neighbors contribute zero.
    padded = np.pad(
        mag,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=0
    )

    # Initialize the output array with zeros.
    suppressed = np.zeros_like(mag, dtype=float)

    # Process each quantized gradient direction separately.
    for bin_index, (dr, dc) in enumerate(directions):

        # Extract the neighbor behind each pixel (p - d).
        behind = padded[
            1 - dr:1 - dr + h,
            1 - dc:1 - dc + w
        ]

        # Extract the neighbor ahead of each pixel (p + d).
        ahead = padded[
            1 + dr:1 + dr + h,
            1 + dc:1 + dc + w
        ]

        # Keep pixels that:
        # 1. Belong to the current direction bin.
        # 2. Are greater than or equal to the neighbor behind.
        # 3. Are strictly greater than the neighbor ahead.
        #
        # The asymmetric comparison ensures deterministic
        # tie-breaking when two neighboring pixels have equal values.
        keep = (
            (bins == bin_index) &
            (mag >= behind) &
            (mag > ahead)
        )

        # Preserve the original magnitude at selected pixels.
        suppressed[keep] = mag[keep]

    return suppressed

# Step 6 - hysteresis_threshold
def hysteresis_threshold(strength, low, high):
    """
    Apply hysteresis thresholding to retain weak edges connected
    to at least one strong edge.

    Parameters
    ----------
    strength : np.ndarray
        2-D array containing edge strength values.

    low : float
        Lower threshold. Pixels with strength >= low are
        classified as weak edges.

    high : float
        Upper threshold. Pixels with strength >= high are
        classified as strong edges.

    Returns
    -------
    np.ndarray
        Boolean array indicating retained edge pixels.
        Strong edges and weak edges connected to strong edges
        are True; all other pixels are False.
    """

    # Convert the input to a NumPy array.
    strength = np.asarray(strength)

    # Validate that the input is a 2-D array.
    if strength.ndim != 2:
        raise ValueError("Strength must be a 2-D array.")

    # Identify weak and strong edge pixels.
    weak = strength >= low
    strong = strength >= high

    # Initialize the output with all strong edge pixels.
    edges = strong.copy()

    # Get the image dimensions.
    h, w = strength.shape

    # Initialize a stack with the coordinates of all strong pixels.
    # These pixels act as starting points for edge connectivity.
    stack = [tuple(point) for point in np.argwhere(strong)]

    # Perform iterative depth-first search to find connected weak edges.
    while stack:

        # Retrieve the current strong or already-connected edge pixel.
        r, c = stack.pop()

        # Examine all eight neighboring pixels, including diagonals.
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):

                # Skip the current pixel itself.
                if dr == 0 and dc == 0:
                    continue

                nr = r + dr
                nc = c + dc

                # Check that the neighboring pixel is inside the image.
                if 0 <= nr < h and 0 <= nc < w:

                    # Retain weak neighbors that have not already been visited.
                    if weak[nr, nc] and not edges[nr, nc]:

                        # Mark the connected weak edge as retained.
                        edges[nr, nc] = True

                        # Continue searching from this newly connected pixel.
                        stack.append((nr, nc))

    return edges

# Step 7 - canny
def canny(img, sigma=1.0, low=0.02, high=0.05):
    """
    Perform Canny edge detection on a grayscale image.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale input image.

    sigma : float, default=1.0
        Standard deviation of the Gaussian smoothing kernel.

    low : float, default=0.02
        Lower threshold for weak edge detection.

    high : float, default=0.05
        Upper threshold for strong edge detection.

    Returns
    -------
    np.ndarray
        Boolean edge map with the same dimensions as the input image.
    """

    # Step 1: Smooth the image using Gaussian blur.
    # This reduces noise before computing image gradients.
    blurred = gaussian_blur(img, sigma)

    # Step 2: Compute horizontal and vertical gradients,
    # gradient magnitude, and gradient direction using Sobel operators.
    _, _, mag, theta = sobel_gradients(blurred)

    # Step 3: Apply Non-Maximum Suppression to retain
    # only local maxima along the gradient direction.
    thinned = non_max_suppression(mag, theta)

    # Step 4: Apply hysteresis thresholding.
    # Retain strong edges and weak edges connected to strong edges.
    edges = hysteresis_threshold(thinned, low, high)

    # Return the final Boolean edge map.
    return edges

# Step 8 - structure_tensor
def structure_tensor(img, sigma_d=1.0, sigma_i=2.0):
    """
    Compute the three components of the image structure tensor.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale input image.

    sigma_d : float, default=1.0
        Gaussian standard deviation used for smoothing the image
        before computing its gradients (differentiation scale).

    sigma_i : float, default=2.0
        Gaussian standard deviation used for smoothing the
        gradient products (integration scale).

    Returns
    -------
    Sxx : np.ndarray
        Smoothed squared horizontal gradient.

    Syy : np.ndarray
        Smoothed squared vertical gradient.

    Sxy : np.ndarray
        Smoothed product of horizontal and vertical gradients.
    """

    # Convert the input image to a floating-point NumPy array.
    img = np.asarray(img, dtype=float)

    # Step 1: Smooth the image at the differentiation scale.
    # This reduces noise before calculating image gradients.
    img_blurred = gaussian_blur(img, sigma_d)

    # Step 2: Compute horizontal and vertical Sobel gradients.
    gx, gy, _, _ = sobel_gradients(img_blurred)

    # Step 3: Calculate the gradient products and smooth them
    # at the integration scale.
    Sxx = gaussian_blur(gx * gx, sigma_i)
    Syy = gaussian_blur(gy * gy, sigma_i)
    Sxy = gaussian_blur(gx * gy, sigma_i)

    # Return the three structure tensor components.
    return Sxx, Syy, Sxy

# Step 9 - corner_response
def corner_response(Sxx, Syy, Sxy, k=0.05, method="harris"):
    """
    Calculate corner response scores using Harris or Shi-Tomasi methods.

    Parameters
    ----------
    Sxx : np.ndarray or float
        Smoothed squared horizontal gradient.

    Syy : np.ndarray or float
        Smoothed squared vertical gradient.

    Sxy : np.ndarray or float
        Smoothed product of horizontal and vertical gradients.

    k : float, default=0.05
        Empirical sensitivity parameter used by the Harris method.

    method : str, default="harris"
        Corner detection method:
        - "harris": Harris corner response.
        - "shi-tomasi": Smaller eigenvalue of the structure tensor.

    Returns
    -------
    np.ndarray
        Corner response scores with the same shape as the input arrays.
    """

    # Convert the structure tensor components to floating-point arrays.
    Sxx = np.asarray(Sxx, dtype=float)
    Syy = np.asarray(Syy, dtype=float)
    Sxy = np.asarray(Sxy, dtype=float)

    # Calculate the determinant and trace of the structure tensor.
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy

    # Calculate the corner response using the selected method.
    if method == "harris":

        # Harris response: determinant - k * (trace)^2.
        response = det - k * (trace ** 2)

    elif method == "shi-tomasi":

        # Calculate half of the trace.
        half_trace = trace / 2.0

        # Calculate the discriminant under the square root.
        # Clip negative values to zero to avoid numerical errors.
        discriminant = np.maximum(
            half_trace ** 2 - det,
            0.0
        )

        # The smaller eigenvalue is the Shi-Tomasi corner score.
        response = half_trace - np.sqrt(discriminant)

    else:
        # Reject unsupported corner detection methods.
        raise ValueError(
            "Method must be either 'harris' or 'shi-tomasi'."
        )

    return np.asarray(response)

# Step 10 - local_maxima
def local_maxima(R, radius=3, threshold_rel=0.01, max_points=None):
    """
    Detect local maxima in a corner response map.

    Parameters
    ----------
    R : np.ndarray
        2-D corner response map.

    radius : int, default=3
        Radius of the local neighborhood.
        Each pixel is compared with all values in a
        (2 * radius + 1) x (2 * radius + 1) window.

    threshold_rel : float, default=0.01
        Relative threshold. A pixel must have a response
        strictly greater than threshold_rel * R.max().

    max_points : int, optional
        Maximum number of strongest local maxima to return.
        If None, all qualifying maxima are returned.

    Returns
    -------
    np.ndarray
        Float array of shape (N, 2) containing coordinates
        in (x, y) = (column, row) order.

        Points are sorted by descending response strength.
        Ties are resolved by ascending row, then ascending column.
    """

    # Convert the response map into a floating-point NumPy array.
    R = np.asarray(R, dtype=float)

    # Validate that the response map is two-dimensional.
    if R.ndim != 2:
        raise ValueError("R must be a 2-D array.")

    # Validate the neighborhood radius.
    if not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("Radius must be a non-negative integer.")

    # Validate the maximum number of points.
    if max_points is not None:
        if not isinstance(max_points, (int, np.integer)) or max_points < 0:
            raise ValueError("max_points must be a non-negative integer or None.")

    # Return an empty coordinate array if the response map is empty.
    if R.size == 0:
        return np.empty((0, 2), dtype=float)

    # Calculate the absolute response threshold.
    threshold = threshold_rel * np.max(R)

    # Pad the response map with negative infinity.
    # This ensures that pixels outside the image do not
    # interfere with local maximum detection.
    padded = np.pad(
        R,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=-np.inf
    )

    # Get the dimensions of the response map.
    h, w = R.shape

    # Initialize the local maximum map.
    local_max = np.full(R.shape, -np.inf, dtype=float)

    # Find the maximum response inside each neighborhood.
    # Vectorized slicing avoids looping over individual pixels.
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):

            # Extract the neighboring response values.
            neighbor = padded[
                radius + dr:radius + dr + h,
                radius + dc:radius + dc + w
            ]

            # Update the maximum response at every pixel.
            local_max = np.maximum(local_max, neighbor)

    # A pixel qualifies as a peak when:
    # 1. Its response is greater than or equal to every
    #    value in its neighborhood.
    # 2. Its response is strictly greater than the threshold.
    #
    # Using >= allows equal neighboring values to both
    # qualify as local maxima, as required.
    peaks = (R >= local_max) & (R > threshold)

    # Obtain peak coordinates in (row, column) order.
    coordinates = np.argwhere(peaks)

    # Return an empty (0, 2) float array if no peaks are found.
    if coordinates.size == 0:
        return np.empty((0, 2), dtype=float)

    # Separate row and column coordinates.
    rows = coordinates[:, 0]
    cols = coordinates[:, 1]

    # Retrieve the response strength at each detected peak.
    values = R[rows, cols]

    # Sort by:
    # 1. Descending response strength.
    # 2. Ascending row index.
    # 3. Ascending column index.
    #
    # np.lexsort uses the last supplied key as the primary key.
    order = np.lexsort((cols, rows, -values))

    # Apply the sorting order.
    coordinates = coordinates[order]

    # Limit the number of returned points when max_points is specified.
    if max_points is not None:
        coordinates = coordinates[:max_points]

    # Convert (row, column) into (x, y) = (column, row).
    # Return floating-point coordinates as required.
    return coordinates[:, ::-1].astype(float)

# Step 11 - refine_subpixel
def refine_subpixel(R, pts):
    """
    Refine integer corner coordinates to sub-pixel accuracy
    using independent parabolic interpolation along x and y.

    Parameters
    ----------
    R : np.ndarray
        2-D corner response map.

    pts : np.ndarray
        Array of corner coordinates with shape (N, 2).
        Coordinates are provided in (x, y) = (column, row) order.

    Returns
    -------
    np.ndarray
        New floating-point array of shape (N, 2) containing
        the refined sub-pixel corner coordinates.
    """

    # Convert the response map and corner coordinates to floating-point arrays.
    R = np.asarray(R, dtype=float)
    pts = np.asarray(pts, dtype=float)

    # Validate the response map.
    if R.ndim != 2:
        raise ValueError("R must be a 2-D array.")

    # Handle an empty set of corner coordinates.
    if pts.size == 0:
        return np.empty((0, 2), dtype=float)

    # Validate the corner coordinate dimensions.
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (N, 2).")

    # Create a new array so that the original points are not modified.
    refined = pts.copy()

    # Get the dimensions of the response map.
    h, w = R.shape

    # Refine each detected corner independently.
    for i, (x, y) in enumerate(pts):

        # Convert the input coordinates to their nearest integer pixels.
        xi = int(np.rint(x))
        yi = int(np.rint(y))

        # Start with the integer pixel coordinates.
        refined[i, 0] = xi
        refined[i, 1] = yi

        # Skip refinement if the rounded center is outside the image.
        if not (0 <= xi < w and 0 <= yi < h):
            continue

        # ---------------------------------------------------------
        # Horizontal sub-pixel refinement (x direction).
        # ---------------------------------------------------------

        # A valid horizontal neighborhood requires the center pixel
        # to have one neighbor on both its left and right.
        if 0 < xi < w - 1:

            # Get the three response values along the horizontal direction.
            a = R[yi, xi - 1]
            c = R[yi, xi]
            b = R[yi, xi + 1]

            # Calculate the second difference of the parabola.
            denominator = a - 2.0 * c + b

            # Refine only when the center is a local maximum
            # and the fitted parabola is concave.
            if (
                np.isfinite(a) and
                np.isfinite(c) and
                np.isfinite(b) and
                c >= a and
                c >= b and
                denominator < 0
            ):
                # Calculate the horizontal displacement to the vertex.
                dx = 0.5 * (a - b) / denominator

                # Update the horizontal coordinate.
                refined[i, 0] = xi + dx

        # ---------------------------------------------------------
        # Vertical sub-pixel refinement (y direction).
        # ---------------------------------------------------------

        # A valid vertical neighborhood requires the center pixel
        # to have one neighbor above and one below.
        if 0 < yi < h - 1:

            # Get the three response values along the vertical direction.
            a = R[yi - 1, xi]
            c = R[yi, xi]
            b = R[yi + 1, xi]

            # Calculate the second difference of the parabola.
            denominator = a - 2.0 * c + b

            # Refine only when the center is a local maximum
            # and the fitted parabola is concave.
            if (
                np.isfinite(a) and
                np.isfinite(c) and
                np.isfinite(b) and
                c >= a and
                c >= b and
                denominator < 0
            ):
                # Calculate the vertical displacement to the vertex.
                dy = 0.5 * (a - b) / denominator

                # Update the vertical coordinate.
                refined[i, 1] = yi + dy

    return refined.astype(float)

# Step 12 - harris_corners
def harris_corners(img, max_points=500, border=10, sigma_d=1.0, sigma_i=2.0, k=0.05, radius=3,
                   threshold_rel=0.001, method="harris", subpixel=True):
    """
    Detect corners using the Harris or Shi-Tomasi corner detection method.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale input image.

    max_points : int, default=500
        Maximum number of strongest corners to return.

    border : int, default=10
        Number of pixels excluded along each image boundary.

    sigma_d : float, default=1.0
        Gaussian standard deviation for the differentiation scale.

    sigma_i : float, default=2.0
        Gaussian standard deviation for the integration scale.

    k : float, default=0.05
        Harris detector sensitivity parameter.

    radius : int, default=3
        Neighborhood radius for local maximum detection.

    threshold_rel : float, default=0.001
        Relative threshold for selecting corner responses.

    method : str, default="harris"
        Corner detection method: "harris" or "shi-tomasi".

    subpixel : bool, default=True
        Whether to refine detected corner coordinates to sub-pixel accuracy.

    Returns
    -------
    np.ndarray
        Floating-point array of shape (N, 2) containing corner coordinates
        in (x, y) = (column, row) order, strongest first.
    """

    # Convert the input image to a floating-point NumPy array.
    img = np.asarray(img, dtype=float)

    # Step 1: Compute the structure tensor at the specified scales.
    Sxx, Syy, Sxy = structure_tensor(
        img,
        sigma_d=sigma_d,
        sigma_i=sigma_i
    )

    # Step 2: Calculate the corner response using the selected method.
    R = corner_response(
        Sxx,
        Syy,
        Sxy,
        k=k,
        method=method
    )

    # Step 3: Create a copy of the response map for border masking.
    # The original response R is preserved for sub-pixel refinement.
    R_masked = R.copy()

    # Replace the boundary regions with the minimum response value.
    # This prevents unwanted corner detections near image edges.
    if border > 0:
        min_response = R.min()

        R_masked[:border, :] = min_response
        R_masked[-border:, :] = min_response
        R_masked[:, :border] = min_response
        R_masked[:, -border:] = min_response

    # Step 4: Detect local maxima and select the strongest candidates.
    pts = local_maxima(
        R_masked,
        radius=radius,
        threshold_rel=threshold_rel,
        max_points=max_points
    )

    # Step 5: Optionally refine the detected corners to sub-pixel accuracy.
    # Use the original, unmasked response map for accurate interpolation.
    if subpixel:
        pts = refine_subpixel(R, pts)

    return pts

# Step 13 - apply_homography
def apply_homography(H, pts):
    """
    Transform 2-D points using a 3 x 3 homography matrix.

    Parameters
    ----------
    H : np.ndarray
        3 x 3 homography transformation matrix.

    pts : np.ndarray
        Input points with shape (N, 2), or a single point
        with shape (2,). Coordinates are in (x, y) order.

    Returns
    -------
    np.ndarray
        Transformed points as a floating-point array
        with shape (N, 2).
    """

    # Convert the homography matrix and input points to floating-point arrays.
    H = np.asarray(H, dtype=float)
    pts = np.asarray(pts, dtype=float)

    # Validate the homography matrix dimensions.
    if H.shape != (3, 3):
        raise ValueError("H must be a 3 x 3 homography matrix.")

    # Accept a single point with shape (2,).
    # Convert it into a one-row array with shape (1, 2).
    if pts.ndim == 1:
        if pts.shape != (2,):
            raise ValueError("A single point must have shape (2,).")
        pts = pts.reshape(1, 2)

    # Validate the input point array dimensions.
    elif pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (N, 2) or (2,).")

    # Convert Cartesian coordinates (x, y) into homogeneous coordinates
    # by appending a third coordinate equal to 1.
    pts_h = np.column_stack((
        pts,
        np.ones(pts.shape[0], dtype=float)
    ))

    # Apply the homography transformation.
    # Each homogeneous point is represented as a column vector,
    # so the transformation is equivalent to H @ point.
    transformed_h = pts_h @ H.T

    # Perform perspective division using the third homogeneous coordinate.
    # This converts the transformed homogeneous coordinates back
    # into ordinary 2-D Cartesian coordinates.
    with np.errstate(divide="ignore", invalid="ignore"):
        transformed = (
            transformed_h[:, :2] / transformed_h[:, 2:3]
        )

    # Return the transformed points as a floating-point array.
    return transformed.astype(float)

# Step 14 - normalize_points
def normalize_points(pts):
    """
    Normalize 2-D points using Hartley's normalization method.

    The transformation translates the centroid of the points
    to the origin and scales them so that their mean distance
    from the origin is sqrt(2).

    Parameters
    ----------
    pts : np.ndarray
        Input point coordinates with shape (N, 2).
        Coordinates are provided in (x, y) order.

    Returns
    -------
    pts_n : np.ndarray
        Normalized points with shape (N, 2).

    T : np.ndarray
        3 x 3 similarity transformation matrix used
        to normalize the original points.

    Raises
    ------
    ValueError
        If the input is not a valid (N, 2) array,
        contains no points, or all points coincide.
    """

    # Convert input points to a floating-point NumPy array.
    pts = np.asarray(pts, dtype=float)

    # Validate the input dimensions.
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (N, 2).")

    # Ensure that at least one point is provided.
    if pts.shape[0] == 0:
        raise ValueError("At least one point is required.")

    # Step 1: Calculate the centroid of the input points.
    centroid = np.mean(pts, axis=0)

    cx, cy = centroid

    # Step 2: Calculate the Euclidean distance of each point
    # from the centroid.
    distances = np.linalg.norm(pts - centroid, axis=1)

    # Calculate the mean distance from the centroid.
    mean_distance = np.mean(distances)

    # All points coincide when their mean distance is zero.
    # Normalization is undefined in this case.
    if mean_distance == 0.0:
        raise ValueError("All points coincide; normalization is undefined.")

    # Step 3: Calculate the uniform scaling factor.
    # This makes the mean distance from the origin equal to sqrt(2).
    scale = np.sqrt(2.0) / mean_distance

    # Step 4: Construct the 3 x 3 similarity transformation matrix.
    # The matrix first translates the centroid to the origin
    # and then applies uniform scaling.
    T = np.array([
        [scale, 0.0, -scale * cx],
        [0.0, scale, -scale * cy],
        [0.0, 0.0, 1.0]
    ], dtype=float)

    # Step 5: Apply the transformation to the original points.
    # Reuse apply_homography() implemented in Step 13.
    pts_n = apply_homography(T, pts)

    return pts_n, T

# Step 15 - dlt_homography
def dlt_homography(p1, p2, normalize=True):
    """
    Estimate a 3 x 3 homography matrix using the Direct Linear Transform (DLT).

    Parameters
    ----------
    p1 : np.ndarray
        Source point coordinates with shape (N, 2).

    p2 : np.ndarray
        Corresponding destination point coordinates with shape (N, 2).

    normalize : bool, default=True
        Whether to apply Hartley normalization before constructing
        the DLT matrix.

    Returns
    -------
    np.ndarray
        Estimated 3 x 3 homography matrix H satisfying:

            p2 ~ H @ p1

        The matrix is normalized so that H[2, 2] = 1.

    Raises
    ------
    ValueError
        If fewer than four correspondences are provided,
        the point arrays have different lengths, or their shapes
        are invalid.
    """

    # Convert both point sets into floating-point NumPy arrays.
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    # Validate the dimensions of both point arrays.
    if p1.ndim != 2 or p1.shape[1] != 2:
        raise ValueError("p1 must have shape (N, 2).")

    if p2.ndim != 2 or p2.shape[1] != 2:
        raise ValueError("p2 must have shape (N, 2).")

    # Both point sets must contain the same number of correspondences.
    if len(p1) != len(p2):
        raise ValueError("p1 and p2 must contain the same number of points.")

    # At least four point correspondences are required to estimate
    # a homography using the Direct Linear Transform.
    if len(p1) < 4:
        raise ValueError("At least four point correspondences are required.")

    # ---------------------------------------------------------
    # Step 1: Hartley normalization.
    # ---------------------------------------------------------

    if normalize:

        # Normalize source and destination point sets independently.
        p1_n, T1 = normalize_points(p1)
        p2_n, T2 = normalize_points(p2)

    else:

        # Use the original point coordinates without normalization.
        p1_n = p1
        p2_n = p2

        # Identity matrices represent the absence of normalization.
        T1 = np.eye(3, dtype=float)
        T2 = np.eye(3, dtype=float)

    # ---------------------------------------------------------
    # Step 2: Construct the DLT matrix A (2N x 9).
    # ---------------------------------------------------------

    A = []

    for (x, y), (u, v) in zip(p1_n, p2_n):

        # First row corresponds to the transformed x-coordinate.
        A.append([
            -x, -y, -1.0,
             0.0, 0.0, 0.0,
             u * x, u * y, u
        ])

        # Second row corresponds to the transformed y-coordinate.
        A.append([
             0.0, 0.0, 0.0,
            -x, -y, -1.0,
             v * x, v * y, v
        ])

    # Convert the DLT matrix into a NumPy array.
    A = np.asarray(A, dtype=float)

    # ---------------------------------------------------------
    # Step 3: Solve the homogeneous system using SVD.
    # ---------------------------------------------------------

    # Decompose A into U, S and Vt.
    #
    # full_matrices=True is important for exactly four
    # correspondences because A then has shape (8, 9).
    # It preserves the ninth right singular vector corresponding
    # to the null space of A.
    _, _, Vt = np.linalg.svd(A, full_matrices=True)

    # The right singular vector corresponding to the smallest
    # singular value gives the homography parameters.
    h = Vt[-1]

    # Reshape the nine parameters into a 3 x 3 homography matrix.
    H_normalized = h.reshape(3, 3)

    # ---------------------------------------------------------
    # Step 4: Denormalize the estimated homography.
    # ---------------------------------------------------------

    # Convert the normalized homography back to the original
    # source and destination coordinate systems.
    #
    # H = inv(T2) @ H_normalized @ T1
    H = np.linalg.inv(T2) @ H_normalized @ T1

    # Ensure that the homography can be scaled to H[2, 2] = 1.
    if not np.isfinite(H[2, 2]) or H[2, 2] == 0.0:
        raise ValueError(
            "The estimated homography cannot be normalized to H[2, 2] = 1."
        )

    # Normalize the homography matrix.
    H /= H[2, 2]

    return H

# Step 16 - transfer_error
def transfer_error(H, p1, p2):
    """
    Calculate the homography transfer error for corresponding points.

    Parameters
    ----------
    H : np.ndarray
        3 x 3 homography transformation matrix.

    p1 : np.ndarray
        Source points with shape (N, 2).

    p2 : np.ndarray
        Corresponding destination points with shape (N, 2).

    Returns
    -------
    np.ndarray
        Floating-point array of shape (N,) containing the
        Euclidean transfer error for each correspondence.
    """

    # Convert point coordinates to floating-point NumPy arrays.
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    # Validate that both point arrays have shape (N, 2).
    if p1.ndim != 2 or p1.shape[1] != 2:
        raise ValueError("p1 must have shape (N, 2).")

    if p2.ndim != 2 or p2.shape[1] != 2:
        raise ValueError("p2 must have shape (N, 2).")

    # Ensure that both point arrays contain the same number of points.
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("p1 and p2 must contain the same number of points.")

    # Transform source points into the destination coordinate system.
    mapped_points = apply_homography(H, p1)

    # Calculate the Euclidean distance between transformed points
    # and their corresponding destination points.
    errors = np.linalg.norm(mapped_points - p2, axis=1)

    return errors.astype(float)

# Step 17 - describe_patches
def describe_patches(img, pts, size=8, spacing=2, sigma=None):
    """
    Generate normalized patch descriptors around detected points.

    Parameters
    ----------
    img : np.ndarray
        2-D grayscale input image.

    pts : np.ndarray
        Corner coordinates with shape (N, 2).
        Coordinates are provided in (x, y) = (column, row) order.

    size : int, default=8
        Number of samples along each dimension of the patch.
        Each descriptor contains size * size elements.

    spacing : int, default=2
        Distance in pixels between consecutive sampling locations.

    sigma : float, optional
        Standard deviation of the Gaussian blur.
        Defaults to spacing when None.

    Returns
    -------
    desc : np.ndarray
        Normalized patch descriptors with shape
        (number_of_kept_points, size * size).

    keep : np.ndarray
        Boolean array of shape (N,).
        True indicates that the complete sampling grid
        around the corresponding point lies inside the image.

    Raises
    ------
    ValueError
        If spacing * (size - 1) is odd, or input dimensions
        are invalid.
    """

    # Convert the image and point coordinates into NumPy arrays.
    img = np.asarray(img, dtype=float)
    pts = np.asarray(pts, dtype=float)

    # Validate the grayscale image.
    if img.ndim != 2:
        raise ValueError("img must be a 2-D grayscale image.")

    # Validate the point coordinates.
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("pts must have shape (N, 2).")

    # Ensure that patch size is a positive integer.
    if not isinstance(size, (int, np.integer)) or size <= 0:
        raise ValueError("size must be a positive integer.")

    # Ensure that spacing is an integer.
    if not isinstance(spacing, (int, np.integer)):
        raise ValueError("spacing must be an integer.")

    # Calculate the total sampling span.
    span = spacing * (size - 1)

    # The sampling offsets must be integers symmetric about zero.
    # An odd span would place the sampling grid between pixel centers.
    if span % 2 != 0:
        raise ValueError(
            "spacing * (size - 1) must be even."
        )

    # If sigma is not specified, use spacing as the Gaussian
    # standard deviation.
    if sigma is None:
        sigma = spacing

    # Step 1: Apply Gaussian smoothing before extracting patches.
    # This helps reduce sensitivity to small image variations.
    blurred = gaussian_blur(img, sigma)

    # Step 2: Generate symmetric sampling offsets.
    # For the default size=8 and spacing=2:
    # offsets = [-7, -5, -3, -1, 1, 3, 5, 7].
    offsets = (
        np.arange(size, dtype=int) * spacing - span // 2
    )

    # Get image dimensions.
    h, w = blurred.shape

    # Initialize a Boolean mask for valid sampling locations.
    keep = np.zeros(len(pts), dtype=bool)

    # Store the normalized descriptors of valid points.
    descriptors = []

    # Step 3: Process each detected corner.
    for i, (x, y) in enumerate(pts):

        # Round the corner coordinates to the nearest integer pixel.
        x = int(np.rint(x))
        y = int(np.rint(y))

        # Calculate the complete sampling grid boundaries.
        x_min = x + offsets.min()
        x_max = x + offsets.max()

        y_min = y + offsets.min()
        y_max = y + offsets.max()

        # Check whether every sampling location lies inside the image.
        # Points with incomplete sampling grids are discarded.
        if (
            x_min < 0 or
            x_max >= w or
            y_min < 0 or
            y_max >= h
        ):
            continue

        # Mark this point as valid.
        keep[i] = True

        # Step 4: Extract the size x size patch.
        # Rows correspond to y + offsets and columns to x + offsets.
        # Advanced indexing preserves row-by-row sampling order.
        patch = blurred[
            y + offsets[:, None],
            x + offsets[None, :]
        ]

        # Flatten the patch into a one-dimensional descriptor.
        descriptor = patch.flatten()

        # Step 5: Normalize the descriptor.
        # Subtract the mean to obtain zero-mean values.
        descriptor = descriptor - descriptor.mean()

        # Divide by the standard deviation with a small epsilon
        # to prevent division by zero for constant patches.
        descriptor = descriptor / (
            descriptor.std() + 1e-8
        )

        # Store the normalized descriptor.
        descriptors.append(descriptor)

    # Convert the collected descriptors into a 2-D floating-point array.
    # The reshape also guarantees the correct shape when no points survive.
    desc = np.asarray(
        descriptors,
        dtype=float
    ).reshape(-1, size * size)

    return desc, keep

# Step 18 - match_descriptors
def match_descriptors(d1, d2, ratio=0.8):
    """
    Match feature descriptors using Euclidean distance
    and Lowe's ratio test.

    Parameters
    ----------
    d1 : np.ndarray
        Descriptors from the first image with shape (N1, D).

    d2 : np.ndarray
        Descriptors from the second image with shape (N2, D).
        At least two descriptors are required.

    ratio : float, default=0.8
        Lowe's ratio threshold. A match is accepted only when
        the nearest-neighbor distance is strictly smaller than
        ratio multiplied by the second-nearest-neighbor distance.

    Returns
    -------
    np.ndarray
        Integer array of shape (M, 2), where each row contains
        (i, j), representing a match between d1[i] and d2[j].

        Multiple descriptors from d1 may match the same descriptor
        in d2. Matches are returned in ascending order of i.

    Raises
    ------
    ValueError
        If d2 contains fewer than two descriptors or the input
        descriptor arrays have incompatible dimensions.
    """

    # Convert descriptor inputs to floating-point NumPy arrays.
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)

    # Validate that both descriptor arrays are 2-D.
    if d1.ndim != 2 or d2.ndim != 2:
        raise ValueError("d1 and d2 must be 2-D descriptor arrays.")

    # At least two descriptors are needed to identify
    # the nearest and second-nearest neighbors.
    if d2.shape[0] < 2:
        raise ValueError("d2 must contain at least two descriptors.")

    # Both descriptor sets must have the same feature dimension.
    if d1.shape[1] != d2.shape[1]:
        raise ValueError("d1 and d2 must have the same descriptor dimension.")

    # Store the accepted descriptor correspondences.
    matches = []

    # Process every descriptor from the first image.
    for i in range(d1.shape[0]):

        # Calculate Euclidean distances between the current descriptor
        # and every descriptor in the second image.
        distances = np.linalg.norm(
            d2 - d1[i],
            axis=1
        )

        # Sort descriptor indices by ascending distance.
        # Stable sorting ensures that equal-distance ties are resolved
        # in favor of the smaller index in d2.
        nearest_indices = np.argsort(
            distances,
            kind="stable"
        )

        # Retrieve the nearest and second-nearest descriptor indices.
        j1 = nearest_indices[0]
        j2 = nearest_indices[1]

        # Retrieve their corresponding Euclidean distances.
        d_best = distances[j1]
        d_second = distances[j2]

        # Apply Lowe's ratio test.
        # The strict inequality rejects ambiguous matches.
        if d_best < ratio * d_second:

            # Store the correspondence as (index in d1, index in d2).
            matches.append((i, j1))

    # Convert the accepted matches into an integer NumPy array.
    # Reshape guarantees (0, 2) when no matches are found.
    return np.asarray(
        matches,
        dtype=int
    ).reshape(-1, 2)

# Step 19 - corner_repeatability
def corner_repeatability(pts1, pts2, H, shape2, tol=1.5):
    """
    Measure corner repeatability between two images using
    the ground-truth homography.

    Parameters
    ----------
    pts1 : np.ndarray
        Corner coordinates from the first image with shape (N1, 2).
        Coordinates are provided in (x, y) = (column, row) order.

    pts2 : np.ndarray
        Corner coordinates from the second image with shape (N2, 2).

    H : np.ndarray
        3 x 3 homography mapping points from image 1 to image 2.

    shape2 : tuple
        Shape of the second image as (height, width).

    tol : float, default=1.5
        Maximum Euclidean distance in pixels for considering
        a transformed corner to have a match.

    Returns
    -------
    float
        Fraction of transformed corners inside image 2 that have
        at least one corresponding corner within the specified
        tolerance. Returns 0.0 if there are no valid points to score.
    """

    # Convert corner coordinates into floating-point NumPy arrays.
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)

    # Validate the input point arrays.
    if pts1.ndim != 2 or pts1.shape[1] != 2:
        raise ValueError("pts1 must have shape (N, 2).")

    if pts2.ndim != 2 or pts2.shape[1] != 2:
        raise ValueError("pts2 must have shape (N, 2).")

    # Extract the height and width of the second image.
    h2, w2 = shape2

    # An empty destination corner set cannot produce any matches.
    if len(pts2) == 0:
        return 0.0

    # Step 1: Transform the corners from image 1 into image 2
    # using the provided ground-truth homography.
    mapped_pts = apply_homography(H, pts1)

    # Step 2: Keep only transformed points that fall inside image 2.
    # Both coordinate boundaries are inclusive.
    inside = (
        (mapped_pts[:, 0] >= 0) &
        (mapped_pts[:, 0] <= w2 - 1) &
        (mapped_pts[:, 1] >= 0) &
        (mapped_pts[:, 1] <= h2 - 1)
    )

    # Extract only the transformed points that are inside image 2.
    valid_pts = mapped_pts[inside]

    # Return 0.0 if no transformed corner lies inside image 2.
    if len(valid_pts) == 0:
        return 0.0

    # Step 3: Calculate pairwise Euclidean distances between
    # valid transformed corners and corners detected in image 2.
    #
    # Broadcasting produces an array of shape (N_valid, N2, 2).
    differences = valid_pts[:, None, :] - pts2[None, :, :]

    # Calculate Euclidean distances for every point pair.
    distances = np.linalg.norm(differences, axis=2)

    # A transformed corner is repeatable if at least one
    # corner in image 2 is within the specified tolerance.
    matched = np.any(distances <= tol, axis=1)

    # Calculate the fraction of repeatable corners.
    repeatability = np.mean(matched)

    return float(repeatability)

