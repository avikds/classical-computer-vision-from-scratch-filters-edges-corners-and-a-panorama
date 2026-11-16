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

