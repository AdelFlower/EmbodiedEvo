"""One-shot colour components and per-object camera calibration, in metres."""

import numpy as np

RED_HUE_DEGREES = 12.0
RED_SATURATION = .65
RED_VALUE = .45
WHITE_SATURATION = .15
WHITE_VALUE = .75


def classify(rgb):
    """Return red/white HSV masks; hue is measured about red in degrees."""
    rgb = np.asarray(rgb, dtype=float) / 255
    red, green, blue = np.moveaxis(rgb, -1, 0)
    value, minimum = rgb.max(axis=-1), rgb.min(axis=-1)
    delta = value - minimum
    saturation = np.divide(delta, value, out=np.zeros_like(value), where=value != 0)
    # Within ±12 degrees of red, R must be the largest channel.
    hue = np.divide(60 * (green - blue), delta, out=np.zeros_like(delta), where=delta != 0)
    return np.stack(((red == value) & (abs(hue) <= RED_HUE_DEGREES)
                     & (saturation >= RED_SATURATION) & (value >= RED_VALUE),
                     (saturation <= WHITE_SATURATION) & (value >= WHITE_VALUE)))


# Fixed workspace regions of interest (x0, y0, x1, y1) in the 640x480 cam_head image.
# They cover each object's reachable +/-3 cm layouts plus margin and exclude the robot's
# red/white parts at its fixed initial pose (measured on the reset reference frame).
ROIS = ((380, 215, 466, 286), (250, 175, 371, 281))
MIN_PIXELS = (40, 400)


def centroids(frame):
    """Return cube/plate centroids of all class pixels in their ROIs, or None if too few."""
    points = []
    for mask, (x0, y0, x1, y1), minimum in zip(classify(frame), ROIS, MIN_PIXELS):
        ys, xs = np.nonzero(mask[y0:y1, x0:x1])
        if len(xs) < minimum:
            return None
        points.append([xs.mean() + x0, ys.mean() + y0])
    return np.asarray(points)


def apply_homography(matrix, points):
    """Map (..., 2) pixel coordinates to (..., 2) table coordinates."""
    points = np.asarray(points)
    mapped = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1) @ np.asarray(matrix).T
    return mapped[..., :2] / mapped[..., 2:]


def fit_homography(pixels, xy):
    """Fit a normalized DLT homography to centroid/world correspondences."""
    def normalize(points):
        points = np.asarray(points)
        centre = points.mean(axis=0)
        scale = np.sqrt(2) / np.linalg.norm(points - centre, axis=1).mean()
        matrix = np.array([[scale, 0, -scale * centre[0]],
                           [0, scale, -scale * centre[1]], [0, 0, 1]])
        return apply_homography(matrix, points), matrix

    pixels, source = normalize(pixels)
    xy, target = normalize(xy)
    rows = []
    for (u, v), (x, y) in zip(pixels, xy):
        rows.extend(([-u, -v, -1, 0, 0, 0, x*u, x*v, x],
                     [0, 0, 0, -u, -v, -1, y*u, y*v, y]))
    _, _, vectors = np.linalg.svd(rows)
    matrix = np.linalg.inv(target) @ vectors[-1].reshape(3, 3) @ source
    return matrix / matrix[2, 2]


def image_detection(frame, calibration):
    """Detect XY using only pixels and the frozen calibration."""
    points = centroids(frame)
    if points is None:
        raise ValueError("Cube or plate colour component missing")
    return np.array([apply_homography(matrix, point)
                     for matrix, point in zip(calibration["homographies"], points)])


def simulated_detection(state, rng, sigma):
    """Read reset truth once and draw independent clipped detection errors."""
    truth = np.array([state["sample"]["cube_position"][:2], state["plate_position"][:2]])
    sigma = np.asarray(sigma)[:, None]
    return truth + np.clip(rng.standard_normal((2, 2)) * sigma, -3*sigma, 3*sigma)


def sigma_rule(errors):
    """Set cube/plate sigma from validation RMS, with 2/3 mm floors."""
    rms = np.sqrt(np.mean(np.asarray(errors)**2, axis=0))
    return np.maximum(1.5 * rms.max(axis=1), [.002, .003])
