import cv2
import copy
import numpy as np

from scipy.ndimage import maximum_filter
from scipy.stats import linregress
from typing import List, Optional, Tuple


def generate_gaussian_matrix_vectorized(h, w, px, py, sigma=2):
    # Create a grid of indices
    x, y = np.meshgrid(np.arange(h), np.arange(w))

    # Calculate Gaussian values for the entire grid
    array = (-((x - px)**2 + (y - py)**2) / (2 * sigma**2)).astype(float)
    matrix = np.exp(array)

    return matrix

def resize_keypoints(keypoints, original_size, new_size):

    ratio_h = new_size[0] / original_size[0]
    ratio_w = new_size[1] / original_size[1]

    resized_keypoints = {}
    for kp, values in keypoints.items():
        x_resized = int(values['x'] * ratio_w)
        y_resized = int(values['y'] * ratio_h)
        resized_keypoints[kp] = {'x': x_resized, 'y': y_resized, 'in_frame': values['in_frame']}
        if 'proj_err' in values.keys():
            resized_keypoints[kp]['proj_err'] = values['proj_err']

    return resized_keypoints


def generate_gaussian_array_vectorized(num_matrices, keypoints, original_size, down_ratio=2, sigma=2, proj_err_th=5.):

    new_size = tuple(ti/down_ratio for ti in original_size)
    resized_keypoints = resize_keypoints(keypoints, original_size, new_size)

    # Create an array of center points based on resized keypoints
    center_points = []

    for kp in range(1, num_matrices):
        if kp in resized_keypoints.keys():
            if (resized_keypoints[kp]['in_frame']):
                if 'proj_err' in resized_keypoints[kp].keys():
                    if resized_keypoints[kp]['proj_err'] <= proj_err_th:
                        center_points.append([resized_keypoints[kp]['x'], resized_keypoints[kp]['y']])
                    else:
                        center_points.append([np.inf, np.inf])
                else:
                    center_points.append([resized_keypoints[kp]['x'], resized_keypoints[kp]['y']])
            else:
                center_points.append([np.inf, np.inf])
        else:
            center_points.append([np.inf, np.inf])

    center_points = np.array(center_points)

    # Generate Gaussian matrices for all center points
    matrices = [generate_gaussian_matrix_vectorized(new_size[0], new_size[1], px, py, sigma) for px, py in center_points]
    matrices = np.array(matrices)
    matrices = np.concatenate((matrices, 1-matrices.sum(axis=0, keepdims=True)), axis=0)
    matrices = np.clip(matrices, 0, 1)

    return matrices


def resize_keypoints_l(keypoints, original_size, new_size):
    ratio_h = new_size[0] / original_size[0]
    ratio_w = new_size[1] / original_size[1]

    resized_keypoints = {}
    for kp, values in keypoints.items():
        x1_resized = int(values['x_1'] * ratio_w)
        y1_resized = int(values['y_1'] * ratio_h)
        x2_resized = int(values['x_2'] * ratio_w)
        y2_resized = int(values['y_2'] * ratio_h)

        resized_keypoints[kp] = {'x_1': x1_resized, 'y_1': y1_resized, 'x_2': x2_resized, 'y_2': y2_resized}

    return resized_keypoints

def generate_gaussian_array_vectorized_l(num_matrices, keypoints, original_size, down_ratio=2, sigma=2, sigma_mult=1):

    def sigma_f(px, py, size, sigma):
        #multiply sigma if point in image border
        if (px < 5 or px > size[0] - 5) | (py < 5 or py > size[1] - 5):
            return sigma_mult*sigma
        else:
            return sigma

    new_size = tuple(int(ti / down_ratio) for ti in original_size)
    resized_keypoints = resize_keypoints_l(keypoints, original_size, new_size)

    # Create an array of center points based on resized keypoints for both points
    center_points = []

    for kp in range(1, num_matrices+1):
        if kp in resized_keypoints.keys():
            center_points.append([resized_keypoints[kp]['x_1'], resized_keypoints[kp]['y_1']])
            center_points.append([resized_keypoints[kp]['x_2'], resized_keypoints[kp]['y_2']])

        else:
            center_points.append([np.inf, np.inf])
            center_points.append([np.inf, np.inf])

    center_points = np.array(center_points)

    # Generate Gaussian matrices for both points and sum them
    matrices1 = [generate_gaussian_matrix_vectorized(new_size[0], new_size[1], px, py, sigma_f(px, py, new_size, sigma)) for px, py in
                 center_points[::2]]
    matrices2 = [generate_gaussian_matrix_vectorized(new_size[0], new_size[1], px, py, sigma_f(px, py, new_size, sigma)) for px, py in
                 center_points[1::2]]
    matrices = np.array(matrices1) + np.array(matrices2)

    matrices_border = np.zeros((1, new_size[1], new_size[0]))

    for kp in range(1, num_matrices+1):
        if kp in resized_keypoints.keys():
            x1, y1 = resized_keypoints[kp]['x_1'], resized_keypoints[kp]['y_1']
            x2, y2 = resized_keypoints[kp]['x_2'], resized_keypoints[kp]['y_2']

            pixel_dist = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))
            num_gaussians = int(pixel_dist / (sigma))

            if num_gaussians != 1:
                for i in range(num_gaussians):
                    alpha = i / (num_gaussians - 1)
                    x = int(x1 + alpha * (x2 - x1))
                    y = int(y1 + alpha * (y2 - y1))
                    matrices_border[0, :, :] += generate_gaussian_matrix_vectorized(new_size[0], new_size[1], x, y, sigma)
            else:
                x, y = abs(x2 - x1) / 2, abs(y2 - y1) / 2
                matrices_border[0, :, :] += generate_gaussian_matrix_vectorized(new_size[0], new_size[1], x, y, sigma)

    matrices_border = np.clip(matrices_border, 0, 1)

    matrices_combined = np.concatenate((matrices, matrices_border), axis=0)

    return matrices_combined



def generate_gaussian_array_vectorized_dist_l(num_matrices,
                                              keypoints,
                                              lines_aux,
                                              original_size,
                                              img_original_size,
                                              calibration,
                                              down_ratio=2, sigma=2, sigma_mult=1):
    """
    Generate vectorized array with distortion information
    """
    def sigma_f(px, py, size, sigma):
        #multiply sigma if point in image border
        if (px < 5 or px > size[0] - 5) | (py < 5 or py > size[1] - 5):
            return sigma_mult*sigma
        else:
            return sigma


    def project_points(point, calibration, size, orig_size):
        K, R, t, dist = calibration
        img_point, _ = cv2.projectPoints(point, R, t, K, dist)
        img_point = img_point[0][0]
        img_point[0] *= size[0] / orig_size[0]
        img_point[1] *= size[1] / orig_size[1]

        return img_point


    new_size = tuple(int(ti / down_ratio) for ti in original_size)
    resized_keypoints = resize_keypoints_l(keypoints, original_size, new_size)

    # Create an array of center points based on resized keypoints for both points
    center_points = []

    for kp in range(1, num_matrices+1):
        if kp in resized_keypoints.keys():
            center_points.append([resized_keypoints[kp]['x_1'], resized_keypoints[kp]['y_1']])
            center_points.append([resized_keypoints[kp]['x_2'], resized_keypoints[kp]['y_2']])

        else:
            center_points.append([np.inf, np.inf])
            center_points.append([np.inf, np.inf])

    center_points = np.array(center_points)

    # Generate Gaussian matrices for both points and sum them
    matrices1 = [generate_gaussian_matrix_vectorized(new_size[0], new_size[1], px, py, sigma_f(px, py, new_size, sigma)) for px, py in
                 center_points[::2]]
    matrices2 = [generate_gaussian_matrix_vectorized(new_size[0], new_size[1], px, py, sigma_f(px, py, new_size, sigma)) for px, py in
                 center_points[1::2]]
    matrices = np.array(matrices1) + np.array(matrices2)

    matrices_border = np.zeros((1, new_size[1], new_size[0]))

    for line in lines_aux.keys():
        w_kp1, w_kp2 = lines_aux[line]["pt1"], lines_aux[line]["pt2"]
        img_kp1 = project_points(w_kp1, calibration, new_size, img_original_size)
        img_kp2 = project_points(w_kp2, calibration, new_size, img_original_size)

        pixel_dist = np.linalg.norm(img_kp2 -img_kp1)
        num_gaussians = int(pixel_dist / (sigma))

        if num_gaussians != 1:
            t_values = np.linspace(0, 1, num_gaussians)
            interpolated_points = np.array([w_kp1 + t * (w_kp2 - w_kp1) for t in t_values])
            for w_kp in interpolated_points:
                img_kp = project_points(w_kp, calibration, new_size, img_original_size)
                x, y = img_kp[0], img_kp[1]
                matrices_border[0, :, :] += generate_gaussian_matrix_vectorized(new_size[0], new_size[1], x, y, sigma)

        else:
            x, y = abs(img_kp2[0] - img_kp1[0]) / 2, abs(img_kp2[1] - img_kp1[1]) / 2
            matrices_border[0, :, :] += generate_gaussian_matrix_vectorized(new_size[0], new_size[1], x, y, sigma)


    matrices_border = np.clip(matrices_border, 0, 1)

    matrices_combined = np.concatenate((matrices, matrices_border), axis=0)

    return matrices_combined



def _maxpool_heatmap_topk_np(
        heatmap: np.ndarray,
        scale: int,
        max_keypoints: int,
        kernel: int,
        pad_border_value: Optional[float],
        return_scores: bool,
) -> np.ndarray:
    """Numpy port of the original torch maxpool-then-topk decoder.

    Returns an array of shape ``(B, C, K, 3 if return_scores else 2)``
    with last axis ordered ``[u, v, score]`` (note: ``u`` is column,
    ``v`` is row, scaled by ``scale``).

    ``pad_border_value`` mirrors the original two PnLCalib variants:

    * For point keypoints (``..._maxpool``): ``pad_border_value=1.0`` —
      heatmap is padded with the maximum possible value before pooling so
      that border pixels never win the local-max comparison (suppresses
      noisy border peaks).
    * For line extremities (``..._maxpool_l``): ``pad_border_value=None`` —
      regular zero/symmetric pooling with built-in padding, no border
      suppression.
    """
    batch_size, n_channels, _height, width = heatmap.shape

    if pad_border_value is not None:
        pad = (kernel - 1) // 2
        padded = np.pad(
            heatmap,
            ((0, 0), (0, 0), (pad, pad), (pad, pad)),
            mode="constant",
            constant_values=pad_border_value,
        )
        # maxpool stride=1, no padding -> equivalent to maximum_filter on padded
        # restricted to the central HxW region.
        pooled_padded = maximum_filter(
            padded, size=(1, 1, kernel, kernel), mode="constant", cval=-np.inf
        )
        max_pooled = pooled_padded[:, :, pad:pad + _height, pad:pad + width]
    else:
        # equivalent to max_pool2d with same padding
        max_pooled = maximum_filter(
            heatmap, size=(1, 1, kernel, kernel), mode="constant", cval=-np.inf
        )

    local_maxima = max_pooled == heatmap
    suppressed = heatmap * local_maxima  # zero out non-local-maxima

    flat = suppressed.reshape(batch_size, n_channels, -1)

    if max_keypoints == 1:
        idx = np.argmax(flat, axis=-1)
        scores = np.take_along_axis(flat, idx[..., None], axis=-1).squeeze(-1)
        idx = idx[..., None]      # (B, C, 1)
        scores = scores[..., None]
    else:
        # top-k descending. argpartition is O(N); then sort just the K winners.
        part = np.argpartition(flat, -max_keypoints, axis=-1)[..., -max_keypoints:]
        part_scores = np.take_along_axis(flat, part, axis=-1)
        order = np.argsort(-part_scores, axis=-1)
        idx = np.take_along_axis(part, order, axis=-1)        # (B, C, K)
        scores = np.take_along_axis(part_scores, order, axis=-1)

    rows = idx // width
    cols = idx % width

    last_dim = 3 if return_scores else 2
    out = np.empty((batch_size, n_channels, max_keypoints, last_dim), dtype=np.float32)
    out[..., 0] = cols * scale  # u (x)
    out[..., 1] = rows * scale  # v (y)
    if return_scores:
        out[..., 2] = scores
    return out


def get_keypoints_from_heatmap_batch_maxpool(
        heatmap: np.ndarray,
        scale: int = 2,
        max_keypoints: int = 1,
        min_keypoint_pixel_distance: int = 1,
        return_scores: bool = True,
) -> np.ndarray:
    """Fast extraction of keypoints from a batch of heatmaps using maxpooling.

    Numpy port of the original torch implementation. Suppresses border
    keypoints by padding the heatmap with the maximum possible value
    before max-pooling.

    Args:
        heatmap: (B, C, H, W) numpy array
        max_keypoints: number of keypoints to keep per channel
        min_keypoint_pixel_distance: half-kernel for the local-max test

    Returns:
        ndarray of shape (B, C, max_keypoints, 3) — last axis is (u, v, score)
        with u/v scaled by ``scale``.
    """
    kernel = min_keypoint_pixel_distance * 2 + 1
    return _maxpool_heatmap_topk_np(
        heatmap,
        scale=scale,
        max_keypoints=max_keypoints,
        kernel=kernel,
        pad_border_value=1.0,
        return_scores=return_scores,
    )


def get_keypoints_from_heatmap_batch_maxpool_l(
        heatmap: np.ndarray,
        scale: int = 2,
        max_keypoints: int = 2,
        min_keypoint_pixel_distance: int = 1,
        return_scores: bool = True,
) -> np.ndarray:
    """Numpy port: same as the kp variant but without border suppression
    (lines often touch the image boundary, so border keypoints are valid).
    """
    kernel = min_keypoint_pixel_distance * 2 + 1
    return _maxpool_heatmap_topk_np(
        heatmap,
        scale=scale,
        max_keypoints=max_keypoints,
        kernel=kernel,
        pad_border_value=None,
        return_scores=return_scores,
    )


def coords_to_dict(coords: np.ndarray, threshold: float = 0.05, ground_plane_only: bool = False):
    """Numpy version of the original torch decoder.

    ``coords`` is the (B, C, K, 3) array returned by the maxpool helpers;
    last axis is (u, v, score). Returns a list of per-batch dicts
    matching the original PnLCalib structure.
    """
    batch_size, n_channels, n_keypoints, _ = coords.shape
    kp_list = []
    for batch in range(batch_size):
        keypoints = {}
        for c in range(n_channels):
            label = c + 1
            if n_keypoints == 1:
                if ground_plane_only and label in (12, 15, 16, 19):
                    continue
                if coords[batch, c, 0, -1] > threshold:
                    keypoints[label] = {
                        'x': float(coords[batch, c, 0, 0]),
                        'y': float(coords[batch, c, 0, 1]),
                        'p': float(coords[batch, c, 0, 2]),
                    }
            else:
                if ground_plane_only and label in (7, 8, 9, 10, 11, 12):
                    continue
                if (
                    coords[batch, c, 0, -1] > threshold
                    and coords[batch, c, 1, -1] > threshold
                ):
                    keypoints[label] = {
                        'x_1': float(coords[batch, c, 0, 0]),
                        'y_1': float(coords[batch, c, 0, 1]),
                        'p_1': float(coords[batch, c, 0, 2]),
                        'x_2': float(coords[batch, c, 1, 0]),
                        'y_2': float(coords[batch, c, 1, 1]),
                        'p_2': float(coords[batch, c, 1, 2]),
                    }
        kp_list.append(keypoints)
    return kp_list


def complete_keypoints(kp_dict, lines_dict, w, h, normalize=False):

    def line_intersection(x1, y1, x2, y2):
        #1e-7 sum in case there are two identical coordinate values
        x1[-1] += 1e-7
        x2[-1] += 1e-7
        slope1, intercept1, r1, p1, se1 = linregress(x1, y1)
        slope2, intercept2, r2, p2, se2 = linregress(x2, y2)

        x_intersection = (intercept2 - intercept1) / (slope1 - slope2 + 1e-7)
        y_intersection = slope1 * x_intersection + intercept1

        return x_intersection, y_intersection

    lines_list = ["Big rect. left bottom", "Big rect. left main", "Big rect. left top", "Big rect. right bottom",
                  "Big rect. right main", "Big rect. right top", "Goal left crossbar", "Goal left post left ",
                  "Goal left post right", "Goal right crossbar", "Goal right post left", "Goal right post right",
                  "Middle line", "Side line bottom", "Side line left", "Side line right", "Side line top",
                  "Small rect. left bottom", "Small rect. left main", "Small rect. left top", "Small rect. right bottom",
                  "Small rect. right main", "Small rect. right top"]


    keypoints_line_list = [['Side line top', 'Side line left'], ['Side line top', 'Middle line'],
                           ['Side line right', 'Side line top'], ['Side line left', 'Big rect. left top'],
                           ['Big rect. left top', 'Big rect. left main'], ['Big rect. right top', 'Big rect. right main'],
                           ['Side line right', 'Big rect. right top'], ['Side line left', 'Small rect. left top'],
                           ['Small rect. left top', 'Small rect. left main'], ['Small rect. right top', 'Small rect. right main'],
                           ['Side line right', 'Small rect. right top'], ['Goal left crossbar', 'Goal left post right'],
                           ['Side line left', 'Goal left post right'], ['Side line right', 'Goal right post left'],
                           ['Goal right crossbar', 'Goal right post left'], ['Goal left crossbar', 'Goal left post left '],
                           ['Side line left', 'Goal left post left '], ['Side line right', 'Goal right post right'],
                           ['Goal right crossbar', 'Goal right post right'], ['Side line left', 'Small rect. left bottom'],
                           ['Small rect. left bottom', 'Small rect. left main'], ['Small rect. right bottom', 'Small rect. right main'],
                           ['Side line right', 'Small rect. right bottom'], ['Side line left', 'Big rect. left bottom'],
                           ['Big rect. left bottom', 'Big rect. left main'], ['Big rect. right main', 'Big rect. right bottom'],
                           ['Side line right', 'Big rect. right bottom'], ['Side line left', 'Side line bottom'],
                           ['Side line bottom', 'Middle line'], ['Side line bottom', 'Side line right']]


    keypoint_aux_pair_list = [['Small rect. left main', 'Side line top'], ['Big rect. left main', 'Side line top'],
                              ['Big rect. right main', 'Side line top'], ['Small rect. right main', 'Side line top'],
                              ['Small rect. left main', 'Big rect. left top'], ['Big rect. right top', 'Small rect. right main'],
                              ['Small rect. left top', 'Big rect. left main'], ['Small rect. right top', 'Big rect. right main'],
                              ['Small rect. left bottom', 'Big rect. left main'], ['Small rect. right bottom', 'Big rect. right main'],
                              ['Small rect. left main', 'Big rect. left bottom'], ['Small rect. right main', 'Big rect. right bottom'],
                              ['Small rect. left main', 'Side line bottom'], ['Big rect. left main', 'Side line bottom'],
                              ['Big rect. right main', 'Side line bottom'], ['Small rect. right main', 'Side line bottom']]

    w_extra = 0. * w
    h_extra = 0. * h

    complete_dict = copy.deepcopy(kp_dict)
    for key in range(1, 31):
        if key not in kp_dict.keys():
            line_keys = keypoints_line_list[key-1]
            line_key1, line_key2 = lines_list.index(line_keys[0]) + 1, lines_list.index(line_keys[1]) + 1
            if all(line_key in lines_dict.keys() for line_key in [line_key1, line_key2]):
                x1 = [lines_dict[line_key1]['x_1'], lines_dict[line_key1]['x_2']]
                y1 = [lines_dict[line_key1]['y_1'], lines_dict[line_key1]['y_2']]
                x2 = [lines_dict[line_key2]['x_1'], lines_dict[line_key2]['x_2']]
                y2 = [lines_dict[line_key2]['y_1'], lines_dict[line_key2]['y_2']]
                new_kp = line_intersection(x1, y1, x2, y2)
                if -w_extra < new_kp[0] < w_extra + w and -h_extra < new_kp[1] < h_extra + h:
                    complete_dict[key] = {'x': round(new_kp[0], 0), 'y': round(new_kp[1], 0), 'p': 1.}

    for key in range(1, len(keypoint_aux_pair_list)):
        line_keys = keypoint_aux_pair_list[key-1]
        line_key1, line_key2 = lines_list.index(line_keys[0]) + 1, lines_list.index(line_keys[1]) + 1
        if all(line_key in lines_dict.keys() for line_key in [line_key1, line_key2]):
            x1 = [lines_dict[line_key1]['x_1'], lines_dict[line_key1]['x_2']]
            y1 = [lines_dict[line_key1]['y_1'], lines_dict[line_key1]['y_2']]
            x2 = [lines_dict[line_key2]['x_1'], lines_dict[line_key2]['x_2']]
            y2 = [lines_dict[line_key2]['y_1'], lines_dict[line_key2]['y_2']]
            new_kp = line_intersection(x1, y1, x2, y2)
            if -w_extra < new_kp[0] < w_extra + w and -h_extra < new_kp[1] < h_extra + h:
                complete_dict[key+57] = {'x': round(new_kp[0], 0), 'y': round(new_kp[1], 0), 'p': 1.}

    if normalize:
        for kp in complete_dict.keys():
            complete_dict[kp]['x'] /= w
            complete_dict[kp]['y'] /= h

        for line in lines_dict.keys():
            lines_dict[line]['x_1'] /= w
            lines_dict[line]['y_1'] /= h
            lines_dict[line]['x_2'] /= w
            lines_dict[line]['y_2'] /= h

    complete_dict = dict(sorted(complete_dict.items()))


    return complete_dict, lines_dict






