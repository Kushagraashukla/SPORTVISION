import logging
import os
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LOGGER = logging.getLogger(__name__)


def generate_heatmap(player_id, positions_list, field_width, field_height):
    """
    Generate a passive player-position heatmap.

    This function does not depend on detection, tracking, speed, or team logic.
    It returns None when the available position history is not reliable enough.
    """
    try:
        width = int(field_width)
        height = int(field_height)
        if width <= 0 or height <= 0:
            LOGGER.error("Heatmap skipped for player %s: invalid field size.", player_id)
            return None

        valid_positions = _filter_positions(positions_list, width, height)
        if len(valid_positions) < 20:
            LOGGER.error(
                "Heatmap skipped for player %s: fewer than 20 valid positions.",
                player_id,
            )
            return None

        density = np.zeros((height, width), dtype=np.float32)
        for x_pos, y_pos in valid_positions:
            x_idx = min(max(int(round(x_pos)), 0), width - 1)
            y_idx = min(max(int(round(y_pos)), 0), height - 1)
            density[y_idx, x_idx] += 1.0

        density = _gaussian_blur(density)
        max_density = float(np.max(density))
        if max_density <= 0:
            LOGGER.error("Heatmap skipped for player %s: empty density map.", player_id)
            return None
        density = density / max_density

        os.makedirs("heatmaps", exist_ok=True)
        output_path = os.path.join("heatmaps", f"player_{player_id}.png")
        _save_field_heatmap(output_path, density, width, height, player_id)
        return output_path

    except Exception as exc:
        LOGGER.error("Heatmap generation failed for player %s: %s", player_id, exc)
        return None


def _filter_positions(positions_list, field_width, field_height):
    valid_positions = []
    previous_valid = None

    try:
        iterable_positions = positions_list or []
    except Exception:
        return valid_positions

    for position in iterable_positions:
        try:
            if position is None or len(position) < 2:
                continue

            x_pos = float(position[0])
            y_pos = float(position[1])

            if x_pos < 0 or y_pos < 0:
                continue
            if x_pos >= field_width or y_pos >= field_height:
                continue

            current = (x_pos, y_pos)
            if previous_valid is not None and _distance(previous_valid, current) > 100.0:
                continue

            valid_positions.append(current)
            previous_valid = current
        except Exception as exc:
            LOGGER.error("Invalid heatmap position ignored: %s", exc)
            continue

    return valid_positions


def _gaussian_blur(density):
    try:
        try:
            from scipy.ndimage import gaussian_filter

            return gaussian_filter(density, sigma=18)
        except Exception:
            return _numpy_gaussian_blur(density, sigma=18)
    except Exception as exc:
        LOGGER.error("Gaussian blur failed: %s", exc)
        return density


def _numpy_gaussian_blur(density, sigma=18):
    radius = max(1, int(sigma * 3))
    axis = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(axis ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    blurred = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, density)
    blurred = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, blurred)
    return blurred


def _save_field_heatmap(output_path, density, field_width, field_height, player_id):
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_facecolor("#167a34")
        fig.patch.set_facecolor("#167a34")

        _draw_field(ax, field_width, field_height)
        ax.imshow(
            density,
            cmap="hot",
            alpha=0.62,
            extent=[0, field_width, field_height, 0],
            interpolation="bilinear",
        )

        ax.set_xlim(0, field_width)
        ax.set_ylim(field_height, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Player {player_id} Heatmap", color="white", fontsize=15, pad=10)

        fig.savefig(output_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
    except Exception:
        plt.close("all")
        raise


def _draw_field(ax, field_width, field_height):
    line_color = "white"
    line_width = 2

    ax.plot([0, field_width, field_width, 0, 0], [0, 0, field_height, field_height, 0], line_color, linewidth=line_width)
    ax.plot([field_width / 2, field_width / 2], [0, field_height], line_color, linewidth=line_width)

    center_x = field_width / 2
    center_y = field_height / 2
    center_radius = min(field_width, field_height) * 0.09
    center_circle = plt.Circle((center_x, center_y), center_radius, fill=False, color=line_color, linewidth=line_width)
    ax.add_patch(center_circle)
    ax.scatter([center_x], [center_y], color=line_color, s=18)

    box_width = field_width * 0.16
    box_height = field_height * 0.42
    six_width = field_width * 0.055
    six_height = field_height * 0.22

    _draw_box(ax, 0, center_y - box_height / 2, box_width, box_height, line_color, line_width)
    _draw_box(ax, field_width - box_width, center_y - box_height / 2, box_width, box_height, line_color, line_width)
    _draw_box(ax, 0, center_y - six_height / 2, six_width, six_height, line_color, line_width)
    _draw_box(ax, field_width - six_width, center_y - six_height / 2, six_width, six_height, line_color, line_width)


def _draw_box(ax, x_pos, y_pos, width, height, color, line_width):
    ax.plot(
        [x_pos, x_pos + width, x_pos + width, x_pos, x_pos],
        [y_pos, y_pos, y_pos + height, y_pos + height, y_pos],
        color,
        linewidth=line_width,
    )


def _distance(first, second):
    return sqrt((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2)
