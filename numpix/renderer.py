import base64
import os
import re
import select
import sys
import termios
import tty
from functools import lru_cache
from typing import Literal

import numpy as np

from numpix import kitty_protocol_enabled
from numpix.colormaps import GREY_PIXEL, get_colormap, get_pixel

write = sys.stdout.write
GREY_SENTINAL = np.nan


@lru_cache(maxsize=1)
def _get_cell_size():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        os.write(sys.stdout.fileno(), b"\033[16t")
        response = b""
        for _ in range(100):
            if select.select([fd], [], [], 0.05)[0]:
                response += os.read(fd, 1024)
                if b"t" in response:
                    break
            elif response:
                break
        match = re.search(rb"\033\[6;(\d+);(\d+)t", response)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 16, 8
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _build_rgb(
    normed_display: np.ndarray,
    color_scheme: Literal[
        "magma", "hot", "grey", "inferno", "plasma", "cividis", "coolwarm"
    ] = "cividis",
):
    cmap = get_colormap(color_scheme)
    h, w = normed_display.shape
    rgb = np.full((h, w, 3), GREY_PIXEL, dtype=np.uint8)
    mask = ~np.isnan(normed_display)
    indices = np.clip((normed_display[mask] * 255), 0, 255).astype(int)
    rgb[mask] = cmap[indices]
    return rgb


def _send_kitty(rgb):
    h, w = rgb.shape[:2]
    raw = rgb.tobytes()
    b64 = base64.b64encode(raw).decode("ascii")
    chunk_size = 4096
    if len(b64) <= chunk_size:
        sys.stdout.write(f"\033_Gf=24,s={w},v={h},a=T,t=d;{b64}\033\\")
    else:
        first = b64[:chunk_size]
        sys.stdout.write(f"\033_Gf=24,s={w},v={h},a=T,t=d,m=1;{first}\033\\")
        b64 = b64[chunk_size:]
        while len(b64) > chunk_size:
            chunk = b64[:chunk_size]
            sys.stdout.write(f"\033_Gm=1;{chunk}\033\\")
            b64 = b64[chunk_size:]
        sys.stdout.write(f"\033_Gm=0;{b64}\033\\")
    sys.stdout.write("\n")
    sys.stdout.flush()


def _render_cell(
    d,
    y,
    x,
    color_scheme: Literal[
        "magma", "hot", "grey", "inferno", "plasma", "cividis", "coolwarm"
    ] = "cividis",
):
    top = d[y, x]
    if np.isnan(top):
        top_pixel = GREY_PIXEL
    else:
        top_pixel = get_pixel(top, color_scheme=color_scheme)
    if y + 1 < d.shape[0]:
        bottom = d[y + 1, x]
        if np.isnan(bottom):
            btm_pixel = GREY_PIXEL
        else:
            btm_pixel = get_pixel(bottom, color_scheme=color_scheme)
        write(
            f"\033[48;2;{top_pixel[0]};{top_pixel[1]};{top_pixel[2]};38;2;{btm_pixel[0]};{btm_pixel[1]};{btm_pixel[2]}m▄"
        )
    else:
        write(f"\033[38;2;{top_pixel[0]};{top_pixel[1]};{top_pixel[2]}m▀")


def _render_break():
    write("\033[0m\n")
    sys.stdout.flush()


def _truncate_2d(normed, max_show):
    half = max_show // 2
    n_rows, n_cols = normed.shape
    if n_rows > max_show:
        grey_row = np.full((1, n_cols), GREY_SENTINAL)
        normed = np.concatenate([normed[:half], grey_row, normed[-half:]], axis=0)
    n_rows, n_cols = normed.shape
    if n_cols > max_show:
        grey_col = np.full((n_rows, 1), GREY_SENTINAL)
        normed = np.concatenate([normed[:, :half], grey_col, normed[:, -half:]], axis=1)
    return normed


def pix(
    array,
    max_show: int = 40,
    color_scheme: Literal[
        "magma", "hot", "grey", "inferno", "plasma", "cividis", "coolwarm"
    ] = "cividis",
    max_slices: int = 3,
    layout: Literal["horizontal", "vertical"] = "horizontal",
    use_kitty_protocol: bool = True,
):
    assert array.ndim <= 3, (
        "Only arrays up to 3 dims are supported. Pass a 2D or 3D slice, e.g. numpix.show(arr[0])"
    )
    original_dtype = array.dtype
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    if array.dtype == bool:
        array = array.astype(np.float32)

    if use_kitty_protocol:
        use_kitty_protocol = kitty_protocol_enabled

    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 2:
        array = array[np.newaxis]

    lo, hi = array.min(), array.max()
    original_shape = array.shape

    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim == 2:
        array = array[np.newaxis]

    lo, hi = array.min(), array.max()
    mean = array.mean()
    var = array.var()

    original_shape = array.shape

    write(f"\033[1mshape\033[0m={original_shape}  ")
    write(f"\033[1mmin\033[0m={lo:.4f}  ")
    write(f"\033[1mmax\033[0m={hi:.4f}  ")
    write(f"\033[1mmean\033[0m={mean:.4f}  ")
    write(f"\033[1mvar\033[0m={var:.4f} ")
    write(f"\033[1mdtype\033[0m={original_dtype}\n")

    if use_kitty_protocol:
        cell_h, cell_w = _get_cell_size()
        target_cols = 40
        target_rows = 20

        displays = []
        for b in array:
            normed = ((b - lo) + 1e-8) / ((hi - lo) + 1e-8)
            h, w = normed.shape
            px_per_val_x = max(1, (target_cols * cell_w) // w)
            px_per_val_y = max(1, (target_rows * cell_h) // h)
            px = min(px_per_val_x, px_per_val_y)
            upscaled = np.repeat(np.repeat(normed, px, axis=0), px, axis=1)
            displays.append(upscaled)

        half_slices = max_slices // 2
        if len(displays) > max_slices:
            grey = np.full((displays[0].shape[0], px * 2), GREY_SENTINAL)
            displays = displays[:half_slices] + [grey] + displays[-half_slices:]

        rgbs = [_build_rgb(d, color_scheme) for d in displays]

        if layout == "horizontal":
            gap = np.full((rgbs[0].shape[0], px, 3), GREY_PIXEL, dtype=np.uint8)
            parts = []
            for i, img in enumerate(rgbs):
                if i > 0:
                    parts.append(gap)
                parts.append(img)
            combined = np.concatenate(parts, axis=1)
        else:
            gap = np.full((px, rgbs[0].shape[1], 3), GREY_PIXEL, dtype=np.uint8)
            parts = []
            for i, img in enumerate(rgbs):
                if i > 0:
                    parts.append(gap)
                parts.append(img)
            combined = np.concatenate(parts, axis=0)

        _send_kitty(combined)
    else:
        displays = []
        for b in array:
            normed = ((b - lo) + 1e-8) / ((hi - lo) + 1e-8)
            displays.append(_truncate_2d(normed, max_show))

        half_slices = max_slices // 2
        if len(displays) > max_slices:
            if layout == "horizontal":
                grey_separator = np.full((displays[0].shape[0], 3), GREY_SENTINAL)
            else:
                grey_separator = np.full((1, displays[0].shape[1]), GREY_SENTINAL)
            displays = (
                displays[:half_slices] + [grey_separator] + displays[-half_slices:]
            )

        if layout == "vertical":
            for i, d in enumerate(displays):
                for y in range(0, d.shape[0], 2):
                    for x in range(d.shape[1]):
                        _render_cell(d, y, x, color_scheme)
                    _render_break()
                if i < len(displays) - 1:
                    _render_break()
        else:
            disp_rows = max(d.shape[0] for d in displays)
            for y in range(0, disp_rows, 2):
                for i, d in enumerate(displays):
                    if i > 0:
                        write("\033[0m  ")
                    for x in range(d.shape[1]):
                        if y >= d.shape[0]:
                            write(" ")
                            continue
                        _render_cell(d, y, x, color_scheme)
                _render_break()
