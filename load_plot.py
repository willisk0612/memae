"""Plot loading script and interactive capture plotting helpers."""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils import EMPTY_TS, PLOT_HEIGHT, PLOT_WIDTH, TOOLS


SCROLL_ZOOM_BASE_SCALE = 1.2


def series_xy(
    time_arr: np.ndarray, value_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(time_arr) & np.isfinite(value_arr)
    return (
        np.asarray(time_arr[valid], dtype=np.float64),
        np.asarray(value_arr[valid], dtype=np.float64),
    )


def estimate_sample_rate(time_arr: np.ndarray) -> float | None:
    if time_arr.size < 2:
        return None
    dt = np.diff(time_arr)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return None
    median_dt = float(np.median(dt))
    if median_dt <= 0.0:
        return None
    return 1.0 / median_dt


def spectrum_xy(
    time_arr: np.ndarray, value_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(time_arr) & np.isfinite(value_arr)
    t = np.asarray(time_arr[valid], dtype=np.float64)
    y = np.asarray(value_arr[valid], dtype=np.float64)
    if t.size < 2 or y.size < 2:
        return EMPTY_TS, EMPTY_TS

    sample_rate = estimate_sample_rate(t)
    if sample_rate is None:
        return EMPTY_TS, EMPTY_TS

    y = y - np.mean(y)
    if y.size < 2 or not np.any(np.abs(y) > 0):
        return EMPTY_TS, EMPTY_TS

    window = np.hanning(y.size)
    windowed = y * window
    freqs = np.fft.rfftfreq(windowed.size, d=1.0 / sample_rate)
    coherent_gain = np.sum(window)
    if coherent_gain <= 0.0:
        return EMPTY_TS, EMPTY_TS
    amps = 2.0 * np.abs(np.fft.rfft(windowed)) / coherent_gain
    amps[0] *= 0.5
    if y.size % 2 == 0:
        amps[-1] *= 0.5
    if freqs.size <= 1:
        return EMPTY_TS, EMPTY_TS
    return freqs[1:], amps[1:]


def make_time_plot(
    title: str, ylabel: str, label: str, color: str, xy: tuple[np.ndarray, np.ndarray]
):
    from holoviews import Curve
    from holoviews.operation.datashader import datashade, dynspread

    x_arr, y_arr = xy
    if x_arr.size == 0:
        return None

    curve = Curve((x_arr, y_arr), kdims="Time (s)", vdims=ylabel, label=label)
    shaded = dynspread(datashade(curve, cmap=[color]))
    return shaded.opts(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        xlabel="Time (s)",
        ylabel=ylabel,
        title=title,
        default_tools=TOOLS,
        tools=[],
        active_tools=["wheel_zoom"],
        axiswise=True,
        framewise=True,
        show_grid=True,
        bgcolor="white",
    )


def make_frequency_plot(
    title: str, ylabel: str, color: str, xy: tuple[np.ndarray, np.ndarray]
):
    from holoviews import Curve
    from holoviews.operation.datashader import datashade

    x_arr, y_arr = xy
    if x_arr.size == 0:
        return None

    curve = Curve((x_arr, y_arr), kdims="Frequency (Hz)", vdims=ylabel)
    shaded = datashade(curve, cmap=[color])
    return shaded.opts(
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        xlabel="Frequency (Hz)",
        ylabel=ylabel,
        title=title,
        default_tools=TOOLS,
        tools=[],
        active_tools=["wheel_zoom"],
        axiswise=True,
        framewise=True,
        show_grid=True,
        bgcolor="white",
    )


def build_layout(path: Path, plots: list):
    from holoviews import Layout

    valid_plots = [plot for plot in plots if plot is not None]
    if not valid_plots:
        raise RuntimeError("No plottable data found in file.")
    layout = Layout(valid_plots).cols(1).relabel(path.name)
    styled_layout = layout.opts(shared_axes=False)
    return layout if styled_layout is None else styled_layout


def build_panel_column(path: Path, plots: list):
    import panel as pn

    valid_plots = [plot for plot in plots if plot is not None]
    if not valid_plots:
        raise RuntimeError("No plottable data found in file.")

    return pn.Column(
        pn.pane.Markdown(f"## {path.name}"),
        *[pn.panel(plot) for plot in valid_plots],
    )


def show_plots(path: Path, plots: list, out_path: str = "") -> None:
    """Save plots to HTML when requested, otherwise open the interactive panel."""
    if out_path:
        import holoviews as hv

        out = Path(out_path)
        if out.suffix.lower() != ".html":
            out = out.with_suffix(".html")
        hv.save(build_layout(path, plots), out, backend="bokeh")
        print(f"Saved HTML plot to {out}")
        return

    build_panel_column(path, plots).show()


def _scaled_limits(limits: tuple[float, float], center: float, scale: float):
    lower, upper = limits
    return center - (center - lower) * scale, center + (upper - center) * scale


def enable_matplotlib_scroll_zoom(fig) -> None:
    """Enable mouse-wheel zoom for loaded Matplotlib pickle figures."""

    def on_scroll(event):
        ax = event.inaxes
        if ax is None or not ax.get_navigate():
            return

        scale = (
            1.0 / SCROLL_ZOOM_BASE_SCALE
            if event.step > 0
            else SCROLL_ZOOM_BASE_SCALE
        )
        xdata = event.xdata
        ydata = event.ydata
        if xdata is None or ydata is None:
            return

        ax.set_xlim(*_scaled_limits(ax.get_xlim(), float(xdata), scale))
        ax.set_ylim(*_scaled_limits(ax.get_ylim(), float(ydata), scale))
        if hasattr(ax, "get_zlim") and hasattr(ax, "set_zlim"):
            z_lower, z_upper = ax.get_zlim()
            z_center = (z_lower + z_upper) / 2.0
            ax.set_zlim(*_scaled_limits((z_lower, z_upper), z_center, scale))
        fig.canvas.draw_idle()

    fig._scroll_zoom_handler = on_scroll
    fig._scroll_zoom_cid = fig.canvas.mpl_connect("scroll_event", on_scroll)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python load_plot.py <path_to_pkl_file>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        fig = pickle.load(f)

    if hasattr(fig, "canvas"):
        enable_matplotlib_scroll_zoom(fig)

    plt.show()


if __name__ == "__main__":
    main()
