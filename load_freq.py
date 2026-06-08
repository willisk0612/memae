"""Usage: python load_freq.py path/to/capture.npz [--out path/to/output.html]"""

import argparse
from pathlib import Path

import holoviews as hv
import numpy as np
import panel as pn

from utils import (
    FREQ_COLORS,
    load_capture_data,
)
from load_plot import (
    make_frequency_plot,
    show_plots,
    spectrum_xy,
)

hv.extension("bokeh")
pn.extension()


def build_frequency_plots(
    plc: np.ndarray,
    plc_t: np.ndarray,
    sk: np.ndarray,
    sk_t: np.ndarray,
    us: np.ndarray,
    us_t: np.ndarray,
):
    plots = []

    if sk.size > 0 and sk.shape[1] >= 4 and sk_t.size:
        plots.extend(
            [
                make_frequency_plot(
                    "Vibration X Spectrum",
                    "Amplitude (g peak)",
                    FREQ_COLORS[0],
                    spectrum_xy(sk_t, sk[:, 1]),
                ),
                make_frequency_plot(
                    "Vibration Y Spectrum",
                    "Amplitude (g peak)",
                    FREQ_COLORS[1],
                    spectrum_xy(sk_t, sk[:, 2]),
                ),
                make_frequency_plot(
                    "Vibration Z Spectrum",
                    "Amplitude (g peak)",
                    FREQ_COLORS[2],
                    spectrum_xy(sk_t, sk[:, 3]),
                ),
            ]
        )

    if plc.size > 0 and plc.shape[1] >= 2 and plc_t.size:
        if plc.shape[1] == 2:
            plots.append(
                make_frequency_plot(
                    "Actual Torque Spectrum",
                    "Amplitude",
                    FREQ_COLORS[0],
                    spectrum_xy(plc_t, plc[:, 1]),
                )
            )
        elif plc.shape[1] >= 5:
            plots.extend(
                [
                    make_frequency_plot(
                        "Speed Deviation Spectrum",
                        "Amplitude",
                        FREQ_COLORS[0],
                        spectrum_xy(plc_t, plc[:, 2]),
                    ),
                    make_frequency_plot(
                        "Motor Torque Spectrum",
                        "Amplitude",
                        FREQ_COLORS[1],
                        spectrum_xy(plc_t, plc[:, 3]),
                    ),
                    make_frequency_plot(
                        "Torque Spectrum",
                        "Amplitude",
                        FREQ_COLORS[2],
                        spectrum_xy(plc_t, plc[:, 4]),
                    ),
                ]
            )
        else:
            plots.extend(
                [
                    make_frequency_plot(
                        "Motor Temp Spectrum",
                        "Amplitude",
                        FREQ_COLORS[0],
                        spectrum_xy(plc_t, plc[:, 1]),
                    ),
                    make_frequency_plot(
                        "Torque Spectrum",
                        "Amplitude",
                        FREQ_COLORS[1],
                        spectrum_xy(plc_t, plc[:, 2]),
                    ),
                ]
            )

    if us.size > 0 and us.shape[1] >= 2 and us_t.size:
        plots.append(
            make_frequency_plot(
                "Ultrasound Spectrum",
                "Amplitude",
                FREQ_COLORS[1],
                spectrum_xy(us_t, us[:, 1]),
            )
        )

    return plots


def main():
    parser = argparse.ArgumentParser(
        description="Load a capture .npz and view frequency-domain plots with HoloViews."
    )
    parser.add_argument("path", help="Path to .npz created by acquisition.py")
    parser.add_argument("--out", default="", help="Optional output HTML path.")
    args = parser.parse_args()

    path = Path(args.path)
    capture = load_capture_data(path)

    plots = build_frequency_plots(
        np.asarray(capture["plc"], dtype=np.float64),
        np.asarray(capture["plc_t"], dtype=np.float64),
        np.asarray(capture["sk"], dtype=np.float64),
        np.asarray(capture["sk_t"], dtype=np.float64),
        np.asarray(capture["us"], dtype=np.float64),
        np.asarray(capture["us_t"], dtype=np.float64),
    )

    show_plots(path, plots, args.out)


if __name__ == "__main__":
    main()
