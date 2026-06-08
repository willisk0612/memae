"""Read ST SensorKit streams as chunks of (sensor_timestamp, samples).

Decodes raw USB packets into calibrated NumPy arrays while preserving the
device frame timestamp when present.

Can be run as a standalone script to verify sensorkit connectivity.
"""


from stdatalog_core.HSD_link.HSDLink import HSDLink
from stdatalog_core.HSD.utils.type_conversion import TypeConversion

import contextlib
import io
import numpy as np
import struct
import time
from typing import Optional, Any, TypedDict, cast

from utils import max_numeric_value


class SensorMeta(TypedDict):
    """Calibrated metadata for a SensorKit sensor, including enabled state, ODR in Hz, and full-scale range in g."""
    enabled: bool
    odr_hz: Optional[float]
    fs_g: Optional[float]
    samples_per_ts: int


class SensorKitSession:
    """Context manager for managing HSDLink session and device connection."""
    def __init__(self, *, device_id: int = 0, acquisition_folder: str = ".") -> None:
        self.device_id = device_id
        self.acquisition_folder = acquisition_folder
        self.hsd_link = HSDLink()
        self.hsd_link_instance = None
        self._log_started = False

    def __enter__(self) -> "SensorKitSession":
        """Initialize HSDLink instance and connect to the specified device, suppressing verbose output."""
        with contextlib.redirect_stdout(io.StringIO()):
            self.hsd_link_instance = self.hsd_link.create_hsd_link(
                acquisition_folder=self.acquisition_folder
            )
        if self.hsd_link_instance is None:
            raise RuntimeError("Failed to create HSDLink instance.")
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        """Stop logging if active and clean up HSDLink instance, suppressing verbose output."""
        if self._log_started and self.hsd_link_instance is not None:
            with contextlib.redirect_stdout(io.StringIO()):
                self.hsd_link.stop_log(self.hsd_link_instance, self.device_id)
        self._log_started = False

    def get_sensor_status(self, sensor_name: str) -> dict[str, Any]:
        """Retrieve the status dictionary for a given sensor, extracting the relevant sub-dictionary if needed."""
        comp_status = self.hsd_link.get_component_status(
            self.hsd_link_instance, self.device_id, sensor_name
        )
        if comp_status is None:
            raise RuntimeError(f"Failed to read status for '{sensor_name}'.")
        if isinstance(comp_status, dict):
            return comp_status.get(sensor_name, comp_status)
        return {}

    def set_sensor_enable(self, sensor_name: str, enabled: bool) -> None:
        """Set the enable status for a given sensor."""
        self.hsd_link.set_sensor_enable(
            self.hsd_link_instance, self.device_id, enabled, sensor_name=sensor_name
        )

    def set_sensor_odr(self, sensor_name: str, odr: float) -> None:
        """Set the output data rate (ODR) for a given sensor."""
        self.hsd_link.set_sensor_odr(
            self.hsd_link_instance, self.device_id, odr, sensor_name=sensor_name
        )

    def start_log(self) -> None:
        """Start logging data from the device, suppressing verbose output."""
        if self._log_started:
            return
        with contextlib.redirect_stdout(io.StringIO()):
            self.hsd_link.start_log(
                self.hsd_link_instance,
                self.device_id,
                sub_folder=False,
                save_files=False,
            )
        self._log_started = True


class SensorStream:
    """Chunked data stream reader for a specific SensorKit sensor, decoding binary data packets into calibrated NumPy arrays."""
    @staticmethod
    def _val(x, default=None):
        """Extract 'val' from a dict or return the value itself, with a default fallback."""
        if isinstance(x, dict) and "val" in x:
            return x.get("val", default)
        return x if x is not None else default

    @staticmethod
    def _coerce_float(x: object) -> Optional[float]:
        """Coerce a value to float if it's a numeric type, otherwise return None."""
        if isinstance(x, bool) or x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        return None

    def __init__(
        self,
        session: SensorKitSession,
        sensor_name: str,
        *,
        timeout_s: Optional[float] = None,
        axes: Optional[int] = 3,
    ) -> None:
        self.session = session
        self.sensor_name = sensor_name
        self.timeout_s = timeout_s
        self.axes = axes
        self.status = self.session.get_sensor_status(sensor_name)

        self.dim = int(self.status.get("dim", 1))
        if self.axes is not None and self.dim < self.axes:
            raise ValueError(
                f"Expected dim>={self.axes} for '{sensor_name}', got {self.dim}."
            )

        self.data_type = self.status.get("data_type", "int16")
        self.sensitivity = float(self._val(self.status.get("sensitivity", 1.0), 1.0))  # type: ignore
        self.spts = int(self._val(self.status.get("samples_per_ts", 1), 1))  # type: ignore
        self.usb_dps = int(self._val(self.status.get("usb_dps", 0), 0) or 0)  # type: ignore

        self.sample_size = TypeConversion.check_type_length(self.data_type)
        self.np_dtype = TypeConversion.get_np_dtype(self.data_type)
        self.data_size = (
            int(self.sample_size) * self.dim
            if self.spts == 0
            else int(self.sample_size) * self.spts * self.dim
        )
        self.time_size = 0 if self.spts == 0 else 8
        self.packet_size = self.data_size + self.time_size
        self.usb_packet_len = (self.usb_dps + 4) if self.usb_dps else 0
        self._usb_rx = bytearray()
        self._rx = bytearray()
        self._pending_chunks: list[tuple[Optional[float], np.ndarray]] = []
        self._last_rx_time = time.monotonic()
        self._last_sensor_ts: Optional[float] = None
        self._expected_ts_dt = self._expected_sensor_ts_dt() if axes is None else None

    @property
    def meta(self) -> SensorMeta:
        """Return calibrated sensor metadata, resolving ODR from multiple possible status keys."""
        odr_hz = self._coerce_float(self._val(self.status.get("measodr", None), None))
        if odr_hz is None:
            odr_hz = self._coerce_float(self._val(self.status.get("odr", None), None))
        if odr_hz is None:
            odr_hz = max_numeric_value(self.status.get("measodr"))
        if odr_hz is None:
            odr_hz = max_numeric_value(self.status.get("odr"))
        fs_g = self.sensitivity * (2**15) if self.axes == 3 else None
        return {
            "enabled": bool(self._val(self.status.get("enable", False), False)),
            "odr_hz": odr_hz,
            "fs_g": fs_g,
            "samples_per_ts": self.spts,
        }

    def _check_timeout(self) -> None:
        """Check if the time since the last received data exceeds the timeout threshold, and raise TimeoutError if so."""
        if self.timeout_s is None:
            return
        if (time.monotonic() - self._last_rx_time) >= self.timeout_s:
            raise TimeoutError(
                f"No data received from '{self.sensor_name}' for {self.timeout_s} s."
            )

    def _feed(self, payload: bytes) -> None:
        """Decode binary data bytes and buffer calibrated sample chunks."""
        self._rx.extend(payload)
        while len(self._rx) >= self.packet_size:
            ts = None
            packet = bytes(self._rx[: self.packet_size])
            data_bytes = packet
            if self.time_size:
                data_bytes = packet[: self.data_size]
                ts = struct.unpack("=d", packet[self.data_size : self.packet_size])[0] # =d specifies native and double
                if not self._valid_sensor_ts(ts):
                    del self._rx[0]
                    continue

            del self._rx[: self.packet_size]
            if ts is not None:
                self._last_sensor_ts = ts

            if self.sample_size == 3:
                data_bytes = TypeConversion.int24_buffer_to_int32_buffer(data_bytes)

            samples = np.frombuffer(data_bytes, dtype=self.np_dtype)
            n = samples.size // self.dim
            if n <= 0:
                continue

            data = (
                samples[: n * self.dim]
                .reshape(n, self.dim)
                .astype(np.float32, copy=False)
                * self.sensitivity
            )
            if self.axes is not None:
                data = data[:, : self.axes]
            self._pending_chunks.append((ts, data))

    def _valid_sensor_ts(self, ts: float) -> bool:
        """Reject implausible timestamps while resynchronizing the byte stream."""
        if not np.isfinite(ts) or abs(ts) > 1e12:
            return False
        if self._last_sensor_ts is None:
            return True
        dt = ts - self._last_sensor_ts
        if not (-1.0 <= dt <= 10.0):
            return False
        if self._expected_ts_dt is None or dt <= 0.0:
            return True
        periods = round(dt / self._expected_ts_dt)
        if periods < 1:
            return False
        expected_dt = periods * self._expected_ts_dt
        tolerance = max(0.002, 0.25 * expected_dt)
        return abs(dt - expected_dt) <= tolerance

    def _expected_sensor_ts_dt(self) -> Optional[float]:
        """Return expected seconds between device timestamps when ODR metadata is usable."""
        if self.spts <= 0:
            return None
        odr_hz = self._coerce_float(self._val(self.status.get("measodr", None), None))
        if odr_hz is None:
            odr_hz = self._coerce_float(self._val(self.status.get("odr", None), None))
        if odr_hz is None:
            odr_hz = max_numeric_value(self.status.get("measodr"))
        if odr_hz is None:
            odr_hz = max_numeric_value(self.status.get("odr"))
        if odr_hz is None or not np.isfinite(odr_hz) or odr_hz <= 10.0:
            return None
        if self.spts > 1 and odr_hz < 1000.0:
            return 1.0 / odr_hz
        return float(self.spts) / odr_hz

    def poll_chunk(self) -> Optional[tuple[Optional[float], np.ndarray]]:
        """Poll for the next available chunk of sensor data, checking for timeouts and buffering incoming packets."""
        self._check_timeout()
        if self._pending_chunks:
            return self._pending_chunks.pop(0)

        instance = self.session.hsd_link_instance
        if instance is None:
            raise RuntimeError("HSDLink instance is not initialized.")
        res = cast(Any, instance).get_sensor_data(
            self.session.device_id, self.sensor_name, ss_id=0
        )
        if res is None:
            return None
        _, chunk = res
        if not chunk:
            return None

        self._last_rx_time = time.monotonic()

        if self.usb_packet_len:
            self._usb_rx.extend(chunk)
            while len(self._usb_rx) >= self.usb_packet_len:
                payload = bytes(self._usb_rx[4 : self.usb_packet_len])
                del self._usb_rx[: self.usb_packet_len]
                if payload:
                    self._feed(payload)
        else:
            self._feed(chunk)

        if self._pending_chunks:
            return self._pending_chunks.pop(0)
        return None

if __name__ == "__main__":
    acc_name = "iis3dwb_acc"
    ultrasound_name = "imp23absu_mic"
    window_s = 3.0
    n_readings = 5
    us_count = 0
    start_s = time.monotonic()
    acc_offset0: Optional[float] = None
    acc_offset: Optional[float] = None
    us_offset0: Optional[float] = None
    us_offset: Optional[float] = None

    with SensorKitSession(device_id=0, acquisition_folder=".") as session:
        us_status = session.get_sensor_status(ultrasound_name)
        us_odr_max = max_numeric_value(us_status.get("odr"))
        session.set_sensor_enable(ultrasound_name, True)
        if us_odr_max is not None:
            session.set_sensor_odr(ultrasound_name, us_odr_max)

        acc_stream = SensorStream(session, acc_name, timeout_s=2.0, axes=3)
        us_stream = SensorStream(session, ultrasound_name, timeout_s=2.0, axes=None)
        meta = acc_stream.meta
        us_meta = us_stream.meta

        session.start_log()
        print(f"Enabled: {meta['enabled']}")
        print(
            f"ODR: {meta['odr_hz']:.1f} Hz"
            if meta["odr_hz"] is not None
            else "ODR: (unknown)"
        )
        if meta["fs_g"] is not None:
            print(f"FS: {meta['fs_g']:.1f} g")
        if us_meta or us_odr_max is not None:
            us_odr = us_meta["odr_hz"]
            print(
                f"Ultrasound ODR: {float(us_odr):.1f} Hz"
                if us_odr is not None
                else (
                    f"Ultrasound ODR target: {us_odr_max:.1f} Hz"
                    if us_odr_max is not None
                    else "Ultrasound ODR: (unknown)"
                )
            )
        print()

        for i in range(1, n_readings + 1):
            us_start = us_count
            sum_xyz = np.zeros(3, dtype=np.float64)
            count = 0
            deadline = time.time() + window_s
            while time.time() < deadline:
                us_item = us_stream.poll_chunk()
                if us_item is not None:
                    us_ts, us_data = us_item
                    if us_ts is not None and np.isfinite(us_ts):
                        us_offset = (time.monotonic() - start_s) - float(us_ts)
                        if us_offset0 is None:
                            us_offset0 = us_offset
                    us_count += int(us_data.shape[0])

                sk_item = acc_stream.poll_chunk()
                if sk_item is not None:
                    acc_ts, xyz = sk_item
                    if acc_ts is not None and np.isfinite(acc_ts):
                        acc_offset = (time.monotonic() - start_s) - float(acc_ts)
                        if acc_offset0 is None:
                            acc_offset0 = acc_offset
                    sum_xyz += xyz.sum(axis=0, dtype=np.float64)
                    count += xyz.shape[0]
                    continue
                if us_item is None:
                    time.sleep(0.001)
            avg = sum_xyz / max(1, count)

            print(f"--- Reading {i} ---")
            print(f"x: {float(avg[0])}")
            print(f"y: {float(avg[1])}")
            print(f"z: {float(avg[2])}")
            us_delta = max(0, us_count - us_start)
            print(
                f"ultrasound_sps: {us_delta / window_s:.1f} (samples: {us_delta} over {window_s:.1f}s)"
            )
            if acc_offset is not None and acc_offset0 is not None:
                print(
                    f"acc_ts_offset: {acc_offset:+.6f}s (drift: {acc_offset - acc_offset0:+.6f}s)"
                )
            else:
                print("acc_ts_offset: n/a")
            if us_offset is not None and us_offset0 is not None:
                print(
                    f"ultrasound_ts_offset: {us_offset:+.6f}s (drift: {us_offset - us_offset0:+.6f}s)"
                )
            else:
                print("ultrasound_ts_offset: n/a")
            print()
