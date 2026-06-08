"""
UDP receiver for PLC torque data.

Decodes incoming UDP packets from the PLC, extracts the timestamp and torque values.

Can be run as a standalone script to verify PLC connectivity.
"""

import socket
import struct
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import Iterator

IP = "192.168.0.100"
PORT = 53758
PRINT_EVERY_S = 5.0
VALUE_TYPE = "d"  # "d" for FLOAT64 / LREAL PLC values
CHANNELS_PER_SAMPLE = 1
EXPECTED_SAMPLES_HZ = 1000.0
LOG_DIR = Path("log")
HEADER_BYTES = 16
ITEM_SIZES = {
    "d": 8,
    "f": 4,
    "h": 2,
}

PLCPacket = tuple[int, int, int, int, tuple[float, ...]]


def _log_line(logf, message: str) -> None:
    """Write a UTC timestamped message to the UDP logger file."""
    ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    logf.write(f"{ts} {message}\n")
    logf.flush()


def decode_plc_second_nanosecond_to_unix(
    second: int, nanosecond: int, reference_unix_s: float | None = None
) -> float | None:
    """Convert PLC second and nanosecond fields to a Unix timestamp."""
    if second < 0 or second > 59 or nanosecond < 0 or nanosecond > 999_999_999:
        return None

    ref = (
        datetime.fromtimestamp(reference_unix_s, tz=timezone.utc)
        if reference_unix_s is not None
        else datetime.now(timezone.utc)
    )
    dt = ref.replace(second=second, microsecond=nanosecond // 1000)

    # If the offset is
    delta = dt.timestamp() - ref.timestamp()
    if delta > 30.0:
        dt -= timedelta(minutes=1)
    elif delta < -30.0:
        dt += timedelta(minutes=1)
    return dt.timestamp()


class PLCStream:
    """Context-managed UDP stream for decoded PLC packets."""

    def __init__(
        self,
        ip: str = IP,
        port: int = PORT,
        value_type: str = VALUE_TYPE,
        timeout_s: float = 1.0,
        receive_buffer_bytes: int = 32 * 1024 * 1024,
    ) -> None:
        if value_type not in ITEM_SIZES:
            raise ValueError(f"Unsupported PLC value_type: {value_type!r}")
        self.ip = ip
        self.port = port
        self.value_type = value_type
        self.timeout_s = timeout_s
        self.receive_buffer_bytes = receive_buffer_bytes
        self.item_size = ITEM_SIZES[value_type]
        self._sock: socket.socket | None = None

    def __enter__(self) -> "PLCStream":
        self.open()
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def open(self) -> None:
        """Open and bind the UDP socket."""
        if self._sock is not None:
            return
        self._ensure_private_ethernet()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Bigger receive buffer reduces packet drops when consumer threads are busy.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self.receive_buffer_bytes)
        sock.bind((self.ip, self.port))
        sock.settimeout(self.timeout_s)
        self._sock = sock

    @staticmethod
    def _ensure_private_ethernet() -> None:
        """Verify the Windows Ethernet interface is configured as Private."""
        if sys.platform != "win32":
            return

        try:
            category = subprocess.check_output(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "(Get-NetConnectionProfile -InterfaceAlias 'Ethernet 2').NetworkCategory"
                    ),
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(
                "Failed to verify Windows network profile for Ethernet 2."
            ) from exc

        if category != "Private":
            raise RuntimeError(
                "Ethernet 2 must use the Private network profile to receive UDP packets on Windows. "
                'Run PowerShell as administrator and execute: '
                '`Set-NetConnectionProfile -InterfaceAlias "Ethernet 2" -NetworkCategory Private`'
            )

    def close(self) -> None:
        """Close the UDP socket if it is open."""
        if self._sock is None:
            return
        self._sock.close()
        self._sock = None

    def iter_packets(
        self,
        stop_event: Event | None = None,
        max_duration_s: float = 0.0,
    ) -> Iterator[PLCPacket]:
        """Yield decoded PLC packets until stopped or timed out."""
        self.open()
        assert self._sock is not None
        start = time.monotonic()
        while stop_event is None or not stop_event.is_set():
            if max_duration_s > 0.0 and (time.monotonic() - start) >= max_duration_s:
                break

            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                continue

            if len(data) < HEADER_BYTES:
                continue

            header_raw = data[:HEADER_BYTES]
            payload = data[HEADER_BYTES:]
            if len(payload) % self.item_size != 0:
                continue

            (
                batch_start_second,
                batch_start_nanosecond,
                batch_end_second,
                batch_end_nanosecond,
            ) = struct.unpack(">IIII", header_raw)
            n_samples = len(payload) // self.item_size
            values = struct.unpack(">" + str(n_samples) + self.value_type, payload)
            yield (
                batch_start_second,
                batch_start_nanosecond,
                batch_end_second,
                batch_end_nanosecond,
                values,
            )


if __name__ == "__main__":
    print(f"[UDP] LISTENING on {(IP, PORT)}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
    logf = open(LOG_DIR / log_name, "a", encoding="utf-8")

    total_packets = 0
    window_packets = 0
    window_values = 0
    start = time.monotonic()
    latest_delay_ms: float | None = None
    latest_fill_delay_ms: float | None = None
    try:
        latest_values: tuple[float, ...] = ()
        with PLCStream() as plc_stream:
            for (
                batch_start_second,
                batch_start_nanosecond,
                batch_end_second,
                batch_end_nanosecond,
                values,
            ) in plc_stream.iter_packets():
                n_values = len(values)
                total_packets += 1
                window_packets += 1
                window_values += n_values
                batch_start_time_s = decode_plc_second_nanosecond_to_unix(
                    batch_start_second, batch_start_nanosecond
                )
                batch_end_time_s = decode_plc_second_nanosecond_to_unix(
                    batch_end_second, batch_end_nanosecond
                )
                if batch_start_time_s is not None:
                    latest_fill_delay_ms = (time.time() - batch_start_time_s) * 1000.0
                if batch_end_time_s is not None:
                    latest_delay_ms = (time.time() - batch_end_time_s) * 1000.0
                if n_values >= CHANNELS_PER_SAMPLE:
                    latest_values = (float(values[-1]),)
                now = time.monotonic()
                elapsed = now - start
                if elapsed >= PRINT_EVERY_S:
                    pkt_hz = window_packets / elapsed
                    rx_sample_hz = (window_values / CHANNELS_PER_SAMPLE) / elapsed
                    pct = (rx_sample_hz / EXPECTED_SAMPLES_HZ) * 100.0
                    line = (
                        f"[RATE] pkt={pkt_hz:.1f} Hz rx_sample={rx_sample_hz:.1f} Hz "
                        f"({window_packets} pkts/{elapsed:.2f}s, total={total_packets}) "
                        f"target={pct:.1f}% "
                        f"fill_start_delay={'n/a' if latest_fill_delay_ms is None else f'{latest_fill_delay_ms:.1f} ms'} "
                        f"full_buffer_delay={'n/a' if latest_delay_ms is None else f'{latest_delay_ms:.1f} ms'} "
                        f"latest=({', '.join(f'{v:.6f}' for v in latest_values)})"
                    )
                    print(line)
                    _log_line(logf, line)
                    start = now
                    window_packets = 0
                    window_values = 0
    except KeyboardInterrupt:
        line = "\n[UDP] Stopped."
        print(line)
        _log_line(logf, line.strip())
    finally:
        logf.close()
