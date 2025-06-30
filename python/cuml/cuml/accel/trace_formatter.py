#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trace Formatter for cuML accelerator operations."""

import os
from datetime import datetime

from opentelemetry.sdk.trace import ReadableSpan
from tabulate import tabulate


class AnnotatedTraceFormatter:
    """Formats trace output to show calling code with acceleration status icons."""

    def __init__(self):

        # ANSI color codes for nice formatting
        self.colors = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "gray": "\033[90m",
        }

    def colorize(self, text: str, color: str) -> str:
        """Add color to text if colors are supported."""
        if color in self.colors:
            return f"{self.colors[color]}{text}{self.colors['reset']}"
        return text

    def format_duration(self, duration_seconds: float) -> str:
        """Format duration in a human-readable way."""
        if duration_seconds < 0.001:
            return f"{duration_seconds * 1_000_000:.2f}μs"
        elif duration_seconds < 1:
            return f"{duration_seconds * 1000:.2f}ms"
        else:
            return f"{duration_seconds:.3f}s"

    def format_timestamp(self, timestamp_str: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.strftime("%H:%M:%S.%f")[:-3]  # Show milliseconds
        except Exception:
            return timestamp_str

    def get_acceleration_icon(
        self, operation_type: str, status_code: str
    ) -> str:
        """Get acceleration status icon for the operation."""
        if status_code != "OK":
            return self.colorize("✗", "red")  # Failed operations

        # Map operation types to acceleration status
        acceleration_map = {
            "GPU_CALL": "GPU",  # GPU accelerated
            "GPU_INIT": "GPU",  # GPU accelerated
            "CPU_CALL": "CPU",  # CPU fallback
            "SYNC_PARAMS": "SYNC",  # Internal sync
            "SYNC_ATTRS": "SYNC",  # Internal sync
            "RECONSTRUCT": "RECON",  # Reconstruction
            "USER_SPAN": "USER",  # User span
        }

        icon = acceleration_map.get(operation_type, "???")

        # Color code based on acceleration type
        if operation_type in ["GPU_CALL", "GPU_INIT"]:
            return self.colorize(icon, "green")  # GPU accelerated
        elif operation_type == "CPU_CALL":
            return self.colorize(icon, "yellow")  # CPU fallback
        elif operation_type in ["SYNC_PARAMS", "SYNC_ATTRS", "RECONSTRUCT"]:
            return self.colorize(icon, "blue")  # Internal operations
        elif operation_type == "USER_SPAN":
            return self.colorize(icon, "cyan")  # User spans
        else:
            return self.colorize(icon, "gray")  # Unknown

    def format_span(self, span) -> str:
        """Format a span directly from an OpenTelemetry span object."""
        from datetime import datetime, timezone

        # Convert integer timestamp to ISO format
        def format_timestamp(timestamp_ns):
            if timestamp_ns is None:
                return ""
            try:
                # Convert nanoseconds to seconds and create datetime
                timestamp_s = timestamp_ns / 1_000_000_000
                dt = datetime.fromtimestamp(timestamp_s, tz=timezone.utc)
                return dt.isoformat()
            except (TypeError, ValueError, OSError):
                # Fallback: just return the raw value as string
                return str(timestamp_ns)

        # Calculate duration from start and end times
        duration_seconds = 0.0
        if span.start_time is not None and span.end_time is not None:
            duration_ns = span.end_time - span.start_time
            duration_seconds = duration_ns / 1_000_000_000

        # Get attributes and add duration if not already present
        attributes = dict(span.attributes) if span.attributes else {}
        if "cuml.duration_seconds" not in attributes:
            attributes["cuml.duration_seconds"] = duration_seconds

        # Extract key information
        operation_type = attributes.get("cuml.operation_type", "UNKNOWN")
        duration = attributes.get("cuml.duration_seconds", duration_seconds)
        status_code = (
            span.status.status_code.name if span.status else "UNKNOWN"
        )
        start_time = format_timestamp(span.start_time)

        # Check if this is a user span
        span_type = attributes.get("cuml.span_type", "")
        span_name = attributes.get("cuml.span_name", "")
        is_user_span = span_type == "user_span"

        if is_user_span:
            operation_type = "USER_SPAN"

        # Extract caller information
        caller_code = attributes.get("cuml.caller_code", "")

        # Get acceleration icon
        icon = self.get_acceleration_icon(operation_type, status_code)

        # Build the output line
        parts = [icon]

        # Add duration
        duration_str = self.format_duration(duration)
        parts.append(self.colorize(f"({duration_str})", "gray"))

        # Add timestamp
        if start_time:
            timestamp_str = self.format_timestamp(start_time)
            parts.append(self.colorize(f"[{timestamp_str}]", "gray"))

        # Add the calling code
        if caller_code and caller_code not in ["unknown", "unreadable"]:
            parts.append(caller_code)
        else:
            # Fallback for user spans or when code is not available
            if is_user_span:
                parts.append(f'with create_span("{span_name}"):')
            else:
                parts.append(
                    f"{operation_type}."
                    f"{attributes.get('cuml.method_name', 'unknown')}"
                )

        return " ".join(parts)

    def __call__(self, span: ReadableSpan) -> str:
        """Format a single span for annotated display."""
        return self.format_span(span) + "\n"


class ScriptAnnotatedTraceFormatter:
    """Formats trace output to show the entire script with acceleration status icons."""

    def __init__(self):

        # ANSI color codes for nice formatting
        self.colors = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "red": "\033[91m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "blue": "\033[94m",
            "magenta": "\033[95m",
            "cyan": "\033[96m",
            "gray": "\033[90m",
        }

    def colorize(self, text: str, color: str) -> str:
        """Add color to text if colors are supported."""
        if color in self.colors:
            return f"{self.colors[color]}{text}{self.colors['reset']}"
        return text

    def format_duration(self, duration_seconds: float) -> str:
        """Format duration in a human-readable way."""
        if duration_seconds < 0.001:
            return f"{duration_seconds * 1_000_000:.2f}μs"
        elif duration_seconds < 1:
            return f"{duration_seconds * 1000:.2f}ms"
        else:
            return f"{duration_seconds:.3f}s"

    def format_timestamp(self, timestamp_str: str) -> str:
        """Format timestamp for display."""
        try:
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.strftime("%H:%M:%S.%f")[:-3]  # Show milliseconds
        except Exception:
            return timestamp_str

    def get_acceleration_icon(
        self, operation_type: str, status_code: str
    ) -> str:
        """Get acceleration status icon for the operation."""
        if status_code != "OK":
            return self.colorize("✗", "red")  # Failed operations

        # Map operation types to acceleration status
        acceleration_map = {
            "GPU_CALL": "GPU",  # GPU accelerated
            "GPU_INIT": "GPU",  # GPU accelerated
            "CPU_CALL": "CPU",  # CPU fallback
            "SYNC_PARAMS": "SYNC",  # Internal sync
            "SYNC_ATTRS": "SYNC",  # Internal sync
            "RECONSTRUCT": "RECON",  # Reconstruction
            "USER_SPAN": "USER",  # User span
        }

        icon = acceleration_map.get(operation_type, "")

        # Color code based on acceleration type
        if operation_type in ["GPU_CALL", "GPU_INIT"]:
            return self.colorize(icon, "green")  # GPU accelerated
        elif operation_type == "CPU_CALL":
            return self.colorize(icon, "yellow")  # CPU fallback
        elif operation_type in ["SYNC_PARAMS", "SYNC_ATTRS", "RECONSTRUCT"]:
            return self.colorize(icon, "blue")  # Internal operations
        elif operation_type == "USER_SPAN":
            return self.colorize(icon, "cyan")  # User spans
        else:
            return self.colorize(icon, "gray")  # Unknown

    def format_script_with_annotations(
        self, script_path: str, spans: list
    ) -> str:
        """Format the entire script with acceleration annotations."""
        try:
            with open(script_path, "r") as f:
                script_lines = f.readlines()
        except (IOError, OSError):
            return "Could not read script file for annotation"

        # Group spans by line number
        line_annotations = {}
        for span in spans:
            # Extract caller information from span attributes
            attributes = dict(span.attributes) if span.attributes else {}
            caller_file = attributes.get("cuml.caller_file", "")
            caller_line_raw = attributes.get("cuml.caller_line", 0)

            # Convert caller_line to integer (it might be stored as string)
            try:
                caller_line = int(caller_line_raw)
            except (ValueError, TypeError):
                caller_line = 0

            # Only process spans from the main script
            if os.path.basename(caller_file) == os.path.basename(script_path):
                if caller_line not in line_annotations:
                    line_annotations[caller_line] = {
                        "operations": [],
                        "total_duration": 0.0,
                        "first_timestamp": None,
                        "has_cpu_fallback": False,
                        "has_gpu_acceleration": False,
                    }

                operation_type = attributes.get(
                    "cuml.operation_type", "UNKNOWN"
                )
                status_code = (
                    span.status.status_code.name if span.status else "UNKNOWN"
                )

                # Calculate duration from start and end times
                duration_seconds = 0.0
                if span.start_time is not None and span.end_time is not None:
                    duration_ns = span.end_time - span.start_time
                    duration_seconds = duration_ns / 1_000_000_000

                # Convert timestamp to ISO format for display
                from datetime import datetime, timezone

                start_time = ""
                if span.start_time is not None:
                    try:
                        timestamp_s = span.start_time / 1_000_000_000
                        dt = datetime.fromtimestamp(
                            timestamp_s, tz=timezone.utc
                        )
                        start_time = dt.isoformat()
                    except (TypeError, ValueError, OSError):
                        start_time = str(span.start_time)

                # Check if this is a user span
                span_type = attributes.get("cuml.span_type", "")
                if span_type == "user_span":
                    operation_type = "USER_SPAN"

                # Track operation types and timing
                line_annotations[caller_line]["operations"].append(
                    {
                        "type": operation_type,
                        "status": status_code,
                        "duration": duration_seconds,
                    }
                )
                line_annotations[caller_line][
                    "total_duration"
                ] += duration_seconds

                if line_annotations[caller_line]["first_timestamp"] is None:
                    line_annotations[caller_line][
                        "first_timestamp"
                    ] = start_time

                # Track acceleration status
                if operation_type == "CPU_CALL":
                    line_annotations[caller_line]["has_cpu_fallback"] = True
                elif operation_type in ["GPU_CALL", "GPU_INIT"]:
                    line_annotations[caller_line][
                        "has_gpu_acceleration"
                    ] = True

        # Prepare table data
        table_data = []
        for line_num, line in enumerate(script_lines, 1):
            # Preserve original indentation by not stripping
            line_content = line.rstrip("\n")

            if line_num in line_annotations:
                # This line has profiling data
                line_data = line_annotations[line_num]

                # Extract timestamp
                timestamp_str = ""
                if line_data["first_timestamp"]:
                    timestamp_str = self.format_timestamp(
                        line_data["first_timestamp"]
                    )

                # Extract duration
                duration_str = self.format_duration(
                    line_data["total_duration"]
                )

                # Determine the acceleration icon
                if line_data["has_cpu_fallback"]:
                    # If any operation fell back to CPU, show CPU fallback
                    acceleration_icon = self.colorize("CPU", "yellow")
                elif line_data["has_gpu_acceleration"]:
                    # If all operations were GPU accelerated
                    acceleration_icon = self.colorize("GPU", "green")
                else:
                    # Internal operations (SYNC_ATTRS, SYNC_PARAMS, etc.)
                    acceleration_icon = self.colorize("SYNC", "blue")

                table_data.append(
                    [
                        line_num,
                        timestamp_str,
                        duration_str,
                        acceleration_icon,
                        line_content,
                    ]
                )
            else:
                # Regular line without profiling data
                table_data.append([line_num, "", "", "", line_content])

        table = tabulate(
            table_data,
            tablefmt="plain",
            colalign=["right", "left", "left", "left", None],
        )

        output_lines = []
        output_lines.append(
            f"{self.colorize('Script:', 'bold')} "
            f"{os.path.basename(script_path)}"
        )
        output_lines.append("")
        output_lines.append(table)
        return "\n".join(output_lines)
