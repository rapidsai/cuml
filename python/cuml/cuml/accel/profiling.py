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
"""Profiling and tracing utilities for cuML accelerator operations."""

import contextlib
import inspect
import json
import time
from datetime import datetime, timezone
from enum import Enum

from cuml.accel.trace_formatter import ScriptAnnotatedTraceFormatter
from cuml.internals import logger
from cuml.internals.interop import UnsupportedOnGPU

# OpenTelemetry imports
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        SpanExportResult,
    )
    from opentelemetry.trace import Status, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    trace = None
    Status = None
    StatusCode = None

    class SpanExportResult(Enum):
        SUCCESS = 0
        FAILURE = 1

    class ConsoleSpanExporter:
        def __init__(self, formatter):
            self.formatter = formatter

        def export(self, spans):
            pass


# Get tracer
if _OTEL_AVAILABLE:
    tracer = trace.get_tracer(__name__)
else:
    tracer = None


def _span_to_json(span):
    """Format a span as a JSON string."""

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

    span_dict = {
        "name": span.name,
        "start_time": format_timestamp(span.start_time),
        "end_time": format_timestamp(span.end_time),
        "status": {
            "status_code": span.status.status_code.name
            if span.status
            else "UNKNOWN"
        },
        "attributes": attributes,
        "resource": {
            "attributes": dict(span.resource.attributes)
            if span.resource
            else {}
        },
    }
    return json.dumps(span_dict) + "\n"


class RawSpanFormatter:
    """Formatter that outputs spans as raw JSON."""

    def __call__(self, span):
        """Format a span as a JSON string."""
        return _span_to_json(span)


class AnnotatedConsoleSpanExporter(ConsoleSpanExporter):
    """Custom console exporter that formats spans with annotated output."""

    def __init__(self, script_path=None, show_attributes=False):
        self.script_path = script_path

    def export(self, spans):
        """Export spans with annotated output."""
        if not spans:
            return

        if self.script_path:
            # Use script annotation formatter
            formatter = ScriptAnnotatedTraceFormatter()
            annotated_script = formatter.format_script_with_annotations(
                self.script_path, spans
            )
            print(annotated_script)
        else:
            # Use regular annotated formatter for individual spans
            for span in spans:
                print(self.formatter(span), end="")
        return SpanExportResult.SUCCESS


def _create_span(operation_type: str, method_name: str, **attributes):
    """Create an OpenTelemetry span for operation tracking."""
    if not _OTEL_AVAILABLE or tracer is None:
        return None

    span_name = f"cuml.accel.{operation_type}.{method_name}"
    span = tracer.start_span(span_name)

    # Add standard attributes
    span.set_attribute("cuml.operation_type", operation_type)
    span.set_attribute("cuml.method_name", method_name)
    span.set_attribute("cuml.component", "estimator_proxy")

    # Add custom attributes
    for key, value in attributes.items():
        span.set_attribute(f"cuml.{key}", str(value))

    return span


def _get_caller_info():
    """Get information about the calling script line."""
    try:
        # Get the current frame and traverse up the call stack
        frame = inspect.currentframe()

        # Skip internal profiling and estimator_proxy calls
        while frame is not None:
            frame = frame.f_back
            if frame is None:
                break

            filename = frame.f_code.co_filename
            # Check if this is user code (not internal cuml/estimator_proxy
            # code)
            is_internal = (
                filename.endswith("estimator_proxy.py")
                or filename.endswith("profiling.py")
                or "contextlib.py" in filename  # Skip contextlib frames
                or "/cuml/accel/"
                in filename.replace("\\", "/")  # More specific path check
            )

            if not is_internal:
                # Found user code, extract information
                lineno = frame.f_lineno
                function = frame.f_code.co_name

                # Get the actual line of code if possible
                try:
                    with open(filename, "r") as f:
                        lines = f.readlines()
                        if 0 <= lineno - 1 < len(lines):
                            code_line = lines[lineno - 1].strip()
                        else:
                            code_line = "unknown"
                except (IOError, OSError):
                    code_line = "unreadable"

                return {
                    "caller_file": filename,
                    "caller_line": lineno,
                    "caller_function": function,
                    "caller_code": code_line,
                }

        # If we didn't find user code, fall back to the immediate caller
        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            caller_frame = frame.f_back
            filename = caller_frame.f_code.co_filename
            lineno = caller_frame.f_lineno
            function = caller_frame.f_code.co_name

            try:
                with open(filename, "r") as f:
                    lines = f.readlines()
                    if 0 <= lineno - 1 < len(lines):
                        code_line = lines[lineno - 1].strip()
                    else:
                        code_line = "unknown"
            except (IOError, OSError):
                code_line = "unreadable"

            return {
                "caller_file": filename,
                "caller_line": lineno,
                "caller_function": function,
                "caller_code": code_line,
            }

    except Exception:
        pass

    return {
        "caller_file": "unknown",
        "caller_line": 0,
        "caller_function": "unknown",
        "caller_code": "unknown",
    }


class LogOperation:
    """Context manager for logging operation details with timing information.

    This class automatically measures the duration of operations and logs them
    using either OpenTelemetry spans (if available) or direct logging as a fallback.

    Args:
        operation_type: Type of operation (GPU_CALL, CPU_CALL, etc.)
        method_name: Name of the method being called
        details: Additional details about the operation
        **attributes: Additional attributes to include in the span

    Example:
        with LogOperation("GPU_CALL", "fit", estimator_name="Ridge"):
            # The operation to be timed and logged
            estimator.fit(X, y)
    """

    def __init__(
        self,
        operation_type: str,
        method_name: str,
        details: str = "",
        **attributes,
    ):
        self.operation_type = operation_type
        self.method_name = method_name
        self.details = details
        self.attributes = attributes
        self.start_time = None
        self.span = None

    def __enter__(self):
        """Enter the context and start timing."""
        self.start_time = time.time()

        if not _OTEL_AVAILABLE:
            # No span to return for logging fallback
            return None

        # Get caller information
        caller_info = _get_caller_info()

        # Create span with operation details
        self.span = _create_span(
            self.operation_type,
            self.method_name,
            **self.attributes,
            **caller_info,  # Include caller information in attributes
        )

        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context, calculate duration, and log the operation."""
        duration = time.time() - self.start_time
        success = exc_type is None

        # Emit log message to console
        self._log_to_console(duration, success, exc_type, exc_val)

        if _OTEL_AVAILABLE and False:
            # Use OpenTelemetry span
            exception_details = str(exc_val) if exc_val else ""
            self._log_span(duration, success, exception_details)

    def _log_to_console(
        self, duration: float, success: bool, exc_type, exc_val
    ):
        """Log operation using direct logging fallback."""
        # Get estimator name from attributes
        estimator_name = self.attributes.get("estimator_name", "Unknown")

        # Format the method call
        method_call = f"{estimator_name}.{self.method_name}()"

        # Analyze the failure reason in case the operation failed
        if exc_type is UnsupportedOnGPU:
            # XXX: we should always provide a more specific error message with
            # this type of exception and not need a fallback message
            exception_details = (
                str(exc_val)
                or "specific input/parameter selection not supported on GPU"
            )
        elif exc_type is not None:
            exception_details = str(exc_val) or "unknown error"

        if self.operation_type == "GPU_INIT":
            # Initialization messages
            if success:
                logger.debug(
                    f"[cuml.accel] Initialized estimator '{estimator_name}' "
                    f"for GPU acceleration"
                )
            else:
                logger.warn(
                    f"[cuml.accel] Failed to initialize '{estimator_name}': "
                    f"{exception_details}"
                )
        elif self.operation_type == "GPU_CALL":
            # Acceleration attempt messages
            if success:
                logger.info(
                    f"[cuml.accel] Successfully accelerated "
                    f"'{method_call}' call"
                )
            else:
                logger.debug(
                    f"[cuml.accel] Unable to accelerate "
                    f"'{method_call}' call: {exception_details}"
                )
        elif self.operation_type == "CPU_CALL":
            # CPU fallback messages
            if success:
                logger.warn(
                    f"[cuml.accel] Falling back to CPU for "
                    f"'{method_call}' call"
                )
            else:
                logger.error(
                    f"[cuml.accel] CPU fallback failed for "
                    f"'{method_call}' call: {exception_details}"
                )
        else:
            # Other operation types (SYNC_ATTRS, SYNC_PARAMS, etc.)
            if success:
                logger.debug(
                    f"[cuml.accel] {self.operation_type}: {method_call}"
                )
            else:
                logger.warn(
                    f"[cuml.accel] {self.operation_type} failed for "
                    f"'{method_call}': {exception_details}"
                )

    def _log_span(
        self, duration: float, success: bool, exception_details: str
    ):
        """Log operation using OpenTelemetry span."""
        if self.span is None:
            return

        # Set span status based on success
        if success:
            self.span.set_status(Status(StatusCode.OK))
        else:
            self.span.set_status(Status(StatusCode.ERROR, exception_details))

        # Add timing information
        self.span.set_attribute("cuml.duration_seconds", duration)
        self.span.set_attribute("cuml.duration", str(duration))
        self.span.set_attribute("cuml.success", str(success))

        # Add details (including exception details if failed)
        final_details = self.details
        if not success:
            final_details = (
                f"{self.details} - Exception: {exception_details}"
                if self.details
                else f"Exception: {exception_details}"
            )
        self.span.set_attribute("cuml.details", final_details)

        # End the span
        self.span.end()


def log_operation(
    operation_type: str,
    method_name: str,
    details: str = "",
    **attributes,
):
    """Log operation details with timing information using OpenTelemetry spans.

    This function returns a context manager that automatically measures the
    duration of operations and logs them using either OpenTelemetry spans
    (if available) or direct logging as a fallback.

    Args:
        operation_type: Type of operation (GPU_CALL, CPU_CALL, etc.)
        method_name: Name of the method being called
        details: Additional details about the operation
        **attributes: Additional attributes to include in the span

    Returns:
        A context manager that measures and logs the operation

    Example:
        with log_operation("GPU_CALL", "fit", estimator_name="Ridge"):
            # The operation to be timed and logged
            estimator.fit(X, y)
    """
    return LogOperation(operation_type, method_name, details, **attributes)


def is_otel_available() -> bool:
    """Check if OpenTelemetry is available."""
    return _OTEL_AVAILABLE


def create_span(name: str, **attributes):
    """Create an explicit span for grouping operations.

    This function returns a context manager that creates a span with the given
    name. All operations within the context will be grouped under this span in
    the trace output.

    Args:
        name: Name of the span for grouping operations
        **attributes: Additional attributes to include in the span

    Returns:
        A context manager that creates and manages the span

    Example:
        with create_span("my_workflow"):
            # All operations here will be grouped under "my_workflow"
            estimator.fit(X, y)
            estimator.predict(X)
    """
    if not _OTEL_AVAILABLE or tracer is None:
        # Return a no-op context manager if OpenTelemetry is not available
        @contextlib.contextmanager
        def noop_context():
            yield

        return noop_context()

    @contextlib.contextmanager
    def span_context():
        # Get caller information for the span
        caller_info = _get_caller_info()

        # Create the span
        span_name = f"user.{name}"
        span = tracer.start_span(span_name)

        # Add standard attributes
        span.set_attribute("cuml.span_type", "user_span")
        span.set_attribute("cuml.span_name", name)
        span.set_attribute("cuml.component", "user_span")

        # Add caller information
        for key, value in caller_info.items():
            span.set_attribute(f"cuml.{key}", str(value))

        # Add custom attributes
        for key, value in attributes.items():
            span.set_attribute(f"cuml.{key}", str(value))

        try:
            yield span
            # Mark span as successful if no exception occurred
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            # Mark span as failed if an exception occurs
            span.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            # End the span
            span.end()

    return span_context()
