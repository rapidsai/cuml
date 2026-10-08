# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide benchmark logging that does not interrupt execution."""

import logging


class _BenchmarkLogger(logging.LoggerAdapter):
    """Suppress logging failures during benchmark execution."""

    def log(self, level, msg, *args, **kwargs):
        """Log a message without propagating logging errors.

        Parameters
        ----------
        level : int
            Logging severity.
        msg : str
            Message format string.
        *args : Any
            Message formatting arguments.
        **kwargs : Any
            Keyword arguments passed to the logger.
        """
        try:
            super().log(level, msg, *args, **kwargs)
        except Exception:
            pass

    def handle(self, record):
        """Forward an enabled log record without propagating logging errors.

        Parameters
        ----------
        record : logging.LogRecord
            Record to forward to the underlying logger.
        """
        try:
            if self.isEnabledFor(record.levelno):
                self.logger.handle(record)
        except Exception:
            pass


_logger = logging.getLogger("cuml.benchmark")
_logger.addHandler(logging.NullHandler())
logger = _BenchmarkLogger(_logger, {})
