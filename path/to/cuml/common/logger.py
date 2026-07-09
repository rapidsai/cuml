# Modified cuml.common.logger module to add debug message for label permutation invariance
import logging

logger = logging.getLogger(__name__)

def debug(message):
    """Log a debug message"""
    logger.debug(message)