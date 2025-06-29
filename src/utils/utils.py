"""Utility functions with improved error handling and type safety."""

import re
from typing import Optional

from core.exceptions import ServiceError
from core.logging_config import get_logger

logger = get_logger(__name__)


def extract_json(text: str) -> str:
    """Extract JSON array from text response.

    Args:
        text: Input text containing JSON array

    Returns:
        Extracted JSON array as string

    Raises:
        ParsingException: If no valid JSON array found
    """
    if not text or not isinstance(text, str):
        logger.error(f"Invalid input for JSON extraction: {type(text)}")
        raise ServiceError("Input text is empty or not a string")

    # Try to find JSON array pattern
    patterns = [
        r"\[\s*{.*}\s*\]",  # Standard array pattern
        r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]",  # More specific pattern
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            result = match.group(0)
            logger.debug(f"Successfully extracted JSON array of length: {len(result)}")
            return result

    logger.error(f"No JSON array found in text of length: {len(text)}")
    raise ServiceError("Không tìm thấy JSON array trong phản hồi")


def validate_patient_id(patient_id: Optional[str]) -> bool:
    """Validate patient ID format.

    Args:
        patient_id: Patient identifier

    Returns:
        True if valid, False otherwise
    """
    if not patient_id or not isinstance(patient_id, str):
        return False

    # Validation - adjust according to your ID format requirements
    return len(patient_id.strip()) >= 3 and patient_id.strip().isalnum()


def sanitize_text(text: Optional[str], max_length: Optional[int] = None) -> str:
    """Sanitize and clean text input.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Sanitization
    sanitized = text.strip()

    # Remove excessive whitespace
    sanitized = re.sub(r"\s+", " ", sanitized)

    # Truncate if necessary
    if max_length and len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")

    return sanitized
