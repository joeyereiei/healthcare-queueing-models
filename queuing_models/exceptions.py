"""Custom exceptions for the queuing models.

All exceptions subclass ``ValueError`` so callers can catch them either
specifically or as ordinary value errors.
"""


class InvalidModelError(ValueError):
    """Raised when an unknown model name is given."""


class MissingColumnsError(ValueError):
    """Raised when an experiment DataFrame lacks required columns."""


class InvalidParameterError(ValueError):
    """Raised when parameters are invalid or give no steady state."""
