"""Inference platform exceptions without heavyweight runtime imports."""


class ModelMetadataUnavailableError(OSError):
    """Model configuration is absent from the selected local cache."""
