class SkeletonPipelineException(Exception):
    """Base exception for pipeline errors."""
    pass


class ExtractionException(SkeletonPipelineException):
    """Raised when video extraction fails (codec, corruption, etc.)."""
    pass


class ValidationException(SkeletonPipelineException):
    """Raised when data validation fails (shape, bounds, etc.)."""
    pass


class ConfigurationException(SkeletonPipelineException):
    """Raised when configuration is invalid."""
    pass


class ConversionException(SkeletonPipelineException):
    """Raised when output format conversion fails."""
    pass
