class AppError(Exception):
    """Base exception for the project."""

class ConfigError(AppError):
    """Configuration is missing/invalid."""

class ExternalServiceError(AppError):
    """External dependency failed (API/LLM/etc)."""

class ValidationError(AppError):
    """Raised when LLM output or tool arguments fail validation."""
