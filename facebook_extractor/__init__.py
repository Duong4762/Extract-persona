"""Facebook profile and content persona extraction pipeline."""

from .cli import main
from .config import Config

__all__ = ["Config", "main"]
