"""revit-api-docs: Revit API reference search for MCP hosts.

An optional companion to revit-bridge for models that do not know the Revit
API well. It answers two questions over a local copy of the Revit 2026 API
reference (27,596 entries) and the SDK samples: "which API do I call?" and
"how does the SDK use it?". The data set is downloaded from a GitHub Release
on first run; the package never talks to Revit.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("revit-api-docs")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
