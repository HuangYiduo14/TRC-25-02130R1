"""Bounded Phase 5 backbone-compilation pilot utilities.

The package contains offline scientific oracles and explicit runtime function
classes only.  It does not expose or run a national assignment entry point.
"""

from .bundle_oracle import (
    BundleSpec,
    OracleResult,
    exact_bundle_oracle,
    exact_series_oracle,
    get_exact_root_call_count,
    reset_exact_root_call_count,
)

__all__ = [
    "BundleSpec",
    "OracleResult",
    "exact_bundle_oracle",
    "exact_series_oracle",
    "get_exact_root_call_count",
    "reset_exact_root_call_count",
]
