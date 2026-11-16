"""Phase 3A B-ACN method-selection pipeline.

The package is deliberately separate from the accepted Phase 0--2 producers.
It consumes their immutable artifacts and writes only to Phase 3A campaign
roots supplied by an explicit execution configuration.
"""

from .schemas import PHASE3A_SCHEMA_VERSION, stable_object_id

__all__ = ["PHASE3A_SCHEMA_VERSION", "stable_object_id"]
