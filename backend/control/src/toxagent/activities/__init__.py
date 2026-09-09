"""Product-facing projection of audited runtime/tool events.

Raw tool events remain the audit contract.  This module deliberately maps them
to a small, stable activity vocabulary so transcript UX does not depend on a
specific harness implementation or expose internal tool names.
"""

from .presentation import activity_for_tool

__all__ = ["activity_for_tool"]
