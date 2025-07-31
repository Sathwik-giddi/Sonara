"""Sonara skill layer. Importing a pack registers its tools.

Add a pack by writing the module and importing it here. That is the whole
extensibility contract - "everything a human can do" is one import away, which is
what makes it an architecture rather than a backlog.
"""

from . import notes, pc, web  # noqa: F401  - imported for their registration side effects
from .base import Risk, Tool, ToolRegistry, registry
from .safety import AUDIT_LOG, CONFIRM_PHRASE, Blocked, ConfirmationRequired, Executor, Outcome

__all__ = [
    "AUDIT_LOG", "Blocked", "CONFIRM_PHRASE", "ConfirmationRequired", "Executor",
    "Outcome", "Risk", "Tool", "ToolRegistry", "registry",
]
