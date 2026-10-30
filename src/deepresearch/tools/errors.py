"""Explicit failures raised by the controlled tool layer."""


class ToolError(Exception):
    """Base exception for tool registration, policy, validation, and execution failures."""


class ToolNotFoundError(ToolError):
    """The requested tool is not registered."""


class ToolAlreadyRegisteredError(ToolError):
    """A second definition attempted to reuse an existing tool name."""


class ToolPolicyDeniedError(ToolError):
    """Local policy rejected the proposed action."""


class ToolApprovalRequiredError(ToolPolicyDeniedError):
    """Execution paused because a local operator must decide an approval request."""

    def __init__(self, message: str, *, approval_id: str) -> None:
        super().__init__(message)
        self.approval_id = approval_id


class ToolInputValidationError(ToolError):
    """Arguments did not satisfy the registered Pydantic model."""


class ToolExecutionError(ToolError):
    """The authorized tool failed while executing."""


class ToolTimeoutError(ToolExecutionError):
    """The tool exceeded its declared timeout."""
