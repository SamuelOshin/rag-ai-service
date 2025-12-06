from typing import Any, Dict, Optional
from fastapi.responses import JSONResponse


def success_response(status_code: int, message: str, data: Optional[Any] = None) -> Dict[str, Any]:
    """Creates a standardized success response.

    Args:
        status_code (int): HTTP status code for the response.
        message (str): Human-readable success message.
        data (Optional[Any]): The response data payload.

    Returns:
        Dict[str, Any]: Standardized success response dictionary.

    Examples:
        >>> success_response(201, "Document created", {"id": 1})
        {'status': 'success', 'status_code': 201, 'message': 'Document created', 'data': {'id': 1}}
    """
    return {
        "status": "success",
        "status_code": status_code,
        "message": message,
        "data": data
    }


def error_response(status_code: int, message: str, error_code: Optional[str] = None, errors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Creates a standardized error response.

    Args:
        status_code (int): HTTP status code for the error.
        message (str): Human-readable error message.
        error_code (Optional[str]): Optional internal error code.
        errors (Optional[Dict[str, Any]]): Field-level errors.

    Returns:
        Dict[str, Any]: Standardized error response dictionary.

    Examples:
        >>> error_response(400, "Validation failed", "VALIDATION_ERROR", {"email": ["Invalid format"]})
        {'status': 'failure', 'status_code': 400, 'message': 'Validation failed', 'error_code': 'VALIDATION_ERROR', 'errors': {'email': ['Invalid format']}}
    """
    if errors is None:
        errors = {}
    return {
        "status": "failure" if status_code < 500 else "error",
        "status_code": status_code,
        "message": message,
        "error_code": error_code,
        "errors": errors
    }