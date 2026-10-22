"""Custom exceptions and error handlers for OceanScope backend API.

All errors are formatted as:
{"error": {"code": "<snake_case>", "message": "<human readable>", "details": {...}|null}}
per PRD_BACKEND Section 4.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class APIError(HTTPException):
    """Base API exception that renders to standard error envelope."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details
        super().__init__(
            status_code=status_code,
            detail={"error": {"code": code, "message": message, "details": details}},
        )


class DataNotLoadedError(APIError):
    def __init__(self, message: str = "Data store is not loaded") -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="data_not_loaded",
            message=message,
        )


class OutOfRegionError(APIError):
    def __init__(
        self, message: str = "Requested coordinates fall outside the modelled region"
    ) -> None:
        super().__init__(
            status_code=422,
            code="out_of_region",
            message=message,
        )


class BadDepthError(APIError):
    def __init__(
        self, message: str = "Requested depth does not match any valid model depth"
    ) -> None:
        super().__init__(
            status_code=422,
            code="bad_depth",
            message=message,
        )


class BothDepthParamsError(APIError):
    def __init__(self, message: str = "Provide either depth_m or depth_index, not both") -> None:
        super().__init__(
            status_code=422,
            code="both_depth_params",
            message=message,
        )


class DateUnavailableError(APIError):
    def __init__(self, message: str = "Requested date is outside available dates") -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="date_unavailable",
            message=message,
        )


class ReferenceUnavailableError(APIError):
    def __init__(self, message: str = "GLORYS reference is not available for live dates") -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="reference_unavailable",
            message=message,
        )


class TestNotEvaluatedError(APIError):
    def __init__(self, message: str = "Test split has not been evaluated") -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="test_not_evaluated",
            message=message,
        )


class HeadlineNotAvailableError(APIError):
    def __init__(self, message: str = "Headline evaluation metrics are not available") -> None:
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            code="headline_not_available",
            message=message,
        )


class RangeTooLargeError(APIError):
    def __init__(
        self, message: str = "Requested time range exceeds maximum limit of 366 days"
    ) -> None:
        super().__init__(
            status_code=422,
            code="range_too_large",
            message=message,
        )


class InvalidDateFormatError(APIError):
    def __init__(self, message: str = "Date must be formatted as YYYY-MM-DD") -> None:
        super().__init__(
            status_code=422,
            code="invalid_date_format",
            message=message,
        )


class ChatDisabledError(APIError):
    def __init__(self, message: str = "Chat assistant is disabled") -> None:
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            code="chat_disabled",
            message=message,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle custom APIError exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
    )


async def http_exception_handler(
    request: Request, exc: HTTPException | StarletteHTTPException
) -> JSONResponse:
    """Handle standard FastAPI and Starlette HTTPException instances."""
    if isinstance(exc.detail, dict) and "error" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    code = (
        "not_found"
        if exc.status_code == 404
        else ("method_not_allowed" if exc.status_code == 405 else "http_error")
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": code, "message": str(exc.detail), "details": None}},
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI request validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Invalid request parameters",
                "details": {"errors": exc.errors()},
            }
        },
    )
