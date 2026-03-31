from __future__ import annotations

from connect.transport.http import HttpTransport


def test_extract_error_message_prefers_nested_error_message() -> None:
    transport = HttpTransport.__new__(HttpTransport)

    message = transport._extract_error_message(
        {
            "error": {
                "error": {
                    "message": "vendor nested message",
                }
            }
        }
    )

    assert message == "vendor nested message"


def test_extract_error_message_uses_detail_when_message_missing() -> None:
    transport = HttpTransport.__new__(HttpTransport)

    message = transport._extract_error_message({"detail": "provider detail text"})

    assert message == "provider detail text"