from typing import Any
from ckanapi import RemoteCKAN

__all__ = ["search", "get_dataset"]


def _text(value: str | dict[str, str] | None, language: str) -> str:
    if isinstance(value, dict):
        for key in (language, "de", "en", "fr", "it"):
            if value.get(key):
                return value[key]
        return next((text for text in value.values() if text), "")
    return value or ""


def _dataset(metadata: dict[str, Any], language: str) -> dict[str, Any]:
    return {
        **metadata,
        "title": _text(metadata.get("title"), language),
        "description": _text(metadata.get("notes"), language),
        "resources": metadata.get("resources") or [],
    }


def search(query: str, limit: int = 10, offset: int = 0, *, language: str = "de") -> list[dict[str, Any]]:
    """Search opendata.swiss and return dataset metadata with localized titles and descriptions.

    Resources include download URLs and formats. Language falls back to another
    available translation. Use offset to page through results. CKAN and network
    errors propagate to the caller; requests time out after 30 seconds.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if offset < 0:
        raise ValueError("offset must be nonnegative")
    with RemoteCKAN("https://opendata.swiss", get_only=True) as client:
        client.base_url = "api/3/action/"
        result = client.call_action("package_search", {"q": query, "rows": limit, "start": offset}, requests_kwargs={"timeout": 30})
    return [_dataset(metadata, language) for metadata in result["results"]]


def get_dataset(id_or_name: str, *, language: str = "de") -> dict[str, Any]:
    """Retrieve dataset metadata and resources by CKAN ID or name.

    Titles and descriptions use the requested language with a fallback to another
    available translation. CKAN and network errors propagate to the caller;
    requests time out after 30 seconds.
    """
    with RemoteCKAN("https://opendata.swiss", get_only=True) as client:
        client.base_url = "api/3/action/"
        metadata = client.call_action("package_show", {"id": id_or_name}, requests_kwargs={"timeout": 30})
    return _dataset(metadata, language)
