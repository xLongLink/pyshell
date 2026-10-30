import pytest
from src import swissdata
from src.shell import PythonShell
from unittest.mock import MagicMock
from ckanapi.errors import NotFound
from requests.exceptions import Timeout


@pytest.fixture
def client(monkeypatch):
    remote = MagicMock()
    monkeypatch.setattr(swissdata, "RemoteCKAN", remote)
    return remote


def test_search_forwards_query_pagination_and_timeout(client):
    metadata = {
        "id": "wages",
        "title": {"de": "Lohn", "en": "Wages"},
        "notes": {"de": "Lohnstatistik", "en": "Wage statistics"},
        "organization": {"title": "Statistics office"},
        "license_id": "terms_open",
        "resources": [{"id": "csv", "url": "https://example.org/wages.csv", "format": "CSV"}],
    }
    api = client.return_value.__enter__.return_value
    api.call_action.return_value = {"count": 20, "results": [metadata]}

    results = swissdata.search("Lohn", limit=5, offset=10, language="en")

    client.assert_called_once_with("https://opendata.swiss", get_only=True)
    assert api.base_url == "api/3/action/"
    api.call_action.assert_called_once_with("package_search", {"q": "Lohn", "rows": 5, "start": 10}, requests_kwargs={"timeout": 30})
    assert results == [{**metadata, "title": "Wages", "description": "Wage statistics"}]
    assert metadata["title"] == {"de": "Lohn", "en": "Wages"}
    client.return_value.__exit__.assert_called_once()


def test_search_empty_results(client):
    client.return_value.__enter__.return_value.call_action.return_value = {"count": 0, "results": []}
    assert swissdata.search("nonexistent") == []


@pytest.mark.parametrize(("kwargs", "message"), [({"limit": 0}, "limit"), ({"limit": -1}, "limit"), ({"offset": -1}, "offset")])
def test_search_rejects_invalid_pagination(client, kwargs, message):
    with pytest.raises(ValueError, match=message):
        swissdata.search("Lohn", **kwargs)
    client.assert_not_called()


@pytest.mark.parametrize(
    ("title", "notes", "language", "expected_title", "expected_description"),
    [
        ({"de": "Lohn", "en": "Wages"}, {"fr": "Salaires"}, "it", "Lohn", "Salaires"),
        ({"en": "", "fr": "Salaires"}, {"en": "Statistics"}, "en", "Salaires", "Statistics"),
        ("Wages", "Statistics", "en", "Wages", "Statistics"),
        ({"rm": "Datas"}, {}, "de", "Datas", ""),
        (None, None, "de", "", ""),
    ],
)
def test_get_dataset_localizes_metadata(client, title, notes, language, expected_title, expected_description):
    api = client.return_value.__enter__.return_value
    api.call_action.return_value = {"id": "wages", "title": title, "notes": notes}

    dataset = swissdata.get_dataset("wages", language=language)

    api.call_action.assert_called_once_with("package_show", {"id": "wages"}, requests_kwargs={"timeout": 30})
    assert api.base_url == "api/3/action/"
    assert dataset == {"id": "wages", "title": expected_title, "notes": notes, "description": expected_description, "resources": []}


@pytest.mark.parametrize("operation", [swissdata.search, swissdata.get_dataset])
@pytest.mark.parametrize("error", [NotFound("missing"), Timeout("request timed out")])
def test_errors_propagate_and_close_client(client, operation, error):
    client.return_value.__enter__.return_value.call_action.side_effect = error
    client.return_value.__exit__.return_value = False
    with pytest.raises(type(error), match=str(error)):
        operation("missing")
    client.return_value.__exit__.assert_called_once()


def test_library_is_importable_in_execution_namespace():
    result = PythonShell().execute("from swissdata import search, get_dataset\nprint(search.__name__, get_dataset.__name__)")
    assert result["exception"] is None
    assert result["stdout"] == "search get_dataset\n"
