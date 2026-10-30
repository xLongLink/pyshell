from copy import deepcopy
from src.ui.plot import geography


def install_geography_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("PYSHELL_GEOGRAPHY_CACHE", str(tmp_path / "geographies"))
    entries = {
        "world": ("World", {"FRA": "France", "DEU": "Germany", "ESP": "Spain"}),
        "CHE": ("Switzerland", {"CH-ZH": "Zürich", "CH-BE": "Bern"}),
        "FRA": ("France", {"FR-IDF": "Île-de-France", "FR-BRE": "Bretagne", "FR-20R": "Corse"}),
        "USA": ("United States of America", {"US-CA": "California", "US-NY": "New York"}),
        "DEU": ("Germany", {"DE-BY": "Bayern", "DE-BE": "Berlin"}),
        "GBR": ("United Kingdom", {"GB-ENG": "England", "GB-SCT": "Scotland"}),
        "IND": ("India", {"IN-DL": "Delhi"}),
    }
    responses = {}
    for country, (name, locations) in entries.items():
        url = (
            geography.WORLD_URL
            if country == "world"
            else f"https://raw.githubusercontent.com/wmgeolab/geoBoundaries/tests/{country}.geojson"
        )
        responses[url] = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"ISO_A3_EH": code, "NAME_EN": label}
                    if country == "world"
                    else {"shapeGroup": country, "shapeType": "ADM1", "shapeISO": code, "shapeName": label, "shapeID": f"shape-{index}"},
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
                }
                for index, (code, label) in enumerate(locations.items())
            ],
        }
        if country != "world":
            responses[geography.METADATA_URL.format(geography=country)] = {
                "boundaryISO": country,
                "boundaryType": "ADM1",
                "boundaryName": name,
                "boundaryID": f"{country}-ADM1-v1",
                "boundarySource": "Test provider",
                "boundaryLicense": "Public domain",
                "licenseSource": "example.com/license",
                "boundaryYearRepresented": "2025",
                "admUnitCount": str(len(locations)),
                "simplifiedGeometryGeoJSON": url,
            }
    calls = []

    def download(url):
        calls.append(url)
        if url not in responses and url.startswith("https://www.geoboundaries.org/api/current/gbOpen/"):
            return []
        return deepcopy(responses[url])

    monkeypatch.setattr(geography, "download", download)
    return {"responses": responses, "calls": calls, "download": download, "cache": tmp_path / "geographies"}
