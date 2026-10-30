import type { Geography } from "../components/plot/geography";

const entries: Record<
    string,
    { name: string; locations: Record<string, string> }
> = {
    world: {
        name: "World",
        locations: { FRA: "France", DEU: "Germany", ESP: "Spain" },
    },
    FRA: {
        name: "France (metropolitan)",
        locations: {
            "FR-IDF": "Île-de-France",
            "FR-BRE": "Bretagne",
            "FR-COR": "Corse",
        },
    },
    USA: {
        name: "United States of America",
        locations: { "US-CA": "California", "US-NY": "New York" },
    },
    CHE: {
        name: "Switzerland",
        locations: { "CH-ZH": "Zürich", "CH-BE": "Bern" },
    },
    DEU: { name: "Germany", locations: { "DE-BY": "Bayern" } },
    GBR: { name: "United Kingdom", locations: { "GB-ENG": "England" } },
    JPN: { name: "Japan", locations: { "JP-27": "Osaka Prefecture" } },
    AUS: { name: "Australia", locations: { "AU-NSW": "New South Wales" } },
    FJI: { name: "Fiji", locations: { "FJ-N": "Northern", "FJ-E": "Eastern" } },
};

export function dataset(geography: string): Geography {
    const entry = entries[geography];
    return {
        type: "FeatureCollection",
        geography,
        name: entry.name,
        level: geography === "world" ? "country" : "region",
        source: {
            source:
                geography === "FRA"
                    ? "IGN / geoBoundaries test fixture"
                    : "geoBoundaries test fixture",
            url: "https://example.com/boundaries",
            license: "Public domain test geometry",
            licenseUrl: "https://example.com/license",
        },
        features: Object.entries(entry.locations).map(([id, name], index) => {
            const x =
                geography === "FJI" ? (index === 0 ? 178 : -180) : index * 3;
            return {
                type: "Feature",
                id,
                properties: { name },
                geometry: {
                    type: "Polygon",
                    coordinates: [
                        [
                            [x, 0],
                            [x + 1, 0],
                            [x + 1, 1],
                            [x, 1],
                            [x, 0],
                        ],
                    ],
                },
            };
        }),
    };
}
