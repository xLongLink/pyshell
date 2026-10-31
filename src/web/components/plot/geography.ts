import { geoArea, geoCentroid, geoEqualEarth, geoMercator } from "d3-geo";
import type { FeatureCollection } from "geojson";
import { z } from "zod";

const position = z.tuple([z.number().finite(), z.number().finite()]);
const polygon = z.array(z.array(position));
const geographySchema = z.object({
    type: z.literal("FeatureCollection"),
    geography: z.string(),
    name: z.string(),
    level: z.enum(["country", "region"]),
    source: z.object({
        source: z.string(),
        url: z.url(),
        license: z.string(),
        licenseUrl: z.url(),
        year: z.string().optional(),
    }),
    features: z
        .array(
            z.object({
                type: z.literal("Feature"),
                id: z.string(),
                properties: z.object({ name: z.string() }),
                geometry: z.discriminatedUnion("type", [
                    z.object({
                        type: z.literal("Polygon"),
                        coordinates: polygon,
                    }),
                    z.object({
                        type: z.literal("MultiPolygon"),
                        coordinates: z.array(polygon),
                    }),
                ]),
            }),
        )
        .min(1),
});
export type Geography = z.infer<typeof geographySchema>;
const requests = new Map<string, Promise<Geography>>();

export function loadGeography(geography: string): Promise<Geography> {
    const existing = requests.get(geography);
    if (existing) return existing;
    const request = fetch(`/geographies/${encodeURIComponent(geography)}`)
        .then(async (response) => {
            if (!response.ok) throw new Error("Map data is unavailable");
            const data = geographySchema.parse(await response.json());
            if (data.geography !== geography)
                throw new Error("Unexpected geography");
            for (const feature of data.features) {
                if (geoArea(feature) > 2 * Math.PI) {
                    const polygons =
                        feature.geometry.type === "Polygon"
                            ? [feature.geometry.coordinates]
                            : feature.geometry.coordinates;
                    for (const polygon of polygons)
                        for (const ring of polygon) ring.reverse();
                }
            }
            return data;
        })
        .catch((error) => {
            requests.delete(geography);
            throw error;
        });
    requests.set(geography, request);
    return request;
}

export function mapProjection(boundaries: FeatureCollection, world: boolean) {
    const projection = world
        ? geoEqualEarth()
        : geoMercator().rotate([-geoCentroid(boundaries)[0], 0]);
    return projection.fitExtent(
        [
            [12, 12],
            [788, 438],
        ],
        boundaries,
    );
}
