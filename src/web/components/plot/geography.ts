import { geoArea, geoCentroid, geoEqualEarth, geoMercator } from "d3-geo";
import type { FeatureCollection } from "geojson";
import { z } from "zod";

const position = z.tuple([z.number(), z.number()]);
const polygon = z.array(z.array(position));
const geographySchema = z.object({
    type: z.literal("FeatureCollection"),
    geography: z.string(),
    name: z.string(),
    level: z.enum(["country", "region"]),
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

export function loadGeography(geography: string): Promise<Geography> {
    return fetch(`/geographies/${encodeURIComponent(geography)}`).then(
        async (response) => {
            if (!response.ok) throw new Error("Map data is unavailable");
            const data = geographySchema.parse(await response.json());
            if (data.geography !== geography)
                throw new Error("Unexpected geography");
            return data;
        },
    );
}

export function orientBoundaries<T extends FeatureCollection>(
    boundaries: T,
): T {
    const normalized = structuredClone(boundaries);
    for (const feature of normalized.features) {
        if (geoArea(feature) > 2 * Math.PI) {
            const geometry = feature.geometry;
            if (geometry.type !== "Polygon" && geometry.type !== "MultiPolygon")
                continue;
            const polygons =
                geometry.type === "Polygon"
                    ? [geometry.coordinates]
                    : geometry.coordinates;
            for (const polygon of polygons)
                for (const ring of polygon) ring.reverse();
        }
    }
    return normalized;
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
