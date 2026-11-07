import { z } from "zod";

export const position = z.tuple([
    z.number().min(-180).max(180),
    z.number().min(-90).max(90),
]);
const ring = z
    .array(position)
    .min(4)
    .max(4000)
    .refine((points) => {
        const first = points[0];
        const last = points.at(-1);
        return first?.[0] === last?.[0] && first?.[1] === last?.[1];
    }, "GeoJSON polygon rings must be closed");
const polygon = z.array(ring).min(1).max(200);
export const polygonGeometry = z.discriminatedUnion("type", [
    z.strictObject({
        type: z.literal("Polygon"),
        coordinates: polygon,
    }),
    z.strictObject({
        type: z.literal("MultiPolygon"),
        coordinates: z.array(polygon).min(1).max(200),
    }),
]);

export function countPositions(
    geometry: z.infer<typeof polygonGeometry>,
): number {
    const polygons =
        geometry.type === "Polygon"
            ? [geometry.coordinates]
            : geometry.coordinates;
    return polygons.reduce(
        (sum, rings) => sum + rings.reduce((n, points) => n + points.length, 0),
        0,
    );
}

export const geojsonSchema = z
    .object({
        type: z.literal("FeatureCollection"),
        features: z
            .array(
                z.object({
                    type: z.literal("Feature"),
                    id: z.string().min(1).max(10000),
                    properties: z
                        .object({ name: z.string().max(10000).default("") })
                        .prefault({}),
                    geometry: polygonGeometry,
                }),
            )
            .min(1)
            .max(200),
    })
    .superRefine((collection, context) => {
        if (
            new Set(collection.features.map((feature) => feature.id)).size !==
            collection.features.length
        )
            context.addIssue({
                code: "custom",
                message: "GeoJSON feature IDs must be unique",
            });
        const count = collection.features.reduce(
            (total, feature) => total + countPositions(feature.geometry),
            0,
        );
        if (count > 4000)
            context.addIssue({
                code: "custom",
                message: "Inline GeoJSON accepts at most 4,000 positions",
            });
    });

export type InlineGeography = z.infer<typeof geojsonSchema>;
