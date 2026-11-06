import { z } from "zod";
import { polygon, position } from "./geojson";

const text = z.string().max(10000);
const properties = z
    .record(text, z.union([text, z.number(), z.boolean(), z.null()]))
    .refine((value) => Object.keys(value).length <= 12);

export const mapViewSchema = z
    .strictObject({
        zoom: z.number().min(0).max(22).nullish(),
        bounds: z.tuple([position, position]).nullish(),
        center: position.nullish(),
        country: z
            .string()
            .regex(/^[A-Z]{3}$/)
            .nullish(),
        revision: z.number().int().nonnegative().default(0),
    })
    .superRefine((view, context) => {
        if (
            [view.bounds, view.center, view.country].filter(
                (value) => value != null,
            ).length !== 1 ||
            (view.zoom != null && view.center == null) ||
            (view.bounds && view.bounds[0][1] >= view.bounds[1][1])
        )
            context.addIssue({ code: "custom", message: "Invalid map camera" });
    });

export const mapTraceSchema = z
    .strictObject({
        type: z.literal("map"),
        view: mapViewSchema.nullish(),
        layers: z
            .array(
                z.strictObject({
                    id: z.string().regex(/^[A-Za-z0-9_-]{1,64}$/),
                    name: text.default(""),
                    color: z
                        .string()
                        .regex(/^#[0-9a-fA-F]{6}$/)
                        .default("#2563eb"),
                    popup_fields: z.array(text).max(12).default([]),
                    data: z.strictObject({
                        type: z.literal("FeatureCollection"),
                        features: z
                            .array(
                                z.strictObject({
                                    type: z.literal("Feature"),
                                    id: text.min(1),
                                    properties: properties.default({}),
                                    geometry: z.discriminatedUnion("type", [
                                        z.strictObject({
                                            type: z.literal("Point"),
                                            coordinates: position,
                                        }),
                                        z.strictObject({
                                            type: z.literal("Polygon"),
                                            coordinates: polygon,
                                        }),
                                        z.strictObject({
                                            type: z.literal("MultiPolygon"),
                                            coordinates: z
                                                .array(polygon)
                                                .min(1)
                                                .max(200),
                                        }),
                                    ]),
                                }),
                            )
                            .max(200)
                            .default([]),
                    }),
                }),
            )
            .max(8)
            .default([]),
    })
    .superRefine((trace, context) => {
        const fail = (message: string) =>
            context.addIssue({ code: "custom", message });
        if (
            new Set(trace.layers.map((layer) => layer.id)).size !==
            trace.layers.length
        )
            fail("Map layer IDs must be unique");
        let features = 0;
        let positions = 0;
        for (const layer of trace.layers) {
            features += layer.data.features.length;
            if (
                new Set(layer.data.features.map((feature) => feature.id))
                    .size !== layer.data.features.length
            )
                fail("Map feature IDs must be unique within each layer");
            for (const feature of layer.data.features) {
                const geometry = feature.geometry;
                if (geometry.type === "Point") positions += 1;
                else {
                    const polygons =
                        geometry.type === "Polygon"
                            ? [geometry.coordinates]
                            : geometry.coordinates;
                    positions += polygons.reduce(
                        (sum, rings) =>
                            sum + rings.reduce((n, ring) => n + ring.length, 0),
                        0,
                    );
                }
            }
        }
        if (features > 200) fail("A map accepts at most 200 total features");
        if (positions > 4000) fail("A map accepts at most 4,000 positions");
    });

export type MapTrace = z.infer<typeof mapTraceSchema>;
