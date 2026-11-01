import { z } from "zod";
import { geojsonSchema } from "./geojson";

export const identifier = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const text = z.string().max(10000);
const number = z.number();
const coordinates = {
    name: text.default(""),
    x: z
        .array(z.union([number, text]))
        .min(1)
        .max(200),
    y: z.array(number.nullable()).min(1).max(200),
};
export const traceSchema = z.discriminatedUnion("type", [
    z.strictObject({
        type: z.literal("scatter"),
        ...coordinates,
        mode: z.enum(["lines", "markers", "lines+markers"]).default("lines"),
    }),
    z.strictObject({ type: z.literal("bar"), ...coordinates }),
    z.strictObject({
        type: z.literal("pie"),
        name: text.default(""),
        labels: z.array(text).min(1).max(200),
        values: z.array(number.nonnegative().nullable()).min(1).max(200),
    }),
    z.strictObject({
        type: z.literal("choropleth"),
        name: text.default(""),
        geography: z
            .string()
            .regex(/^(world|[A-Z]{3})$/)
            .default("world"),
        geojson: geojsonSchema.nullish(),
        locations: z.array(text).min(1).max(200),
        values: z.array(number.nullable()).min(1).max(200),
    }),
]);
const title = z.strictObject({ text: text.default("") });
const axis = z.strictObject({ title: title.prefault({}) });
const layout = z.strictObject({
    title: title.prefault({}),
    xaxis: axis.prefault({}),
    showlegend: z.boolean().default(true),
});

function validateFigure(
    figure: { data: z.infer<typeof traceSchema>[] },
    context: z.RefinementCtx,
) {
    const fail = (message: string) =>
        context.addIssue({ code: "custom", message });
    let count = 0;
    const coordinateTypes = new Set<string>();
    for (const trace of figure.data) {
        if (trace.type === "pie") {
            if (figure.data.length !== 1)
                fail("A pie figure requires exactly one pie trace");
            if (trace.labels.length !== trace.values.length)
                fail("labels and values must have equal lengths");
            count += trace.values.length;
        } else if (trace.type === "choropleth") {
            if (figure.data.length !== 1)
                fail("A map figure requires exactly one choropleth trace");
            if (trace.locations.length !== trace.values.length)
                fail("locations and values must have equal lengths");
            if (new Set(trace.locations).size !== trace.locations.length)
                fail("locations must be unique");
            if (trace.geojson) {
                const ids = new Set(
                    trace.geojson.features.map((feature) => feature.id),
                );
                if (trace.locations.some((location) => !ids.has(location)))
                    fail("Unknown location for the supplied GeoJSON");
            }
            count += trace.values.length;
        } else {
            if (trace.x.length !== trace.y.length)
                fail("x and y must have equal lengths");
            trace.x.forEach((value) => coordinateTypes.add(typeof value));
            count += trace.x.length;
        }
    }
    if (coordinateTypes.size > 1)
        fail("All x coordinates must be numeric or all categorical");
    if (count > 200) fail("A figure accepts at most 200 total points");
}

export const figureProps = z
    .strictObject({
        data: z.array(traceSchema).min(1).max(20),
        layout: layout.prefault({}),
    })
    .superRefine(validateFigure);
export const componentSchema = figureProps.safeExtend({
    id: z.literal("root"),
    component: z.literal("Figure"),
});

const createSurfaceSchema = z.strictObject({
    version: z.literal("v0.9"),
    createSurface: z.strictObject({
        surfaceId: identifier,
    }),
});
const updateComponentsSchema = z.strictObject({
    version: z.literal("v0.9"),
    updateComponents: z.strictObject({
        surfaceId: identifier,
        components: z.array(componentSchema).length(1),
    }),
});

const surfaceBatchSchema = z
    .tuple([createSurfaceSchema, updateComponentsSchema])
    .superRefine(([creation, update], context) => {
        if (
            update.updateComponents.surfaceId !==
            creation.createSurface.surfaceId
        )
            context.addIssue({
                code: "custom",
                message: "Surface update must match its creation",
            });
    });

export const a2uiBatchSchema = z.preprocess((value, context) => {
    const pending: [unknown, number][] = [[value, 0]];
    let count = 0;
    while (pending.length) {
        const [item, depth] = pending.pop()!;
        if (
            ++count > 20000 ||
            depth > 24 ||
            (Array.isArray(item) && item.length > 500) ||
            (item !== null &&
                typeof item === "object" &&
                !Array.isArray(item) &&
                Object.keys(item).length > 16)
        ) {
            context.addIssue({
                code: "custom",
                message: "UI payload exceeds the display limits",
                fatal: true,
            });
            return z.NEVER;
        }
        if (item !== null && typeof item === "object")
            pending.push(
                ...Object.values(item).map((child): [unknown, number] => [
                    child,
                    depth + 1,
                ]),
            );
    }
    if (
        !Array.isArray(value) ||
        new TextEncoder().encode(JSON.stringify(value)).length > 1048576
    ) {
        context.addIssue({
            code: "custom",
            message: "UI batch exceeds the display limits",
            fatal: true,
        });
        return z.NEVER;
    }
    return value;
}, surfaceBatchSchema);
