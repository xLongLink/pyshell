import { z } from "zod";

export const catalogId = "urn:pyshell:catalog:v3";
export const identifier = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const text = z.string().max(10000);
const number = z.number().finite();
const coordinates = {
    name: text.default(""),
    x: z
        .array(z.union([number, text]))
        .min(1)
        .max(200),
    y: z.array(number.nullable()).min(1).max(200),
};
export const traceSchema = z.discriminatedUnion("type", [
    z
        .object({
            type: z.literal("scatter"),
            ...coordinates,
            mode: z
                .enum(["lines", "markers", "lines+markers"])
                .default("lines"),
        })
        .strict(),
    z.object({ type: z.literal("bar"), ...coordinates }).strict(),
    z
        .object({
            type: z.literal("pie"),
            name: text.default(""),
            labels: z.array(text).min(1).max(200),
            values: z.array(number.nonnegative().nullable()).min(1).max(200),
        })
        .strict(),
]);
const title = z.object({ text: text.default("") }).strict();
const axis = z.object({ title: title.default({}) }).strict();
const layout = z
    .object({
        title: title.default({}),
        xaxis: axis.default({}),
        yaxis: axis.default({}),
        showlegend: z.boolean().default(true),
    })
    .strict();

const figureFields = {
    data: z.array(traceSchema).min(1).max(20),
    layout: layout.default({}),
};

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
    .object(figureFields)
    .strict()
    .superRefine(validateFigure);
export const componentSchema = z
    .object({
        id: z.literal("root"),
        component: z.literal("Figure"),
        ...figureFields,
    })
    .strict()
    .superRefine(validateFigure);

const components = z.array(componentSchema).length(1);

export const a2uiMessageSchema = z.union([
    z
        .object({
            version: z.literal("v0.9"),
            createSurface: z
                .object({
                    surfaceId: identifier,
                    catalogId: z.literal(catalogId),
                })
                .strict(),
        })
        .strict(),
    z
        .object({
            version: z.literal("v0.9"),
            updateComponents: z
                .object({ surfaceId: identifier, components })
                .strict(),
        })
        .strict(),
    z
        .object({
            version: z.literal("v0.9"),
            deleteSurface: z.object({ surfaceId: identifier }).strict(),
        })
        .strict(),
]);

export type A2uiMessage = z.infer<typeof a2uiMessageSchema>;

export const a2uiBatchSchema = z.preprocess((value, context) => {
    const pending: [unknown, number][] = [[value, 0]];
    let count = 0;
    while (pending.length) {
        const [item, depth] = pending.pop()!;
        if (
            ++count > 20000 ||
            depth > 24 ||
            (typeof item === "string" && item.length > 10000) ||
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
        value.length > 100 ||
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
}, z.array(a2uiMessageSchema).max(100));
