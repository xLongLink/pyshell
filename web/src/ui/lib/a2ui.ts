import { z } from "zod";

export const catalogId = "urn:pyshell:catalog:v1";
const identifier = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const text = z.string().max(10000);
const cell = z.union([text, z.number().finite(), z.boolean(), z.null()]);

export const textProps = z.object({ text, title: text.default("") }).strict();
export const layoutProps = z
  .object({
    children: z
      .array(identifier)
      .min(1)
      .max(50)
      .describe("REF:#/$defs/ChildList"),
  })
  .strict();
export const metricProps = z
  .object({ label: text, value: text, detail: text.default("") })
  .strict();
export const tableProps = z
  .object({
    title: text.default(""),
    columns: z.array(text).min(1).max(20),
    rows: z.array(z.array(cell).max(20)).max(500),
  })
  .strict();
export const chartProps = z
  .object({
    title: text.default(""),
    points: z
      .array(
        z
          .object({ label: text, value: z.number().finite().nullable() })
          .strict(),
      )
      .min(1)
      .max(200),
  })
  .strict();

export const pieChartProps = chartProps.extend({
  points: z
    .array(
      z
        .object({
          label: text,
          value: z.number().finite().nonnegative().nullable(),
        })
        .strict(),
    )
    .min(1)
    .max(200),
});

export const componentSchema = z.discriminatedUnion("component", [
  textProps.extend({ id: identifier, component: z.literal("Text") }),
  layoutProps.extend({ id: identifier, component: z.literal("Column") }),
  layoutProps.extend({ id: identifier, component: z.literal("Row") }),
  layoutProps.extend({ id: identifier, component: z.literal("Card") }),
  metricProps.extend({ id: identifier, component: z.literal("Metric") }),
  tableProps.extend({ id: identifier, component: z.literal("Table") }),
  chartProps.extend({ id: identifier, component: z.literal("BarChart") }),
  chartProps.extend({ id: identifier, component: z.literal("LineChart") }),
  pieChartProps.extend({ id: identifier, component: z.literal("PieChart") }),
]);

const components = z
  .array(componentSchema)
  .min(1)
  .max(100)
  .superRefine((nodes, context) => {
    const byId = new Map(nodes.map((node) => [node.id, node]));
    const pending: [string, number][] = [["root", 1]];
    const visited = new Set<string>();
    if (byId.size !== nodes.length)
      context.addIssue({ code: "custom", message: "Duplicate component IDs" });
    while (pending.length) {
      const [id, depth] = pending.pop()!;
      const node = byId.get(id);
      if (!node || visited.has(id) || depth > 16) {
        context.addIssue({ code: "custom", message: "Invalid component tree" });
        return;
      }
      visited.add(id);
      if ("children" in node)
        pending.push(
          ...node.children.map((child): [string, number] => [child, depth + 1]),
        );
      if (
        node.component === "Table" &&
        node.rows.some((row) => row.length !== node.columns.length)
      )
        context.addIssue({
          code: "custom",
          message: "Invalid table row width",
        });
    }
    if (visited.size !== nodes.length)
      context.addIssue({ code: "custom", message: "Unreachable components" });
  });

export const a2uiMessageSchema = z.union([
  z
    .object({
      version: z.literal("v0.9"),
      createSurface: z
        .object({ surfaceId: identifier, catalogId: z.literal(catalogId) })
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
