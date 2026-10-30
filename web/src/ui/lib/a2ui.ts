import { z } from "zod";

export const catalogId = "urn:pyshell:catalog:v2";
const identifier = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
const text = z.string().max(10000);
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
  chartProps.extend({
    id: z.literal("root"),
    component: z.literal("BarChart"),
  }),
  chartProps.extend({
    id: z.literal("root"),
    component: z.literal("LineChart"),
  }),
  pieChartProps.extend({
    id: z.literal("root"),
    component: z.literal("PieChart"),
  }),
]);

const components = z.array(componentSchema).length(1);

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
