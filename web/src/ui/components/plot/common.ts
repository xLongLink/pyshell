import type { z } from "zod";
import type { figureProps, traceSchema } from "@/lib/a2ui";

export type FigureProps = z.infer<typeof figureProps>;
export type Trace = z.infer<typeof traceSchema>;
export const plotClassName =
    "aspect-auto h-full w-full rounded-lg has-[:focus-visible]:ring-2 has-[:focus-visible]:ring-ring";
export const traceColor = (index: number) => `var(--chart-${(index % 5) + 1})`;
