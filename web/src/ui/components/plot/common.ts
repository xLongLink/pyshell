import type { z } from "zod";
import type { figureProps, traceSchema } from "@/lib/a2ui";

export type FigureProps = z.infer<typeof figureProps>;
export type Trace = z.infer<typeof traceSchema>;
export const plotClassName =
    "aspect-auto h-72 w-full focus-within:ring-2 focus-within:ring-ring";
export const traceColor = (index: number) => `var(--chart-${(index % 5) + 1})`;
