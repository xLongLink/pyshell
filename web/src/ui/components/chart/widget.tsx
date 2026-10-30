import type { ReactNode } from "react";
import type { z } from "zod";
import type { chartProps } from "@/lib/a2ui";
import type { ChartConfig } from "@/components/ui/chart";

export type ChartProps = z.infer<typeof chartProps>;

export const valueConfig = {
  value: { label: "Value", color: "var(--chart-1)" },
} satisfies ChartConfig;

export function Widget({
  title,
  children,
}: ChartProps & { children: ReactNode }) {
  return (
    <figure className="min-w-0 rounded-lg border bg-card p-5">
      <figcaption className="mb-4 font-semibold">{title}</figcaption>
      {children}
    </figure>
  );
}
