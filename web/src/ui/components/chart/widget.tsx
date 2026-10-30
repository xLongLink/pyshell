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
  points,
  children,
}: ChartProps & { children: ReactNode }) {
  return (
    <figure className="min-w-0 rounded-lg border bg-card p-5">
      <figcaption className="mb-4 font-semibold">{title}</figcaption>
      {children}
      <details className="mt-4 text-sm">
        <summary className="cursor-pointer text-muted-foreground">
          View chart data
        </summary>
        <table className="mt-2 w-full text-left">
          <caption className="sr-only">{title} data</caption>
          <thead>
            <tr>
              <th scope="col">Label</th>
              <th scope="col">Value</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point, index) => (
              <tr key={index}>
                <th scope="row" className="font-normal">
                  {point.label}
                </th>
                <td>{point.value ?? "Missing"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
