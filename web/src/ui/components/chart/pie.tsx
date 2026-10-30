import { Pie, PieChart as RechartsPieChart } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { Widget, type ChartProps } from "@/components/chart/widget";

const colors = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];

export function PieChart({ title, points }: ChartProps) {
  const config: ChartConfig = Object.fromEntries(
    points.map((point, index) => [
      `slice_${index}`,
      { label: point.label, color: colors[index % colors.length] },
    ]),
  );
  const slices = points
    .map((point, index) => ({
      ...point,
      category: `slice_${index}`,
      fill: `var(--color-slice_${index})`,
    }))
    .filter((point) => point.value !== null);

  return (
    <Widget title={title || "Pie chart"} points={points}>
      {slices.some((point) => point.value !== null && point.value > 0) ? (
        <ChartContainer
          config={config}
          className="aspect-auto h-72 w-full focus-within:ring-2 focus-within:ring-ring"
        >
          <RechartsPieChart accessibilityLayer>
            <Pie
              data={slices}
              dataKey="value"
              nameKey="label"
              isAnimationActive={false}
            />
            <ChartTooltip
              content={<ChartTooltipContent nameKey="category" hideLabel />}
            />
            <ChartLegend
              content={
                <ChartLegendContent nameKey="category" className="flex-wrap" />
              }
            />
          </RechartsPieChart>
        </ChartContainer>
      ) : (
        <p className="py-8 text-sm text-muted-foreground">
          No positive values to display.
        </p>
      )}
    </Widget>
  );
}
