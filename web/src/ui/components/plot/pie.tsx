import { Legend, Pie, PieChart, Tooltip } from "recharts";
import {
    ChartContainer,
    ChartTooltipContent,
    ChartLegendContent,
    type ChartConfig,
} from "@/components/ui/chart";
import { plotClassName, traceColor, type Trace } from "./common";

export function PiePlot({
    trace,
    showlegend,
}: {
    trace: Extract<Trace, { type: "pie" }>;
    showlegend: boolean;
}) {
    const config: ChartConfig = Object.fromEntries(
        trace.labels.map((label, index) => [
            `slice_${index}`,
            { label, color: traceColor(index) },
        ]),
    );
    const slices = trace.labels.flatMap((label, index) => {
        const value = trace.values[index];
        return value === null
            ? []
            : [
                  {
                      label,
                      value,
                      category: `slice_${index}`,
                      fill: `var(--color-slice_${index})`,
                  },
              ];
    });
    if (!slices.some((point) => point.value > 0)) {
        return (
            <p className="py-8 text-sm text-muted-foreground">
                No positive values to display.
            </p>
        );
    }
    return (
        <ChartContainer config={config} className={plotClassName}>
            <PieChart accessibilityLayer>
                <Pie
                    data={slices}
                    dataKey="value"
                    nameKey="label"
                    stroke="var(--card)"
                    strokeWidth={2}
                    isAnimationActive={false}
                />
                <Tooltip
                    cursor={false}
                    content={<ChartTooltipContent nameKey="category" />}
                />
                {showlegend && (
                    <Legend
                        content={<ChartLegendContent nameKey="category" />}
                    />
                )}
            </PieChart>
        </ChartContainer>
    );
}
