import { Legend, Pie, PieChart, Tooltip } from "recharts";
import {
    ChartContainer,
    ChartTooltipContent,
    ChartLegendContent,
} from "@/components/ui/chart";
import { plotClassName, traceColor, type Trace } from "./common";

export function PiePlot({
    trace,
    showlegend,
}: {
    trace: Extract<Trace, { type: "pie" }>;
    showlegend: boolean;
}) {
    const slices = trace.labels.flatMap((label, index) => {
        const value = trace.values[index];
        return value === null
            ? []
            : [
                  {
                      label,
                      value,
                      fill: traceColor(index),
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
        <ChartContainer className={plotClassName}>
            <PieChart accessibilityLayer>
                <Pie
                    data={slices}
                    dataKey="value"
                    nameKey="label"
                    stroke="var(--card)"
                    strokeWidth={2}
                    isAnimationActive={false}
                />
                <Tooltip cursor={false} content={<ChartTooltipContent />} />
                {showlegend && <Legend content={<ChartLegendContent />} />}
            </PieChart>
        </ChartContainer>
    );
}
