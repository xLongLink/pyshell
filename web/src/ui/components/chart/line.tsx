import {
  CartesianGrid,
  Line,
  LineChart as RechartsLineChart,
  XAxis,
  YAxis,
} from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Widget,
  valueConfig,
  type ChartProps,
} from "@/components/chart/widget";

export function LineChart({ title, points }: ChartProps) {
  return (
    <Widget title={title || "Line chart"} points={points}>
      <ChartContainer
        config={valueConfig}
        className="aspect-auto h-72 w-full focus-within:ring-2 focus-within:ring-ring"
      >
        <RechartsLineChart data={points} accessibilityLayer>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="label" tickLine={false} axisLine={false} />
          <YAxis tickLine={false} axisLine={false} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Line
            dataKey="value"
            type="linear"
            stroke="var(--color-value)"
            connectNulls={false}
            isAnimationActive={false}
          />
        </RechartsLineChart>
      </ChartContainer>
    </Widget>
  );
}
