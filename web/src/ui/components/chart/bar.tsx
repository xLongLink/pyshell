import {
  Bar,
  BarChart as RechartsBarChart,
  CartesianGrid,
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

export function BarChart({ title, points }: ChartProps) {
  return (
    <Widget title={title || "Bar chart"} points={points}>
      <ChartContainer
        config={valueConfig}
        className="aspect-auto h-72 w-full focus-within:ring-2 focus-within:ring-ring"
      >
        <RechartsBarChart data={points} accessibilityLayer>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="label" tickLine={false} axisLine={false} />
          <YAxis tickLine={false} axisLine={false} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Bar
            dataKey="value"
            fill="var(--color-value)"
            isAnimationActive={false}
          />
        </RechartsBarChart>
      </ChartContainer>
    </Widget>
  );
}
