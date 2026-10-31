import {
    Bar,
    CartesianGrid,
    ComposedChart,
    Legend,
    Line,
    Tooltip,
    XAxis,
    YAxis,
} from "recharts";
import {
    ChartContainer,
    ChartLegendContent,
    ChartTooltipContent,
} from "@/components/ui/chart";
import { traceColor, type FigureProps, type Trace } from "./common";

type CartesianTrace = Extract<Trace, { type: "bar" | "scatter" }>;

export function Cartesian({
    data,
    layout,
}: {
    data: CartesianTrace[];
    layout: FigureProps["layout"];
}) {
    const numeric = typeof data[0].x[0] === "number";
    const categories = numeric
        ? []
        : [
              ...new Set(
                  data
                      .flatMap((trace) => trace.x)
                      .filter((x): x is string => typeof x === "string"),
              ),
          ];
    const categoryIndices = new Map(
        categories.map((category, index) => [category, index]),
    );
    const coordinate = (x: string | number) =>
        typeof x === "number" ? x : categoryIndices.get(x)!;
    const xs = [
        ...new Set(data.flatMap((trace) => trace.x.map(coordinate))),
    ].sort((a, b) => a - b);
    const hasBars = data.some((trace) => trace.type === "bar");
    const padding =
        xs.length === 1
            ? 0.5
            : hasBars
              ? Math.min(...xs.slice(1).map((x, index) => x - xs[index])) / 2
              : 0;
    const ys = data.flatMap((trace) =>
        trace.y.filter((y): y is number => y !== null),
    );
    if (ys.length === 0) {
        return (
            <p className="py-8 text-sm text-muted-foreground">
                No values to display.
            </p>
        );
    }
    const bars: Record<string, string | number | null>[] = [];
    const rows = new Map<
        string | number,
        Record<string, string | number | null>[]
    >();
    data.forEach((trace, index) => {
        if (trace.type !== "bar") return;
        const occurrences = new Map<string | number, number>();
        trace.x.forEach((x, point) => {
            const occurrence = occurrences.get(x) ?? 0;
            occurrences.set(x, occurrence + 1);
            let group = rows.get(x);
            if (!group) {
                group = [];
                rows.set(x, group);
            }
            if (!group[occurrence]) {
                group[occurrence] = { x: coordinate(x) };
                bars.push(group[occurrence]);
            }
            group[occurrence][`trace_${index}`] = trace.y[point];
        });
    });
    return (
        <ChartContainer>
            <ComposedChart
                data={bars}
                accessibilityLayer
                margin={{
                    top: 8,
                    right: 12,
                    bottom: layout.xaxis.title.text ? 24 : 0,
                    left: 0,
                }}
            >
                <CartesianGrid vertical={false} />
                <XAxis
                    dataKey="x"
                    type="number"
                    domain={[xs[0] - padding, xs[xs.length - 1] + padding]}
                    ticks={
                        numeric
                            ? undefined
                            : categories.map((_, index) => index)
                    }
                    tickFormatter={
                        numeric ? undefined : (value) => categories[value] ?? ""
                    }
                    allowDataOverflow
                    tickLine={false}
                    axisLine={false}
                    tickMargin={8}
                    minTickGap={24}
                    label={{
                        className: "plot-xaxis-title",
                        value: layout.xaxis.title.text,
                        position: "bottom",
                    }}
                />
                <YAxis
                    domain={[Math.min(0, ...ys), Math.max(0, ...ys)]}
                    hide
                    width={0}
                />
                <Tooltip
                    cursor={
                        hasBars
                            ? { fill: "var(--accent)", fillOpacity: 0.5 }
                            : false
                    }
                    content={
                        <ChartTooltipContent
                            formatLabel={(payload) =>
                                numeric
                                    ? String(payload[0]?.payload?.x ?? "")
                                    : (categories[payload[0]?.payload?.x] ?? "")
                            }
                        />
                    }
                />
                {layout.showlegend && (
                    <Legend content={<ChartLegendContent />} />
                )}
                {data.map((trace, index) => {
                    const key = `trace_${index}`;
                    const name = trace.name || `Series ${index + 1}`;
                    const color = traceColor(index);
                    return trace.type === "bar" ? (
                        <Bar
                            key={key}
                            dataKey={key}
                            name={name}
                            fill={color}
                            radius={4}
                            maxBarSize={48}
                            isAnimationActive={false}
                        />
                    ) : (
                        <Line
                            key={key}
                            data={trace.x.map((x, point) => ({
                                x: coordinate(x),
                                [key]: trace.y[point],
                            }))}
                            dataKey={key}
                            name={name}
                            type="linear"
                            stroke={color}
                            strokeWidth={trace.mode === "markers" ? 0 : 2}
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            dot={
                                trace.mode === "lines"
                                    ? false
                                    : { r: 4, fill: color, stroke: color }
                            }
                            activeDot={{
                                r: 6,
                                fill: color,
                                stroke: "var(--card)",
                                strokeWidth: 2,
                            }}
                            connectNulls={false}
                            isAnimationActive={false}
                        />
                    );
                })}
            </ComposedChart>
        </ChartContainer>
    );
}
