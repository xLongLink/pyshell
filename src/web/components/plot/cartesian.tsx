import {
    Area,
    Bar,
    CartesianGrid,
    ComposedChart,
    Legend,
    Line,
    Tooltip,
    usePlotArea,
    XAxis,
    YAxis,
} from "recharts";
import { useId } from "react";
import {
    ChartContainer,
    ChartLegendContent,
    ChartTooltipContent,
} from "@/components/ui/chart";
import { traceColor, type FigureProps, type Trace } from "./common";

type CartesianTrace = Extract<Trace, { type: "bar" | "scatter" | "area" }>;

function SizedBar({
    spacing,
    count,
    ...props
}: React.ComponentProps<typeof Bar> & {
    spacing: number;
    count: number;
}) {
    const plot = usePlotArea();
    return (
        <Bar
            {...props}
            barSize={Math.min(48, ((plot?.width ?? 0) * spacing * 0.8) / count)}
        />
    );
}

export function Cartesian({
    data,
    layout,
}: {
    data: CartesianTrace[];
    layout: FigureProps["layout"];
}) {
    const gradientId = useId().replace(/:/g, "");
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
    const barCount = data.filter((trace) => trace.type === "bar").length;
    const hasBars = barCount > 0;
    const spacing =
        xs.length === 1
            ? 1
            : Math.min(...xs.slice(1).map((x, index) => x - xs[index]));
    const padding = xs.length === 1 ? 0.5 : hasBars ? spacing / 2 : 0;
    const domain: [number, number] = [
        xs[0] - padding,
        xs[xs.length - 1] + padding,
    ];
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
    xs.forEach((x) => {
        if (!bars.some((row) => row.x === x)) bars.push({ x });
    });
    bars.sort((a, b) => Number(a.x) - Number(b.x));
    // Interaction ticks must be distinct even when observations share x.
    bars.forEach((row, index) => {
        row.tooltipX =
            index === 0 || row.x !== bars[index - 1].x ? row.x : null;
    });
    return (
        <ChartContainer>
            <ComposedChart
                data={bars}
                barGap={0}
                accessibilityLayer
                margin={{
                    top: 8,
                    right: 12,
                    bottom: layout.xaxis.title.text ? 24 : 0,
                    left: 0,
                }}
            >
                <defs>
                    {data.map((trace, index) =>
                        trace.type === "area" ? (
                            <linearGradient
                                key={index}
                                id={`${gradientId}-area-${index}`}
                                x1="0"
                                y1="0"
                                x2="0"
                                y2="1"
                            >
                                <stop
                                    offset="0%"
                                    stopColor={traceColor(index)}
                                    stopOpacity={0.65}
                                />
                                <stop
                                    offset="100%"
                                    stopColor={traceColor(index)}
                                    stopOpacity={0.12}
                                />
                            </linearGradient>
                        ) : null,
                    )}
                </defs>
                <CartesianGrid vertical={false} />
                <XAxis
                    dataKey="tooltipX"
                    type="number"
                    domain={domain}
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
                <XAxis
                    xAxisId="traces"
                    dataKey="x"
                    type="number"
                    domain={domain}
                    allowDataOverflow
                    hide
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
                    const points =
                        trace.type === "bar"
                            ? undefined
                            : trace.x.map((x, point) => ({
                                  x: coordinate(x),
                                  tooltipX: coordinate(x),
                                  [key]: trace.y[point],
                              }));
                    if (trace.type === "area") {
                        return (
                            <Area
                                key={key}
                                xAxisId="traces"
                                data={points}
                                dataKey={key}
                                name={name}
                                type={trace.curve}
                                stroke={color}
                                strokeWidth={2}
                                fill={`url(#${gradientId}-area-${index})`}
                                fillOpacity={1}
                                baseValue={0}
                                dot={false}
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
                    }
                    return trace.type === "bar" ? (
                        <SizedBar
                            key={key}
                            xAxisId="traces"
                            spacing={
                                spacing /
                                (xs[xs.length - 1] - xs[0] + 2 * padding)
                            }
                            count={barCount}
                            dataKey={key}
                            name={name}
                            fill={color}
                            radius={4}
                            isAnimationActive={false}
                        />
                    ) : (
                        <Line
                            key={key}
                            xAxisId="traces"
                            data={points}
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
