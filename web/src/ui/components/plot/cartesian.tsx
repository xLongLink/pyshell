import {
    Bar,
    CartesianGrid,
    ComposedChart,
    Line,
    XAxis,
    YAxis,
    useChartWidth,
} from "recharts";
import {
    ChartContainer,
    ChartLegend,
    ChartLegendContent,
    ChartTooltip,
    ChartTooltipContent,
    type ChartConfig,
} from "@/components/ui/chart";
import {
    plotClassName,
    traceColor,
    type FigureProps,
    type Trace,
} from "./common";

type CartesianTrace = Exclude<Trace, { type: "pie" }>;

function ValueAxis({
    domain,
    title,
}: {
    domain: [number, number];
    title: string;
}) {
    const compact = (useChartWidth() ?? 0) < 400;
    return (
        <YAxis
            domain={domain}
            hide={compact}
            width={compact ? 0 : title ? 64 : 48}
            tickLine={false}
            axisLine={false}
            tickMargin={8}
            label={
                compact
                    ? undefined
                    : { value: title, angle: -90, position: "insideLeft" }
            }
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
    const config: ChartConfig = Object.fromEntries(
        data.map((trace, index) => [
            `trace_${index}`,
            {
                label: trace.name || `Series ${index + 1}`,
                color: traceColor(index),
            },
        ]),
    );
    const numeric = typeof data[0].x[0] === "number";
    const categories = [
        ...new Set(
            data
                .flatMap((trace) => trace.x)
                .filter((x): x is string => typeof x === "string"),
        ),
    ];
    const coordinate = (x: string | number) =>
        typeof x === "number" ? x : categories.indexOf(x);
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
            const group = rows.get(x) ?? [];
            if (!group[occurrence]) {
                group[occurrence] = { x: coordinate(x) };
                bars.push(group[occurrence]);
            }
            group[occurrence][`trace_${index}`] = trace.y[point];
            rows.set(x, group);
        });
    });
    return (
        <ChartContainer config={config} className={plotClassName}>
            <ComposedChart
                data={bars}
                accessibilityLayer
                margin={{
                    top: 8,
                    right: 12,
                    bottom: layout.xaxis.title.text ? 24 : 0,
                    left: layout.yaxis.title.text ? 12 : 0,
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
                <ValueAxis
                    domain={[Math.min(0, ...ys), Math.max(0, ...ys)]}
                    title={layout.yaxis.title.text}
                />
                <ChartTooltip
                    cursor={
                        hasBars
                            ? { fill: "var(--accent)", fillOpacity: 0.5 }
                            : false
                    }
                    content={
                        <ChartTooltipContent
                            indicator={hasBars ? "dot" : "line"}
                            labelFormatter={(_, payload) =>
                                numeric
                                    ? String(payload[0]?.payload?.x ?? "")
                                    : (categories[payload[0]?.payload?.x] ?? "")
                            }
                        />
                    }
                />
                {layout.showlegend && (
                    <ChartLegend
                        content={
                            <ChartLegendContent className="flex-wrap gap-x-4 gap-y-2 pt-4" />
                        }
                    />
                )}
                {data.map((trace, index) => {
                    const key = `trace_${index}`;
                    const color = `var(--color-${key})`;
                    return trace.type === "bar" ? (
                        <Bar
                            key={key}
                            dataKey={key}
                            name={key}
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
                            name={key}
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
                            legendType={
                                trace.mode === "markers" ? "circle" : "line"
                            }
                            connectNulls={false}
                            isAnimationActive={false}
                        />
                    );
                })}
            </ComposedChart>
        </ChartContainer>
    );
}
