import * as React from "react";
import { cn } from "cn";
import * as RechartsPrimitive from "recharts";
import type { TooltipPayloadEntry, TooltipValueType } from "recharts";

const INITIAL_DIMENSION = { width: 320, height: 200 } as const;
type TooltipNameType = number | string;

export type ChartConfig = Record<
    string,
    {
        label?: React.ReactNode;
        color?: string;
    }
>;

type ChartContextProps = {
    config: ChartConfig;
};

const ChartContext = React.createContext<ChartContextProps | null>(null);

function useChart() {
    const context = React.useContext(ChartContext);

    if (!context) {
        throw new Error("useChart must be used within a <ChartContainer />");
    }

    return context;
}

function ChartContainer({
    className,
    children,
    config,
    style,
    ...props
}: React.ComponentProps<"div"> & {
    config: ChartConfig;
    children: React.ComponentProps<
        typeof RechartsPrimitive.ResponsiveContainer
    >["children"];
}) {
    const colors = Object.fromEntries(
        Object.entries(config)
            .filter(([, item]) => item.color)
            .map(([key, item]) => [`--color-${key}`, item.color]),
    );

    return (
        <ChartContext.Provider value={{ config }}>
            <div
                data-slot="chart"
                className={cn(
                    "flex aspect-video justify-center text-xs [&_.recharts-cartesian-axis-tick_text]:fill-muted-foreground [&_.recharts-cartesian-grid_line[stroke='#ccc']]:stroke-border/50 [&_.recharts-curve.recharts-tooltip-cursor]:stroke-border [&_.recharts-dot[stroke='#fff']]:stroke-transparent [&_.recharts-layer]:outline-hidden [&_.recharts-polar-grid_[stroke='#ccc']]:stroke-border [&_.recharts-radial-bar-background-sector]:fill-muted [&_.recharts-rectangle.recharts-tooltip-cursor]:fill-muted [&_.recharts-reference-line_[stroke='#ccc']]:stroke-border [&_.recharts-sector]:outline-hidden [&_.recharts-sector[stroke='#fff']]:stroke-transparent [&_.recharts-surface]:outline-hidden",
                    className,
                )}
                {...props}
                style={{ ...colors, ...style } as React.CSSProperties}
            >
                <RechartsPrimitive.ResponsiveContainer
                    initialDimension={INITIAL_DIMENSION}
                >
                    {children}
                </RechartsPrimitive.ResponsiveContainer>
            </div>
        </ChartContext.Provider>
    );
}

const ChartTooltip = RechartsPrimitive.Tooltip;

function ChartTooltipContent({
    active,
    payload,
    className,
    indicator = "dot",
    hideLabel = false,
    label,
    labelFormatter,
    nameKey,
}: Omit<
    React.ComponentProps<typeof RechartsPrimitive.Tooltip>,
    "formatter" | "color" | "labelClassName"
> &
    Omit<React.ComponentProps<"div">, "color"> & {
        hideLabel?: boolean;
        indicator?: "line" | "dot";
        nameKey?: "category";
    } & Omit<
        RechartsPrimitive.DefaultTooltipContentProps<
            TooltipValueType,
            TooltipNameType
        >,
        "accessibilityLayer" | "formatter" | "color" | "labelClassName"
    >) {
    const { config } = useChart();

    const tooltipLabel = React.useMemo(() => {
        if (hideLabel || !payload?.length) {
            return null;
        }

        const [item] = payload;
        const itemConfig = getChartItemConfig(config, item);
        const value =
            typeof label === "string"
                ? (config[label]?.label ?? label)
                : itemConfig?.label;

        if (labelFormatter) {
            return (
                <div className="font-medium">
                    {labelFormatter(value, payload)}
                </div>
            );
        }

        if (!value) {
            return null;
        }

        return <div className="font-medium">{value}</div>;
    }, [label, labelFormatter, payload, hideLabel, config]);

    if (!active || !payload?.length) {
        return null;
    }

    const nestLabel = payload.length === 1 && indicator !== "dot";

    return (
        <div
            className={cn(
                "grid min-w-[8rem] items-start gap-1.5 rounded-lg border border-border/50 bg-card px-2.5 py-1.5 text-xs shadow-xl",
                className,
            )}
        >
            {!nestLabel ? tooltipLabel : null}
            <div className="grid gap-1.5">
                {payload
                    .filter((item) => item.type !== "none")
                    .map((item, index) => {
                        const itemConfig = getChartItemConfig(
                            config,
                            item,
                            nameKey,
                        );
                        const indicatorColor = item.payload?.fill ?? item.color;

                        return (
                            <div
                                key={index}
                                className={cn(
                                    "flex w-full flex-wrap items-stretch gap-2 [&>svg]:h-2.5 [&>svg]:w-2.5 [&>svg]:text-muted-foreground",
                                    indicator === "dot" && "items-center",
                                )}
                            >
                                <div
                                    className={cn(
                                        "shrink-0 rounded-[2px] border-(--color-border) bg-(--color-bg)",
                                        {
                                            "h-2.5 w-2.5": indicator === "dot",
                                            "w-1": indicator === "line",
                                        },
                                    )}
                                    style={
                                        {
                                            "--color-bg": indicatorColor,
                                            "--color-border": indicatorColor,
                                        } as React.CSSProperties
                                    }
                                />
                                <div
                                    className={cn(
                                        "flex flex-1 justify-between leading-none",
                                        nestLabel
                                            ? "items-end"
                                            : "items-center",
                                    )}
                                >
                                    <div className="grid gap-1.5">
                                        {nestLabel ? tooltipLabel : null}
                                        <span className="text-muted-foreground">
                                            {itemConfig?.label ?? item.name}
                                        </span>
                                    </div>
                                    {item.value != null && (
                                        <span className="font-mono font-medium text-foreground tabular-nums">
                                            {typeof item.value === "number"
                                                ? item.value.toLocaleString()
                                                : String(item.value)}
                                        </span>
                                    )}
                                </div>
                            </div>
                        );
                    })}
            </div>
        </div>
    );
}

const ChartLegend = RechartsPrimitive.Legend;

function ChartLegendContent({
    className,
    payload,
    nameKey,
}: React.ComponentProps<"div"> & {
    nameKey?: "category";
} & Omit<RechartsPrimitive.DefaultLegendContentProps, "verticalAlign">) {
    const { config } = useChart();

    if (!payload?.length) {
        return null;
    }

    return (
        <div
            className={cn(
                "flex max-h-16 flex-wrap items-center justify-center gap-x-4 gap-y-2 overflow-hidden pt-4",
                className,
            )}
        >
            {payload
                .filter((item) => item.type !== "none")
                .map((item, index) => {
                    const itemConfig = getChartItemConfig(
                        config,
                        item,
                        nameKey,
                    );

                    return (
                        <div
                            key={index}
                            className={cn(
                                "flex h-5 min-w-0 max-w-full items-center gap-1.5 [&>svg]:h-3 [&>svg]:w-3 [&>svg]:shrink-0 [&>svg]:text-muted-foreground",
                            )}
                        >
                            <div
                                className="h-2 w-2 shrink-0 rounded-[2px]"
                                style={{
                                    backgroundColor: item.color,
                                }}
                            />
                            <span className="truncate">
                                {itemConfig?.label}
                            </span>
                        </div>
                    );
                })}
        </div>
    );
}

function getChartItemConfig(
    config: ChartConfig,
    item: Pick<TooltipPayloadEntry, "dataKey"> & { payload?: unknown },
    nameKey?: "category",
) {
    if (nameKey === "category") {
        const slice = item.payload;
        return typeof slice === "object" &&
            slice !== null &&
            "category" in slice &&
            typeof slice.category === "string"
            ? config[slice.category]
            : undefined;
    }

    return config[typeof item.dataKey === "string" ? item.dataKey : "value"];
}

export {
    ChartContainer,
    ChartTooltip,
    ChartTooltipContent,
    ChartLegend,
    ChartLegendContent,
};
