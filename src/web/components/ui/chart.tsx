import * as React from "react";
import { cn } from "cn";
import * as RechartsPrimitive from "recharts";
import { plotClassName } from "@/components/plot/common";

const INITIAL_DIMENSION = { width: 320, height: 200 } as const;

function ChartContainer({
    children,
}: {
    children: React.ComponentProps<
        typeof RechartsPrimitive.ResponsiveContainer
    >["children"];
}) {
    return (
        <div
            data-slot="chart"
            className={cn(
                "flex aspect-video justify-center text-xs [&_.recharts-cartesian-axis-tick_text]:fill-muted-foreground [&_.recharts-cartesian-grid_line[stroke='#ccc']]:stroke-border/50 [&_.recharts-curve.recharts-tooltip-cursor]:stroke-border [&_.recharts-dot[stroke='#fff']]:stroke-transparent [&_.recharts-layer]:outline-hidden [&_.recharts-rectangle.recharts-tooltip-cursor]:fill-muted [&_.recharts-sector]:outline-hidden [&_.recharts-sector[stroke='#fff']]:stroke-transparent [&_.recharts-surface]:outline-hidden",
                plotClassName,
            )}
        >
            <RechartsPrimitive.ResponsiveContainer
                initialDimension={INITIAL_DIMENSION}
            >
                {children}
            </RechartsPrimitive.ResponsiveContainer>
        </div>
    );
}

function ChartTooltipContent({
    active,
    payload,
    formatLabel,
}: {
    active?: boolean;
    payload?: readonly RechartsPrimitive.TooltipPayloadEntry[];
    formatLabel?: (
        payload: readonly RechartsPrimitive.TooltipPayloadEntry[],
    ) => React.ReactNode;
}) {
    if (!active || !payload?.length) {
        return null;
    }

    const tooltipLabel = formatLabel ? (
        <div className="font-medium">{formatLabel(payload)}</div>
    ) : null;

    return (
        <div className="grid min-w-[8rem] items-start gap-1.5 rounded-lg border border-border/50 bg-card px-2.5 py-1.5 text-xs shadow-xl">
            {tooltipLabel}
            <div className="grid gap-1.5">
                {payload
                    .filter((item) => item.type !== "none")
                    .map((item, index) => {
                        const indicatorColor = item.payload?.fill ?? item.color;

                        return (
                            <div
                                key={index}
                                className="flex w-full flex-wrap items-center gap-2"
                            >
                                <div
                                    className="h-2.5 w-2.5 shrink-0 rounded-[2px]"
                                    style={{ backgroundColor: indicatorColor }}
                                />
                                <div className="flex flex-1 items-center justify-between leading-none">
                                    <span className="text-muted-foreground">
                                        {item.name}
                                    </span>
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

function ChartLegendContent({
    payload,
}: {
    payload?: RechartsPrimitive.DefaultLegendContentProps["payload"];
}) {
    if (!payload?.length) {
        return null;
    }

    return (
        <div className="flex max-h-16 flex-wrap items-center justify-center gap-x-4 gap-y-2 overflow-hidden pt-4">
            {payload
                .filter((item) => item.type !== "none")
                .map((item, index) => (
                    <div
                        key={index}
                        className="flex h-5 min-w-0 max-w-full items-center gap-1.5"
                    >
                        <div
                            className="h-2 w-2 shrink-0 rounded-[2px]"
                            style={{
                                backgroundColor: item.color,
                            }}
                        />
                        <span className="truncate">{item.value}</span>
                    </div>
                ))}
        </div>
    );
}

export { ChartContainer, ChartTooltipContent, ChartLegendContent };
