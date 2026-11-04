import { lazy, Suspense } from "react";
import type { FigureProps, Trace } from "./common";
import { Cartesian } from "./cartesian";
import { PiePlot } from "./pie";
import { cn } from "cn";

const Choropleth = lazy(() => import("./choropleth"));

export function Figure({
    data,
    layout,
    compact = false,
}: FigureProps & { compact?: boolean }) {
    const trace = data[0];
    const cartesian = data as Extract<
        Trace,
        { type: "bar" | "scatter" | "area" }
    >[];
    const showlegend = !compact && layout.showlegend;
    return (
        <figure className="flex h-full min-h-0 min-w-0 flex-col rounded-xl border bg-card shadow-sm">
            <figcaption
                className={cn(
                    "shrink-0 font-semibold tracking-tight",
                    compact
                        ? "px-3 pt-3 pb-1 text-xs leading-tight"
                        : "p-6 text-base leading-none",
                )}
            >
                {layout.title.text || "Plot"}
            </figcaption>
            <div
                className={cn(
                    "min-h-0 flex-1",
                    compact ? "px-3 pb-1" : "px-6 pb-6",
                )}
            >
                {trace.type === "choropleth" ? (
                    <Suspense
                        fallback={
                            <p className="py-8 text-sm text-muted-foreground">
                                Loading map…
                            </p>
                        }
                    >
                        <Choropleth trace={trace} showlegend={showlegend} />
                    </Suspense>
                ) : trace.type === "pie" ? (
                    <PiePlot trace={trace} showlegend={showlegend} />
                ) : (
                    <Cartesian
                        data={cartesian}
                        xAxisTitle={compact ? "" : layout.xaxis.title.text}
                        showlegend={showlegend}
                    />
                )}
            </div>
        </figure>
    );
}
