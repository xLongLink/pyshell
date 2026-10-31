import { lazy, Suspense } from "react";
import type { FigureProps, Trace } from "./common";
import { Cartesian } from "./cartesian";
import { PiePlot } from "./pie";

const Choropleth = lazy(() => import("./choropleth"));

export function Figure({ data, layout }: FigureProps) {
    const trace = data[0];
    const cartesian = data as Extract<Trace, { type: "bar" | "scatter" }>[];
    return (
        <figure className="flex h-full min-h-0 min-w-0 flex-col rounded-xl border bg-card shadow-sm">
            <figcaption className="shrink-0 p-6 text-base leading-none font-semibold tracking-tight">
                {layout.title.text || "Plot"}
            </figcaption>
            <div className="min-h-0 flex-1 px-6 pb-6">
                {trace.type === "choropleth" ? (
                    <Suspense
                        fallback={
                            <p className="py-8 text-sm text-muted-foreground">
                                Loading map…
                            </p>
                        }
                    >
                        <Choropleth
                            key={trace.geography}
                            trace={trace}
                            showlegend={layout.showlegend}
                        />
                    </Suspense>
                ) : trace.type === "pie" ? (
                    <PiePlot trace={trace} showlegend={layout.showlegend} />
                ) : (
                    <Cartesian data={cartesian} layout={layout} />
                )}
            </div>
        </figure>
    );
}
