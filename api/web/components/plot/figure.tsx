import { lazy, Suspense } from "react";
import type { FigureProps } from "./common";
import { Cartesian } from "./cartesian";
import { PiePlot } from "./pie";

const Choropleth = lazy(() => import("./choropleth"));

export function Figure({ data, layout }: FigureProps) {
    const pie = data[0].type === "pie" ? data[0] : undefined;
    const map = data[0].type === "choropleth" ? data[0] : undefined;
    const cartesian = data.filter(
        (trace) => trace.type === "bar" || trace.type === "scatter",
    );
    return (
        <figure className="flex h-full min-h-0 min-w-0 flex-col rounded-xl border bg-card shadow-sm">
            <figcaption className="shrink-0 p-6 text-base leading-none font-semibold tracking-tight">
                {layout.title.text || "Plot"}
            </figcaption>
            <div className="min-h-0 flex-1 px-6 pb-6">
                {map ? (
                    <Suspense
                        fallback={
                            <p className="py-8 text-sm text-muted-foreground">
                                Loading map…
                            </p>
                        }
                    >
                        <Choropleth
                            key={map.geography}
                            trace={map}
                            showlegend={layout.showlegend}
                        />
                    </Suspense>
                ) : pie ? (
                    <PiePlot trace={pie} showlegend={layout.showlegend} />
                ) : cartesian.some((trace) =>
                      trace.y.some((value) => value !== null),
                  ) ? (
                    <Cartesian data={cartesian} layout={layout} />
                ) : (
                    <p className="py-8 text-sm text-muted-foreground">
                        No values to display.
                    </p>
                )}
            </div>
        </figure>
    );
}
