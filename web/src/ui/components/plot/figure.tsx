import type { FigureProps } from "./common";
import { Cartesian } from "./cartesian";
import { PiePlot } from "./pie";

export function Figure({ data, layout }: FigureProps) {
    const pie = data[0].type === "pie" ? data[0] : undefined;
    const cartesian = data.filter((trace) => trace.type !== "pie");
    return (
        <figure className="flex h-full min-h-0 min-w-0 flex-col rounded-xl border bg-card shadow-sm">
            <figcaption className="shrink-0 p-6 text-base leading-none font-semibold tracking-tight">
                {layout.title.text || "Plot"}
            </figcaption>
            <div className="min-h-0 flex-1 px-6 pb-6">
                {pie ? (
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
