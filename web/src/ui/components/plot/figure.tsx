import type { FigureProps } from "./common";
import { Cartesian } from "./cartesian";
import { PiePlot } from "./pie";

export function Figure({ data, layout }: FigureProps) {
    const pie = data[0].type === "pie" ? data[0] : undefined;
    const cartesian = data.filter((trace) => trace.type !== "pie");
    return (
        <figure className="min-w-0 rounded-lg border bg-card p-5">
            <figcaption className="mb-4 font-semibold">
                {layout.title.text || "Plot"}
            </figcaption>
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
        </figure>
    );
}
