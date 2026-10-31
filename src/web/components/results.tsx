import { Figure } from "@/components/plot/figure";
import type { Placements, Surface } from "@/lib/workspace";
import { cn } from "cn";

export function Results({
    surfaces,
    placements,
}: {
    surfaces: Surface[];
    placements: Placements;
}) {
    if (!surfaces.length) return null;
    const centered = surfaces.filter(
        (surface) => !placements[surface.id].pinned,
    );
    const sidebar = surfaces.filter((surface) => placements[surface.id].pinned);

    function widget(surface: Surface, index: number, center = false) {
        return (
            <article
                key={surface.id}
                aria-label={`Widget ${surface.id}`}
                className={cn(
                    "flex min-h-0 min-w-0 flex-col [&>*]:min-h-0 [&>*]:flex-1",
                    center &&
                        centered.length === 3 &&
                        index === 2 &&
                        "md:col-span-2",
                    !center &&
                        "overflow-hidden [&_.recharts-legend-wrapper]:hidden [&_.plot-xaxis-title]:hidden",
                )}
            >
                <Figure {...surface.props} />
            </article>
        );
    }

    return (
        <section aria-label="Results" className="space-y-4">
            <div
                className={cn(
                    "grid items-start gap-6",
                    sidebar.length > 0 && "lg:grid-cols-[minmax(0,1fr)_16rem]",
                )}
            >
                <section
                    aria-label="Center widgets"
                    className={cn(
                        "grid min-w-0 auto-rows-[minmax(20rem,1fr)] gap-4 md:h-[calc(100dvh-4rem)] md:auto-rows-fr",
                        centered.length === 1 &&
                            "h-[calc(100dvh-2rem)] min-h-80 p-6 md:min-h-80 md:p-12",
                        centered.length > 1 &&
                            "md:min-h-[40rem] md:grid-rows-2",
                        centered.length >= 3 && "md:grid-cols-2",
                    )}
                >
                    {centered.map((surface, index) =>
                        widget(surface, index, true),
                    )}
                </section>
                {sidebar.length > 0 && (
                    <aside
                        aria-label="Side widgets"
                        className="grid h-[calc(100dvh-2rem)] min-w-0 grid-rows-4 gap-3 overflow-hidden lg:h-[calc(100dvh-4rem)]"
                    >
                        {sidebar.map((surface, index) =>
                            widget(surface, index),
                        )}
                    </aside>
                )}
            </div>
        </section>
    );
}
