import { useMemo } from "react";
import { Figure } from "@/components/plot/figure";
import type { FigureProps } from "@/components/plot/common";
import type { A2uiMessage } from "@/lib/a2ui";
import type { Placements } from "@/lib/workspace";
import { cn } from "cn";

type Surface = { id: string; props?: FigureProps };

export function Results({
    messages,
    placements = {},
}: {
    messages: A2uiMessage[];
    placements?: Placements;
}) {
    const { surfaces, error } = useMemo(() => {
        const active = new Map<string, Surface>();
        try {
            for (const message of messages) {
                if ("createSurface" in message) {
                    if (
                        active.has(message.createSurface.surfaceId) ||
                        active.size >= 20
                    )
                        throw new Error("Invalid surface creation");
                    const id = message.createSurface.surfaceId;
                    active.set(id, { id });
                } else if ("deleteSurface" in message) {
                    if (!active.has(message.deleteSurface.surfaceId))
                        throw new Error("Unknown surface");
                    active.delete(message.deleteSurface.surfaceId);
                } else {
                    const id = message.updateComponents.surfaceId;
                    if (!active.has(id)) throw new Error("Unknown surface");
                    const { data, layout } =
                        message.updateComponents.components[0];
                    active.set(id, { id, props: { data, layout } });
                }
            }
            return { surfaces: Array.from(active.values()), error: "" };
        } catch {
            return {
                surfaces: [],
                error: "The generated interface could not be rendered. Check the server console for execution details.",
            };
        }
    }, [messages]);

    if (!surfaces.length && !error) return null;
    const centered = surfaces
        .filter((surface) => !placements[surface.id]?.pinned)
        .slice(-4);
    const centerIds = new Set(centered.map((surface) => surface.id));
    const sidebar = surfaces.filter((surface) => !centerIds.has(surface.id));

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
                        "h-80 [&_.recharts-legend-wrapper]:hidden [&_.plot-xaxis-title]:hidden",
                )}
            >
                {surface.props && <Figure {...surface.props} />}
            </article>
        );
    }

    return (
        <section aria-label="Results" className="space-y-4">
            {error && (
                <p role="alert" className="text-sm text-destructive">
                    {error}
                </p>
            )}
            <div
                className={cn(
                    "grid items-start gap-6",
                    sidebar.length > 0 && "lg:grid-cols-[minmax(0,1fr)_20rem]",
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
                        className="min-w-0 space-y-4 lg:max-h-[calc(100dvh-4rem)] lg:overflow-y-auto"
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
