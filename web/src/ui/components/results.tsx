import { useEffect, useState } from "react";
import { A2uiSurface, type ReactCatalogComponent } from "@a2ui/react/v0_9";
import { MessageProcessor, type SurfaceModel } from "@a2ui/web_core/v0_9";
import { resultCatalog } from "@/components/catalog";
import type { A2uiMessage } from "@/lib/a2ui";
import type { Placements } from "@/lib/workspace";
import { cn } from "@/lib/utils";

export function Results({
    messages,
    placements = {},
}: {
    messages: A2uiMessage[];
    placements?: Placements;
}) {
    const [surfaces, setSurfaces] = useState<
        SurfaceModel<ReactCatalogComponent>[]
    >([]);
    const [error, setError] = useState("");

    useEffect(() => {
        const processor = new MessageProcessor<ReactCatalogComponent>([
            resultCatalog,
        ]);
        try {
            const active = new Set<string>();
            for (const message of messages) {
                if ("createSurface" in message) {
                    if (
                        active.has(message.createSurface.surfaceId) ||
                        active.size >= 20
                    )
                        throw new Error("Invalid surface creation");
                    active.add(message.createSurface.surfaceId);
                } else if ("deleteSurface" in message) {
                    if (!active.delete(message.deleteSurface.surfaceId))
                        throw new Error("Unknown surface");
                } else if (!active.has(message.updateComponents.surfaceId))
                    throw new Error("Unknown surface");
                processor.processMessages(message);
            }
            setSurfaces(Array.from(processor.getSurfaces().values()));
            setError("");
        } catch {
            setSurfaces([]);
            setError(
                "The generated interface could not be rendered. Check the server console for execution details.",
            );
        }
        return () => processor.dispose();
    }, [messages]);

    if (!surfaces.length && !error) return null;
    const centered = surfaces
        .filter((surface) => !placements[surface.id]?.pinned)
        .slice(-4);
    const centerIds = new Set(centered.map((surface) => surface.id));
    const sidebar = surfaces.filter((surface) => !centerIds.has(surface.id));

    function widget(
        surface: SurfaceModel<ReactCatalogComponent>,
        index: number,
        center = false,
    ) {
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
                    !center && "h-80",
                )}
            >
                <A2uiSurface surface={surface} />
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
