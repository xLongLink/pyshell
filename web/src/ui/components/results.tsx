import { useEffect, useRef, useState } from "react";
import { A2uiSurface, type ReactCatalogComponent } from "@a2ui/react/v0_9";
import { MessageProcessor, type SurfaceModel } from "@a2ui/web_core/v0_9";
import { resultCatalog } from "@/components/catalog";
import type { A2uiMessage } from "@/lib/a2ui";
import type { Placements, WidgetAction } from "@/lib/workspace";
import { Focus, Pin, PinOff } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

export function Results({
  messages,
  placements = {},
  onAction,
}: {
  messages: A2uiMessage[];
  placements?: Placements;
  onAction?: (surfaceId: string, action: WidgetAction) => void;
}) {
  const [surfaces, setSurfaces] = useState<
    SurfaceModel<ReactCatalogComponent>[]
  >([]);
  const [error, setError] = useState("");
  const elements = useRef(new Map<string, HTMLElement>());
  const focusedElement = useRef<HTMLElement | null>(null);
  const focusedId = Object.entries(placements).find(
    ([, placement]) => placement.focused,
  )?.[0];

  useEffect(() => {
    if (!focusedId) {
      if (document.activeElement === focusedElement.current)
        focusedElement.current?.blur();
      focusedElement.current = null;
      return;
    }
    const element = elements.current.get(focusedId);
    focusedElement.current = element ?? null;
    element?.scrollIntoView?.({ block: "center", behavior: "smooth" });
    element?.focus({ preventScroll: true });
  }, [focusedId, surfaces]);

  useEffect(() => {
    const processor = new MessageProcessor<ReactCatalogComponent>([
      resultCatalog,
    ]);
    try {
      const active = new Set<string>();
      for (const message of messages) {
        if ("createSurface" in message) {
          if (active.has(message.createSurface.surfaceId) || active.size >= 20)
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
        "The generated interface could not be rendered. Execution output is still available below.",
      );
    }
    return () => processor.dispose();
  }, [messages]);

  if (!surfaces.length && !error) return null;
  return (
    <section aria-label="Results" className="space-y-4 border-b pb-6">
      <h2 className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
        Results
      </h2>
      {error && (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
      {[...surfaces]
        .sort(
          (left, right) =>
            Number(placements[right.id]?.pinned ?? false) -
            Number(placements[left.id]?.pinned ?? false),
        )
        .map((surface) => {
          const pinned = placements[surface.id]?.pinned ?? false;
          const focused = placements[surface.id]?.focused ?? false;
          return (
            <article
              key={surface.id}
              aria-label={`Widget ${surface.id}`}
              tabIndex={-1}
              ref={(element) => {
                if (element) elements.current.set(surface.id, element);
                else elements.current.delete(surface.id);
              }}
              className={cn(
                "min-w-0 rounded-lg p-1 outline-none",
                focused && "ring-2 ring-primary ring-offset-2",
              )}
            >
              <header className="mb-2 flex items-center justify-between gap-3">
                <span className="text-xs text-muted-foreground">
                  {surface.id}
                  {pinned ? " · Pinned" : ""}
                </span>
                <div className="flex gap-2">
                  <Button
                    variant="outline"
                    size="icon"
                    aria-label={`${focused ? "Unfocus" : "Focus"} ${surface.id}`}
                    aria-pressed={focused}
                    disabled={!onAction}
                    onClick={() =>
                      onAction?.(surface.id, focused ? "unfocus" : "focus")
                    }
                  >
                    <Focus />
                  </Button>
                  <Button
                    variant="outline"
                    size="icon"
                    aria-label={`${pinned ? "Unpin" : "Pin"} ${surface.id}`}
                    aria-pressed={pinned}
                    disabled={!onAction}
                    onClick={() =>
                      onAction?.(surface.id, pinned ? "unpin" : "pin")
                    }
                  >
                    {pinned ? <PinOff /> : <Pin />}
                  </Button>
                </div>
              </header>
              <A2uiSurface surface={surface} />
            </article>
          );
        })}
    </section>
  );
}
