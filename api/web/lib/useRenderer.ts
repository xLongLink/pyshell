import { useEffect, useState } from "react";
import { z } from "zod";
import { a2uiMessageSchema, type A2uiMessage } from "@/lib/a2ui";
import { placementsSchema, type Placements } from "@/lib/placements";

const updateSchema = z.discriminatedUnion("type", [
    z.object({
        type: z.literal("snapshot"),
        messages: z.array(a2uiMessageSchema).max(240),
        placements: placementsSchema,
    }),
    z.object({
        type: z.literal("placement"),
        placements: placementsSchema,
    }),
]);

export function useRenderer() {
    const [state, setState] = useState<{
        messages: A2uiMessage[];
        placements: Placements;
    }>({ messages: [], placements: {} });
    const [error, setError] = useState("");

    useEffect(() => {
        let port: MessagePort | undefined;
        function connect(event: MessageEvent) {
            if (window.parent === window || event.source !== window.parent)
                return;
            if (
                event.data?.type !== "pyshell:renderer:connect" ||
                event.data.version !== 1 ||
                !event.ports[0]
            )
                return;
            port?.close();
            port = event.ports[0];
            port.onmessage = (message) => {
                const parsed = updateSchema.safeParse(message.data);
                if (!parsed.success) {
                    setState({ messages: [], placements: {} });
                    setError("The workspace contains unsupported widget data.");
                } else {
                    const update = parsed.data;
                    setState((previous) =>
                        update.type === "snapshot"
                            ? update
                            : { ...previous, placements: update.placements },
                    );
                    setError("");
                }
                port?.postMessage({ type: "updated" });
            };
        }
        window.addEventListener("message", connect);
        window.parent.postMessage(
            { type: "pyshell:renderer:ready", version: 1 },
            "*",
        );
        return () => {
            window.removeEventListener("message", connect);
            port?.close();
        };
    }, []);

    return { ...state, error };
}
