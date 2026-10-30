import { useEffect, useRef, useState } from "react";
import type { Placements, WorkspaceMessage } from "@/lib/workspace";

export function Renderer({
    messages,
    placements,
}: {
    messages: WorkspaceMessage[];
    placements: Placements;
}) {
    const frame = useRef<HTMLIFrameElement>(null);
    const port = useRef<MessagePort | null>(null);
    const latest = useRef({ messages, placements });
    const [error, setError] = useState("");
    latest.current = { messages, placements };

    useEffect(() => {
        let timer: ReturnType<typeof setTimeout>;
        function wait() {
            clearTimeout(timer);
            timer = setTimeout(
                () =>
                    setError(
                        "The widget renderer did not respond. Reload to retry.",
                    ),
                15000,
            );
        }
        function receive(event: MessageEvent) {
            if (
                !event.source ||
                event.source !== frame.current?.contentWindow ||
                event.origin !== "null"
            )
                return;
            if (event.data?.type !== "pyshell:renderer:ready") return;
            if (event.data.version !== 1) {
                clearTimeout(timer);
                setError(
                    "The widget renderer uses an unsupported bridge version.",
                );
                return;
            }
            port.current?.close();
            const channel = new MessageChannel();
            port.current = channel.port1;
            channel.port1.onmessage = (message) => {
                if (message.data?.type === "updated") {
                    clearTimeout(timer);
                    setError("");
                }
            };
            event.source.postMessage(
                { type: "pyshell:renderer:connect", version: 1 },
                { targetOrigin: "*", transfer: [channel.port2] },
            );
            channel.port1.postMessage({ type: "snapshot", ...latest.current });
            wait();
        }
        window.addEventListener("message", receive);
        wait();
        return () => {
            window.removeEventListener("message", receive);
            clearTimeout(timer);
            port.current?.close();
            port.current = null;
        };
    }, []);

    useEffect(() => {
        port.current?.postMessage({
            type: "snapshot",
            messages,
            placements: latest.current.placements,
        });
    }, [messages]);

    useEffect(() => {
        port.current?.postMessage({ type: "placement", placements });
    }, [placements]);

    return (
        <>
            <iframe
                ref={frame}
                src="/renderer/"
                title="Widget workspace"
                sandbox="allow-scripts"
                className="h-full w-full border-0"
            />
            {error && (
                <p
                    role="alert"
                    className="fixed top-4 right-4 z-20 rounded bg-background p-3 text-sm text-destructive"
                >
                    {error}
                </p>
            )}
        </>
    );
}
