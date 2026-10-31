import { useEffect, useState } from "react";
import { messageSchema, type Placements } from "@/lib/workspace";
import type { A2uiMessage } from "@/lib/a2ui";

export function useWorkspace() {
    const [error, setError] = useState("");
    const [uiMessages, setUiMessages] = useState<A2uiMessage[]>([]);
    const [placements, setPlacements] = useState<Placements>({});

    useEffect(() => {
        let disposed = false;
        let socket: WebSocket;
        let reconnect: ReturnType<typeof setTimeout>;

        function connect() {
            const protocol = location.protocol === "https:" ? "wss:" : "ws:";
            const ws = new WebSocket(`${protocol}//${location.host}/ui/ws`);
            socket = ws;
            ws.onmessage = (message) => {
                if (disposed || socket !== ws) return;
                try {
                    const snapshot = messageSchema.parse(
                        JSON.parse(message.data),
                    );
                    setUiMessages(
                        snapshot.events.flatMap((event) => event.messages),
                    );
                    setPlacements(snapshot.placements);
                    setError("");
                } catch {
                    setError("The container sent an invalid widget snapshot.");
                }
            };
            ws.onclose = () => {
                if (disposed || socket !== ws) return;
                setError(
                    "Widget connection lost. Reconnecting; Python may still be running.",
                );
                reconnect = setTimeout(connect, 1500);
            };
        }

        connect();
        return () => {
            disposed = true;
            clearTimeout(reconnect);
            socket?.close();
        };
    }, []);

    return { uiMessages, placements, error };
}
