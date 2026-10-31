import { useCallback, useEffect, useRef, useState } from "react";
import { messageSchema, type VoiceEvent } from "@/lib/workspace";

export function useWorkspace() {
    const socket = useRef<WebSocket | null>(null);
    const [connected, setConnected] = useState(false);
    const [error, setError] = useState("");
    const [voiceEvent, setVoiceEvent] = useState<VoiceEvent | null>(null);

    useEffect(() => {
        let disposed = false;
        let reconnect: ReturnType<typeof setTimeout>;

        function connect() {
            const protocol = location.protocol === "https:" ? "wss:" : "ws:";
            const ws = new WebSocket(`${protocol}//${location.host}/ws`);
            socket.current = ws;
            ws.onmessage = (message) => {
                if (disposed || socket.current !== ws) return;
                try {
                    const event = messageSchema.parse(JSON.parse(message.data));
                    if (event.type === "voice") {
                        setVoiceEvent(event);
                    } else if (event.type === "snapshot") {
                        setError("");
                        setConnected(true);
                    } else {
                        if (event.type === "error") setError(event.text);
                        if (event.type === "instruction") setError("");
                    }
                } catch {
                    setError("The API sent an invalid workspace event.");
                }
            };
            ws.onclose = () => {
                if (disposed || socket.current !== ws) return;
                setConnected(false);
                setError(
                    "Connection lost. Reconnecting; submitted work may still be running.",
                );
                reconnect = setTimeout(connect, 1500);
            };
        }

        connect();
        return () => {
            disposed = true;
            clearTimeout(reconnect);
            socket.current?.close();
        };
    }, []);

    const sendVoice = useCallback((action: "start" | "stop", sdp?: string) => {
        if (socket.current?.readyState !== WebSocket.OPEN) return false;
        socket.current.send(
            JSON.stringify({ type: "voice", action, ...(sdp ? { sdp } : {}) }),
        );
        return true;
    }, []);

    return {
        connected,
        error,
        sendVoice,
        voiceEvent,
    };
}
