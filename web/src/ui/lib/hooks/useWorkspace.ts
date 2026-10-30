import { useCallback, useEffect, useRef, useState } from "react";
import type { z } from "zod";
import {
    eventSchema,
    messageSchema,
    type Placements,
    type VoiceEvent,
} from "@/lib/workspace";
import type { A2uiMessage } from "@/lib/a2ui";

function reduceOutput(text: string, event: z.infer<typeof eventSchema>) {
    if (event.type === "instruction") return "";
    if (event.type === "stdout_delta" || event.type === "stderr_delta")
        return (text + event.text).slice(-65536);
    if (event.type === "result" && event.exception)
        return (text + event.exception).slice(-65536);
    return text;
}

function retainWidgets(messages: A2uiMessage[]) {
    if (messages.length <= 200) return messages;
    const surfaces = new Map<string, A2uiMessage[]>();
    for (const message of messages) {
        if ("createSurface" in message)
            surfaces.set(message.createSurface.surfaceId, [message]);
        else if ("deleteSurface" in message)
            surfaces.delete(message.deleteSurface.surfaceId);
        else {
            const surface = surfaces.get(message.updateComponents.surfaceId);
            if (surface)
                surfaces.set(message.updateComponents.surfaceId, [
                    surface[0],
                    message,
                ]);
        }
    }
    return [...surfaces.values()].flat();
}

export function useWorkspace() {
    const socket = useRef<WebSocket | null>(null);
    const [connected, setConnected] = useState(false);
    const [busy, setBusy] = useState(false);
    const [error, setError] = useState("");
    const [uiMessages, setUiMessages] = useState<A2uiMessage[]>([]);
    const [placements, setPlacements] = useState<Placements>({});
    const [voiceEvent, setVoiceEvent] = useState<VoiceEvent | null>(null);
    const [output, setOutput] = useState("");

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
                        setPlacements(event.placements);
                        setUiMessages(
                            retainWidgets(
                                event.events.flatMap((item) =>
                                    item.type === "a2ui" ? item.messages : [],
                                ),
                            ),
                        );
                        setBusy(event.busy);
                        setError("");
                        setConnected(true);
                        setOutput(event.events.reduce(reduceOutput, ""));
                    } else if (event.type === "state") {
                        setBusy(event.busy);
                    } else {
                        if (event.type === "error") setError(event.text);
                        if (event.type === "instruction") setError("");
                        setOutput((previous) => reduceOutput(previous, event));
                        if (event.type === "placement")
                            setPlacements(event.widgets);
                        if (event.type === "a2ui")
                            setUiMessages((previous) =>
                                retainWidgets([...previous, ...event.messages]),
                            );
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

    const send = useCallback(
        (instruction: string) => {
            if (
                socket.current?.readyState !== WebSocket.OPEN ||
                !connected ||
                busy
            )
                return false;
            socket.current.send(JSON.stringify({ type: "run", instruction }));
            setBusy(true);
            return true;
        },
        [connected, busy],
    );

    const sendVoice = useCallback((action: "start" | "stop", sdp?: string) => {
        if (socket.current?.readyState !== WebSocket.OPEN) return false;
        socket.current.send(
            JSON.stringify({ type: "voice", action, ...(sdp ? { sdp } : {}) }),
        );
        return true;
    }, []);

    return {
        connected,
        busy,
        uiMessages,
        placements,
        error,
        send,
        sendVoice,
        voiceEvent,
        output,
    };
}
