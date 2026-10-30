import { useCallback, useEffect, useRef, useState } from "react";
import {
  applyEvent,
  messageSchema,
  type Entry,
  type Placements,
  type WidgetCommand,
} from "@/lib/workspace";
import type { A2uiMessage } from "@/lib/a2ui";

export function useWorkspace() {
  const socket = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState("");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [error, setError] = useState("");
  const [uiMessages, setUiMessages] = useState<A2uiMessage[]>([]);
  const [placements, setPlacements] = useState<Placements>({});

  useEffect(() => {
    let disposed = false;
    let reconnect: ReturnType<typeof setTimeout>;

    function connect() {
      const protocol = location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${location.host}/ws`);
      socket.current = ws;
      ws.onmessage = (message) => {
        if (disposed) return;
        try {
          const event = messageSchema.parse(JSON.parse(message.data));
          if (event.type === "snapshot") {
            setEntries(event.events.reduce(applyEvent, []));
            setPlacements(event.placements);
            setUiMessages(
              event.events.flatMap((item) =>
                item.type === "a2ui" ? item.messages : [],
              ),
            );
            setBusy(event.busy);
            setModel(event.model);
            setError("");
            setConnected(true);
          } else if (event.type === "state") {
            setBusy(event.busy);
          } else {
            if (event.type === "reset") {
              setUiMessages([]);
              setPlacements({});
            }
            if (event.type === "placement") setPlacements(event.widgets);
            if (event.type === "a2ui")
              setUiMessages((previous) => [...previous, ...event.messages]);
            setEntries((previous) => applyEvent(previous, event));
          }
        } catch {
          setError("The API sent an invalid workspace event.");
        }
      };
      ws.onclose = () => {
        if (disposed) return;
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
    (
      command:
        | { type: "run"; instruction: string }
        | { type: "reset" }
        | WidgetCommand,
    ) => {
      if (
        socket.current?.readyState !== WebSocket.OPEN ||
        !connected ||
        (busy && command.type !== "widget")
      )
        return false;
      socket.current.send(JSON.stringify(command));
      if (command.type !== "widget") setBusy(true);
      return true;
    },
    [connected, busy],
  );

  return {
    connected,
    busy,
    model,
    entries,
    uiMessages,
    placements,
    error,
    send,
  };
}
