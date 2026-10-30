import { useCallback, useEffect, useRef, useState } from "react";
import { applyEvent, messageSchema, type Entry } from "@/lib/workspace";

export function useWorkspace() {
  const socket = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState("");
  const [entries, setEntries] = useState<Entry[]>([]);
  const [error, setError] = useState("");

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
            setBusy(event.busy);
            setModel(event.model);
            setError("");
            setConnected(true);
          } else if (event.type === "state") {
            setBusy(event.busy);
          } else {
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
    (command: { type: "run"; instruction: string } | { type: "reset" }) => {
      if (socket.current?.readyState !== WebSocket.OPEN || !connected || busy)
        return false;
      socket.current.send(JSON.stringify(command));
      setBusy(true);
      return true;
    },
    [connected, busy],
  );

  return { connected, busy, model, entries, error, send };
}
