import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useWorkspace } from "../lib/hooks/useWorkspace";

const sockets: Socket[] = [];

class Socket {
    static OPEN = 1;
    readyState = Socket.OPEN;
    onmessage: ((message: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    send = vi.fn();
    close = vi.fn();
    constructor() {
        sockets.push(this);
    }
    receive(event: unknown) {
        act(() => this.onmessage?.({ data: JSON.stringify(event) }));
    }
}

beforeEach(() => {
    sockets.length = 0;
    vi.stubGlobal("WebSocket", Socket);
});

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

test("application sends voice controls and receives answers without widget state", () => {
    const { result } = renderHook(useWorkspace);
    const socket = sockets[0];
    socket.receive({ type: "snapshot" });
    expect(result.current.connected).toBe(true);
    expect(result.current.sendVoice("start", "offer")).toBe(true);
    expect(socket.send).toHaveBeenLastCalledWith(
        JSON.stringify({ type: "voice", action: "start", sdp: "offer" }),
    );
    socket.receive({ type: "voice", name: "answer", sdp: "answer" });
    expect(result.current.voiceEvent).toEqual({
        type: "voice",
        name: "answer",
        sdp: "answer",
    });
    expect(result.current).not.toHaveProperty("uiMessages");
    socket.receive({ type: "error", text: "Execution failed" });
    expect(result.current.error).toBe("Execution failed");
    socket.receive({ type: "instruction", text: "Retry" });
    expect(result.current.error).toBe("");
});

test("reconnection does not replay commands and ignores stale connections", () => {
    vi.useFakeTimers();
    const { result, unmount } = renderHook(useWorkspace);
    sockets[0].receive({ type: "snapshot" });
    act(() => sockets[0].onclose?.());
    expect(result.current.connected).toBe(false);
    act(() => vi.advanceTimersByTime(1500));
    expect(sockets).toHaveLength(2);
    sockets[1].receive({ type: "snapshot" });
    sockets[0].receive({ type: "error", text: "Stale" });
    expect(result.current.error).toBe("");
    expect(sockets[1].send).not.toHaveBeenCalled();
    unmount();
    expect(sockets[1].close).toHaveBeenCalledOnce();
});
