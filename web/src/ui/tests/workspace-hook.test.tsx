import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useWorkspace } from "../lib/hooks/useWorkspace";
import { catalogId } from "../lib/a2ui";

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

const messages = [
    { version: "v0.9", createSurface: { surfaceId: "sales", catalogId } },
    {
        version: "v0.9",
        updateComponents: {
            surfaceId: "sales",
            components: [
                {
                    id: "root",
                    component: "Figure",
                    data: [{ type: "bar", x: ["A"], y: [1] }],
                },
            ],
        },
    },
];

test("workspace restores widgets, ignores transcript fragments, and handles live placement and reset", () => {
    const { result } = renderHook(useWorkspace);
    const socket = sockets[0];
    socket.receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        busy: false,
        model: "test",
        placements: { sales: { pinned: true } },
    });
    expect(result.current.connected).toBe(true);
    expect(result.current.uiMessages).toHaveLength(2);
    expect(result.current.placements).toEqual({ sales: { pinned: true } });
    const state = result.current;
    socket.receive({ type: "code_delta", text: "print('not rendered')" });
    expect(result.current).toBe(state);
    socket.receive({
        type: "placement",
        widgets: { sales: { pinned: false } },
    });
    expect(result.current.placements).toEqual({ sales: { pinned: false } });
    socket.receive({ type: "error", text: "Execution failed" });
    expect(result.current.error).toBe("Execution failed");
    socket.receive({ type: "instruction", text: "Retry" });
    expect(result.current.error).toBe("");
    socket.receive({
        type: "a2ui",
        messages: [{ version: "v0.9", deleteSurface: { surfaceId: "sales" } }],
    });
    expect(result.current.uiMessages).toHaveLength(3);
    socket.receive({ type: "reset" });
    expect(result.current.uiMessages).toEqual([]);
    expect(result.current.placements).toEqual({});
});

test("reconnect snapshots replace widget history and keep commands blocked until ready", () => {
    vi.useFakeTimers();
    const { result, unmount } = renderHook(useWorkspace);
    const socket = sockets[0];
    let sent = false;
    act(() => {
        sent = result.current.send({ type: "run", instruction: "Create" });
    });
    expect(sent).toBe(false);
    socket.receive({
        type: "snapshot",
        events: [],
        busy: false,
        model: "test",
    });
    act(() => {
        sent = result.current.send({ type: "run", instruction: "Create" });
    });
    expect(sent).toBe(true);
    expect(result.current.busy).toBe(true);
    act(() => socket.onclose?.());
    expect(result.current.connected).toBe(false);
    act(() => vi.advanceTimersByTime(1500));
    const reconnected = sockets[1];
    reconnected.receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        busy: false,
        model: "test",
        placements: { sales: { pinned: false } },
    });
    expect(result.current.connected).toBe(true);
    expect(result.current.busy).toBe(false);
    expect(result.current.error).toBe("");
    expect(result.current.uiMessages).toHaveLength(2);
    unmount();
    reconnected.receive({ type: "reset" });
    expect(reconnected.close).toHaveBeenCalledOnce();
});
