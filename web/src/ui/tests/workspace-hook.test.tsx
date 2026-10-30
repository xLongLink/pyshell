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

test("workspace restores widgets and handles live placement", () => {
    const { result } = renderHook(useWorkspace);
    const socket = sockets[0];
    socket.receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        busy: false,
        placements: { sales: { pinned: true } },
    });
    expect(result.current.connected).toBe(true);
    expect(result.current.uiMessages).toHaveLength(2);
    expect(result.current.placements).toEqual({ sales: { pinned: true } });
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
    socket.receive({
        type: "snapshot",
        events: [],
        busy: false,
        placements: {},
    });
    expect(result.current.uiMessages).toEqual([]);
    expect(result.current.placements).toEqual({});
});

test("reconnect snapshots replace widget history and keep commands blocked until ready", () => {
    vi.useFakeTimers();
    const { result, unmount } = renderHook(useWorkspace);
    const socket = sockets[0];
    let sent = false;
    act(() => {
        sent = result.current.send("Create");
    });
    expect(sent).toBe(false);
    expect(socket.send).not.toHaveBeenCalled();
    socket.receive({
        type: "snapshot",
        events: [],
        busy: false,
    });
    act(() => {
        sent = result.current.send("Create");
    });
    expect(sent).toBe(true);
    expect(result.current.busy).toBe(true);
    expect(socket.send).toHaveBeenCalledExactlyOnceWith(
        JSON.stringify({ type: "run", instruction: "Create" }),
    );
    act(() => {
        sent = result.current.send("Wait");
    });
    expect(sent).toBe(false);
    expect(socket.send).toHaveBeenCalledOnce();
    act(() => socket.onclose?.());
    expect(result.current.connected).toBe(false);
    act(() => vi.advanceTimersByTime(1500));
    const reconnected = sockets[1];
    reconnected.receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        busy: false,
        placements: { sales: { pinned: false } },
    });
    expect(result.current.connected).toBe(true);
    expect(result.current.busy).toBe(false);
    expect(result.current.error).toBe("");
    expect(result.current.uiMessages).toHaveLength(2);
    reconnected.readyState = 3;
    act(() => {
        sent = result.current.send("Disconnected");
    });
    expect(sent).toBe(false);
    expect(result.current.busy).toBe(false);
    expect(reconnected.send).not.toHaveBeenCalled();
    unmount();
    reconnected.receive({ type: "state", busy: true });
    expect(reconnected.close).toHaveBeenCalledOnce();
});

test("snapshot and live output agree on resets, exceptions, and truncation", () => {
    const live = renderHook(useWorkspace);
    const replay = renderHook(useWorkspace);
    const events = [
        { type: "stdout_delta", text: "previous output" },
        { type: "instruction", text: "Start again" },
        { type: "stdout_delta", text: "x".repeat(65536) },
        { type: "stderr_delta", text: "warning\n" },
        {
            type: "result",
            exception: "failure\n",
        },
    ];
    for (const event of events) sockets[0].receive(event);
    sockets[1].receive({
        type: "snapshot",
        events,
        busy: false,
    });
    const expected = "x".repeat(65520) + "warning\nfailure\n";
    expect(live.result.current.output).toBe(expected);
    expect(replay.result.current.output).toBe(expected);
    sockets[0].receive({ type: "instruction", text: "Clear" });
    sockets[1].receive({
        type: "snapshot",
        events: [...events, { type: "instruction", text: "Clear" }],
        busy: false,
    });
    expect(live.result.current.output).toBe("");
    expect(replay.result.current.output).toBe("");
});

test("execution output streams without duplicating final output and voice controls work while busy", () => {
    const { result } = renderHook(useWorkspace);
    const socket = sockets[0];
    socket.receive({ type: "snapshot", events: [], busy: true });
    socket.receive({ type: "stdout_delta", text: "first\n" });
    socket.receive({ type: "stderr_delta", text: "warning\n" });
    socket.receive({
        type: "result",
        exception: null,
    });
    expect(result.current.output).toBe("first\nwarning\n");
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
});
