import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
    act,
    cleanup,
    render,
    renderHook,
    screen,
} from "@testing-library/react";
import { useWorkspace } from "../lib/hooks/useWorkspace";
import { catalogId } from "../lib/a2ui";
import { Results } from "../components/results";

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
        placements: { sales: { pinned: true } },
    });
    expect(result.current.connected).toBe(true);
    expect(result.current.uiMessages).toHaveLength(2);
    expect(result.current.placements).toEqual({ sales: { pinned: true } });
    expect(result.current.error).toBe("");
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
        placements: {},
    });
    expect(result.current.uiMessages).toEqual([]);
    expect(result.current.placements).toEqual({});
});

test("history compaction preserves an unaffected chart and its keyboard focus", () => {
    const { result } = renderHook(useWorkspace);
    const socket = sockets[0];
    const created = {
        version: "v0.9",
        createSurface: { surfaceId: "other", catalogId },
    };
    const removed = {
        version: "v0.9",
        deleteSurface: { surfaceId: "other" },
    };
    socket.receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages: [...messages, created] }],
    });
    const { rerender } = render(
        <Results messages={result.current.uiMessages} />,
    );
    const chart = screen.getByRole("application");
    chart.focus();
    for (let batch = 0; batch < 2; batch++) {
        socket.receive({
            type: "a2ui",
            messages: Array.from({ length: 98 }, (_, index) =>
                index % 2 === 0 ? removed : created,
            ),
        });
        rerender(<Results messages={result.current.uiMessages} />);
    }
    expect(result.current.uiMessages).toHaveLength(199);
    socket.receive({ type: "a2ui", messages: [removed, created] });
    expect(result.current.uiMessages).toHaveLength(3);
    rerender(<Results messages={result.current.uiMessages} />);
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("application")).toBe(chart);
    expect(document.activeElement).toBe(chart);
    expect(
        screen.getByRole("article", { name: "Widget other" }).children,
    ).toHaveLength(0);
});

test("invalid snapshots do not replace current widgets or placements", () => {
    const { result } = renderHook(useWorkspace);
    const socket = sockets[0];
    socket.receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        placements: { sales: { pinned: true } },
    });
    const restored = result.current;
    socket.receive({
        type: "snapshot",
        events: [{ type: "instruction", text: "Not snapshot data" }],
        placements: {},
    });
    expect(result.current.error).toBe(
        "The API sent an invalid workspace event.",
    );
    expect(result.current.uiMessages).toBe(restored.uiMessages);
    expect(result.current.placements).toBe(restored.placements);
    expect(result.current.connected).toBe(true);
});

test("reconnect snapshots replace widget history and stale sockets cannot change state", () => {
    vi.useFakeTimers();
    const { result, unmount } = renderHook(useWorkspace);
    const socket = sockets[0];
    expect(result.current.connected).toBe(false);
    expect(socket.send).not.toHaveBeenCalled();
    socket.receive({
        type: "snapshot",
        events: [],
    });
    expect(result.current.connected).toBe(true);
    act(() => socket.onclose?.());
    expect(result.current.connected).toBe(false);
    act(() => vi.advanceTimersByTime(1500));
    const reconnected = sockets[1];
    reconnected.receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        placements: { sales: { pinned: false } },
    });
    expect(result.current.connected).toBe(true);
    expect(result.current.error).toBe("");
    expect(result.current.uiMessages).toHaveLength(2);
    const restored = result.current;
    socket.receive({ type: "error", text: "Stale connection" });
    socket.receive({ type: "placement", widgets: {} });
    expect(result.current).toBe(restored);
    reconnected.readyState = 3;
    act(() => {
        expect(result.current.sendVoice("start", "offer")).toBe(false);
    });
    expect(reconnected.send).not.toHaveBeenCalled();
    act(() => reconnected.onclose?.());
    unmount();
    reconnected.receive({ type: "error", text: "Late event" });
    expect(reconnected.close).toHaveBeenCalledOnce();
    act(() => vi.advanceTimersByTime(1500));
    expect(sockets).toHaveLength(2);
});

test("workspace sends voice controls and receives answers", () => {
    const { result } = renderHook(useWorkspace);
    const socket = sockets[0];
    socket.receive({ type: "snapshot", events: [] });
    expect(result.current.error).toBe("");
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
