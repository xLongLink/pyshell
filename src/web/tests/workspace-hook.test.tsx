import { afterEach, expect, test, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useWorkspace } from "../lib/hooks/useWorkspace";
import { catalogId } from "../lib/a2ui";

const sockets: Socket[] = [];
class Socket {
    onmessage: ((message: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    close = vi.fn();
    constructor(public url: string) {
        sockets.push(this);
    }
    receive(event: unknown) {
        act(() => this.onmessage?.({ data: JSON.stringify(event) }));
    }
}

afterEach(() => {
    cleanup();
    sockets.length = 0;
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

test("widget snapshots replace state and restore it after reconnect without submissions", () => {
    vi.stubGlobal("WebSocket", Socket);
    vi.useFakeTimers();
    const { result, unmount } = renderHook(useWorkspace);
    expect(sockets[0].url).toMatch(/\/ui\/ws$/);
    const messages = [
        { version: "v0.9", createSurface: { surfaceId: "sales", catalogId } },
    ];
    sockets[0].receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        placements: { sales: { pinned: true } },
    });
    expect(result.current.uiMessages).toHaveLength(1);
    expect(result.current.placements.sales.pinned).toBe(true);
    sockets[0].receive({ type: "snapshot", events: [{ type: "unknown" }] });
    expect(result.current.uiMessages).toHaveLength(1);
    expect(result.current.error).toContain("invalid");
    act(() => sockets[0].onclose?.());
    act(() => vi.advanceTimersByTime(1500));
    sockets[1].receive({ type: "snapshot", events: [], placements: {} });
    expect(result.current.uiMessages).toEqual([]);
    expect(result.current.placements).toEqual({});
    expect(result.current.error).toBe("");
    sockets[0].receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
    });
    expect(result.current.uiMessages).toEqual([]);
    unmount();
    expect(sockets[1].close).toHaveBeenCalledOnce();
});
