import { afterEach, expect, test, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useWorkspace } from "../lib/hooks/useWorkspace";
import { creation, update } from "./workspace-fixtures.ts";

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
    const messages = [creation("sales"), update("sales")];
    sockets[0].receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        placements: { sales: { pinned: true } },
    });
    expect(result.current.surfaces).toHaveLength(1);
    expect(result.current.surfaces[0].props.data[0]).toEqual({
        type: "bar",
        name: "",
        x: ["A"],
        y: [1],
    });
    expect(result.current.surfaces[0].pinned).toBe(true);
    const surfaces = result.current.surfaces;
    sockets[0].receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages: [messages[0], update("other")] }],
        placements: { sales: { pinned: false } },
    });
    expect(result.current.surfaces).toBe(surfaces);
    expect(result.current.surfaces[0].pinned).toBe(true);
    expect(result.current.error).toContain("invalid");
    act(() => sockets[0].onclose?.());
    act(() => vi.advanceTimersByTime(1500));
    sockets[1].receive({ type: "snapshot", events: [], placements: {} });
    expect(result.current.surfaces).toEqual([]);
    expect(result.current.error).toBe("");
    sockets[0].receive({
        type: "snapshot",
        events: [{ type: "a2ui", messages }],
        placements: { sales: { pinned: true } },
    });
    expect(result.current.surfaces).toEqual([]);
    unmount();
    expect(sockets[1].close).toHaveBeenCalledOnce();
});
