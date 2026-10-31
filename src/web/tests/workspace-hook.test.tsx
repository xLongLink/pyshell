import { afterEach, expect, test, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useWorkspace } from "../lib/hooks/useWorkspace";
import { catalogId } from "../lib/a2ui";

const sockets: Socket[] = [];
function update(surfaceId: string) {
    return {
        version: "v0.9",
        updateComponents: {
            surfaceId,
            components: [
                {
                    id: "root",
                    component: "Figure",
                    data: [{ type: "bar", x: ["A"], y: [1] }],
                },
            ],
        },
    };
}
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
        update("sales"),
    ];
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
    expect(result.current.placements.sales.pinned).toBe(true);
    sockets[0].receive({ type: "snapshot", events: [{ type: "unknown" }] });
    expect(result.current.surfaces).toHaveLength(1);
    expect(result.current.error).toContain("invalid");
    act(() => sockets[0].onclose?.());
    act(() => vi.advanceTimersByTime(1500));
    sockets[1].receive({ type: "snapshot", events: [], placements: {} });
    expect(result.current.surfaces).toEqual([]);
    expect(result.current.placements).toEqual({});
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

test("invalid lifecycle snapshots retain the last valid surfaces and placements atomically", () => {
    vi.stubGlobal("WebSocket", Socket);
    const { result } = renderHook(useWorkspace);
    const creation = {
        version: "v0.9",
        createSurface: { surfaceId: "sales", catalogId },
    };
    const valid = {
        type: "snapshot",
        events: [{ type: "a2ui", messages: [creation, update("sales")] }],
        placements: { sales: { pinned: true } },
    };
    sockets[0].receive(valid);
    const surfaces = result.current.surfaces;
    const placements = result.current.placements;
    for (const incoming of [
        {
            ...valid,
            events: [{ type: "a2ui", messages: [creation, update("other")] }],
            placements: { sales: { pinned: false } },
        },
        { ...valid, placements: {} },
    ]) {
        sockets[0].receive(incoming);
        expect(result.current.surfaces).toBe(surfaces);
        expect(result.current.placements).toBe(placements);
        expect(result.current.error).toContain("invalid");
    }
    sockets[0].receive(valid);
    expect(result.current.error).toBe("");
});
