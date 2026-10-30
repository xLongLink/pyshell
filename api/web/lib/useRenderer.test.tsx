import { afterEach, expect, test, vi } from "vitest";
import { act, cleanup, renderHook } from "@testing-library/react";
import { useRenderer } from "./useRenderer";
import { catalogId } from "./a2ui";

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
});

test("renderer validates snapshots, replaces state, and rejects foreign senders", () => {
    const parent = { postMessage: vi.fn() };
    vi.stubGlobal("parent", parent);
    const port = {
        onmessage: null as null | ((event: { data: unknown }) => void),
        postMessage: vi.fn(),
        close: vi.fn(),
    };
    const { result, unmount } = renderHook(useRenderer);
    expect(parent.postMessage).toHaveBeenCalledWith(
        { type: "pyshell:renderer:ready", version: 1 },
        "*",
    );
    const connect = (source: MessageEventSource) =>
        act(() => {
            window.dispatchEvent(
                new MessageEvent("message", {
                    source,
                    data: { type: "pyshell:renderer:connect", version: 1 },
                    ports: [port as unknown as MessagePort],
                }),
            );
        });
    connect(window);
    expect(port.postMessage).not.toHaveBeenCalled();
    connect(parent as unknown as Window);
    expect(port.postMessage).not.toHaveBeenCalled();
    const messages = [
        { version: "v0.9", createSurface: { surfaceId: "plot", catalogId } },
    ];
    act(() =>
        port.onmessage?.({
            data: {
                type: "snapshot",
                messages,
                placements: { plot: { pinned: true } },
            },
        }),
    );
    expect(result.current.messages).toEqual(messages);
    expect(result.current.placements).toEqual({ plot: { pinned: true } });
    expect(port.postMessage).toHaveBeenLastCalledWith({ type: "updated" });
    const retainedMessages = result.current.messages;
    act(() =>
        port.onmessage?.({
            data: {
                type: "placement",
                placements: { plot: { pinned: false } },
            },
        }),
    );
    expect(result.current.messages).toBe(retainedMessages);
    expect(result.current.placements).toEqual({ plot: { pinned: false } });
    expect(port.postMessage).toHaveBeenLastCalledWith({ type: "updated" });
    act(() =>
        port.onmessage?.({
            data: { type: "snapshot", messages: [], placements: {} },
        }),
    );
    expect(result.current.messages).toEqual([]);
    port.postMessage.mockClear();
    act(() =>
        port.onmessage?.({
            data: {
                type: "snapshot",
                messages: [{ invalid: true }],
                placements: {},
            },
        }),
    );
    expect(result.current.error).toContain("unsupported widget data");
    expect(port.postMessage).toHaveBeenCalledExactlyOnceWith({
        type: "updated",
    });
    act(() =>
        port.onmessage?.({
            data: { type: "snapshot", messages, placements: {} },
        }),
    );
    expect(result.current.error).toBe("");
    expect(result.current.messages).toEqual(messages);
    expect(port.postMessage).toHaveBeenLastCalledWith({ type: "updated" });
    unmount();
    expect(port.close).toHaveBeenCalledOnce();
});
