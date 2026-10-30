import { afterEach, expect, test, vi } from "vitest";
import { act, cleanup, render, screen } from "@testing-library/react";
import { Renderer } from "../components/renderer";
import { messageSchema, type WorkspaceMessage } from "../lib/workspace";

afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.useRealTimers();
});

test("renderer isolates container code and forwards current state after readiness", () => {
    const port = { postMessage: vi.fn(), close: vi.fn(), onmessage: null };
    const transfer = {};
    vi.stubGlobal(
        "MessageChannel",
        class {
            port1 = port;
            port2 = transfer;
        },
    );
    const messages: WorkspaceMessage[] = [];
    const { rerender } = render(
        <Renderer messages={messages} placements={{}} />,
    );
    const iframe = screen.getByTitle("Widget workspace") as HTMLIFrameElement;
    expect(iframe.getAttribute("sandbox")).toBe("allow-scripts");
    expect(iframe.getAttribute("src")).toBe("/renderer/");
    const post = vi
        .spyOn(iframe.contentWindow!, "postMessage")
        .mockImplementation(() => {});
    const ready = { type: "pyshell:renderer:ready", version: 1 };
    act(() =>
        window.dispatchEvent(
            new MessageEvent("message", {
                data: ready,
                source: window,
                origin: "null",
            }),
        ),
    );
    expect(post).not.toHaveBeenCalled();
    act(() =>
        window.dispatchEvent(
            new MessageEvent("message", {
                data: ready,
                source: iframe.contentWindow,
                origin: "null",
            }),
        ),
    );
    expect(post).toHaveBeenCalledWith(
        { type: "pyshell:renderer:connect", version: 1 },
        { targetOrigin: "*", transfer: [transfer] },
    );
    expect(port.postMessage).toHaveBeenLastCalledWith({
        type: "snapshot",
        messages: [],
        placements: {},
    });
    port.postMessage.mockClear();
    rerender(
        <Renderer
            messages={messages}
            placements={{ chart: { pinned: true } }}
        />,
    );
    expect(port.postMessage).toHaveBeenCalledExactlyOnceWith({
        type: "placement",
        placements: { chart: { pinned: true } },
    });
    const updatedMessages: WorkspaceMessage[] = [
        { version: "v0.9", createSurface: { surfaceId: "chart" } },
    ];
    const placements = { chart: { pinned: true } };
    rerender(<Renderer messages={updatedMessages} placements={placements} />);
    expect(port.postMessage).toHaveBeenCalledWith({
        type: "snapshot",
        messages: updatedMessages,
        placements,
    });
});

test("outer workspace preserves future widget schemas without interpreting them", () => {
    const event = {
        type: "a2ui",
        messages: [
            {
                version: "v0.9",
                updateComponents: {
                    surfaceId: "chart",
                    components: [
                        {
                            id: "root",
                            component: "FutureWidget",
                            newField: [1, 2],
                        },
                    ],
                },
            },
        ],
    };
    expect(messageSchema.parse(event)).toEqual(event);
});

test("renderer waits for a snapshot acknowledgement and recovers from timeout", () => {
    vi.useFakeTimers();
    const port = {
        postMessage: vi.fn(),
        close: vi.fn(),
        onmessage: null as null | ((event: { data: unknown }) => void),
    };
    vi.stubGlobal(
        "MessageChannel",
        class {
            port1 = port;
            port2 = {};
        },
    );
    render(<Renderer messages={[]} placements={{}} />);
    const iframe = screen.getByTitle("Widget workspace") as HTMLIFrameElement;
    vi.spyOn(iframe.contentWindow!, "postMessage").mockImplementation(() => {});
    act(() =>
        window.dispatchEvent(
            new MessageEvent("message", {
                data: { type: "pyshell:renderer:ready", version: 1 },
                source: iframe.contentWindow,
                origin: "null",
            }),
        ),
    );
    act(() => vi.advanceTimersByTime(15000));
    expect(screen.getByRole("alert").textContent).toContain("did not respond");
    act(() => port.onmessage?.({ data: { type: "updated" } }));
    expect(screen.queryByRole("alert")).toBeNull();
    act(() => vi.advanceTimersByTime(15000));
    expect(screen.queryByRole("alert")).toBeNull();
});

test("renderer reports incompatible bridge versions", () => {
    vi.useFakeTimers();
    render(<Renderer messages={[]} placements={{}} />);
    const iframe = screen.getByTitle("Widget workspace") as HTMLIFrameElement;
    act(() =>
        window.dispatchEvent(
            new MessageEvent("message", {
                data: { type: "pyshell:renderer:ready", version: 2 },
                source: iframe.contentWindow,
                origin: "null",
            }),
        ),
    );
    expect(screen.getByRole("alert").textContent).toContain(
        "unsupported bridge version",
    );
    act(() => vi.advanceTimersByTime(30000));
    expect(screen.getByRole("alert").textContent).toContain(
        "unsupported bridge version",
    );
});
