import { afterEach, expect, test, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import App from "../App";
import { useWorkspace } from "../lib/hooks/useWorkspace";

vi.mock("../lib/hooks/useWorkspace", () => ({ useWorkspace: vi.fn() }));

afterEach(() => {
    cleanup();
    vi.clearAllMocks();
});

function workspace(overrides = {}) {
    const state = {
        connected: true,
        uiMessages: [],
        placements: {},
        error: "",
        sendVoice: vi.fn(() => true),
        voiceEvent: null,
        ...overrides,
    };
    vi.mocked(useWorkspace).mockReturnValue(state);
    return state;
}

test("workspace offers voice without chat or execution output", () => {
    workspace();
    const { container } = render(<App />);
    expect(screen.queryByRole("heading", { name: "Workspace" })).toBeNull();
    expect(screen.queryByText("Ready")).toBeNull();
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(
        screen.queryByText(/Your generated widgets will appear here/),
    ).toBeNull();
    expect(container.querySelector("pre, code")).toBeNull();
    expect(screen.queryByRole("button", { name: "Open chat" })).toBeNull();
    const voice = screen.getByRole("button", { name: "Start voice" });
    expect((voice as HTMLButtonElement).disabled).toBe(false);
});

test("workspace keeps errors visible and voice available", () => {
    workspace({ error: "The operation failed" });
    render(<App />);
    expect(screen.queryByRole("status")).toBeNull();
    expect(screen.getByRole("alert").textContent).toBe("The operation failed");
    expect(
        (
            screen.getByRole("button", {
                name: "Start voice",
            }) as HTMLButtonElement
        ).disabled,
    ).toBe(false);
});

test("workspace disables voice while disconnected", () => {
    workspace({ connected: false });
    render(<App />);
    expect(
        (
            screen.getByRole("button", {
                name: "Start voice",
            }) as HTMLButtonElement
        ).disabled,
    ).toBe(true);
});
