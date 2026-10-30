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

test("workspace offers voice", () => {
    workspace();
    render(<App />);
    const voice = screen.getByRole("button", { name: "Start voice" });
    expect((voice as HTMLButtonElement).disabled).toBe(false);
});

test("workspace keeps errors visible and voice available", () => {
    workspace({ error: "The operation failed" });
    render(<App />);
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
