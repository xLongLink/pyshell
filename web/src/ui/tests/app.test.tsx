import { afterEach, expect, test, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
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
        busy: false,
        model: "test",
        entries: [
            { kind: "instruction" as const, text: "Show sales" },
            {
                kind: "code" as const,
                text: "print('private execution details')",
            },
            { kind: "output" as const, text: "console output" },
        ],
        uiMessages: [],
        placements: {},
        error: "",
        send: vi.fn(() => true),
        ...overrides,
    };
    vi.mocked(useWorkspace).mockReturnValue(state);
    return state;
}

test("workspace hides the transcript and submits widget instructions", () => {
    const state = workspace();
    const { container } = render(<App />);
    expect(screen.queryByRole("heading", { name: "Workspace" })).toBeNull();
    expect(screen.queryByText("Ready")).toBeNull();
    expect(screen.queryByRole("textbox")).toBeNull();
    expect(
        screen.getByText(/Your generated widgets will appear here/),
    ).toBeTruthy();
    expect(screen.queryByText("Show sales")).toBeNull();
    expect(screen.queryByText("console output")).toBeNull();
    expect(container.querySelector("pre, code")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Open chat" }));
    const input = screen.getByRole("textbox", { name: "Instruction" });
    fireEvent.change(input, { target: { value: "Show revenue" } });
    fireEvent.click(screen.getByRole("button", { name: "Send instruction" }));
    expect(state.send).toHaveBeenCalledWith({
        type: "run",
        instruction: "Show revenue",
    });
    expect(screen.queryByRole("textbox")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Open chat" }));
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe("");
});

test("workspace keeps errors and working status visible without a transcript", () => {
    workspace({ busy: true, error: "The operation failed" });
    render(<App />);
    expect(screen.queryByRole("status")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Open chat" }));
    expect(screen.getByRole("status").textContent).toBe("Working…");
    expect(screen.getByRole("alert").textContent).toBe("The operation failed");
    expect(
        (
            screen.getByRole("textbox", {
                name: "Instruction",
            }) as HTMLTextAreaElement
        ).disabled,
    ).toBe(true);
});

test("chat preserves a draft when closed and stays open if submission fails", () => {
    const state = workspace();
    state.send.mockReturnValue(false);
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Open chat" }));
    fireEvent.change(screen.getByRole("textbox"), {
        target: { value: "Show sales" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Close chat" }));
    expect(screen.queryByRole("textbox")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Open chat" }));
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).value).toBe(
        "Show sales",
    );
    fireEvent.click(screen.getByRole("button", { name: "Send instruction" }));
    expect(screen.getByRole("textbox")).toBeTruthy();
});
