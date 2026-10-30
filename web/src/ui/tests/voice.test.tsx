import { afterEach, expect, test, vi } from "vitest";
import {
    act,
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
} from "@testing-library/react";
import { Voice } from "../components/voice";

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

test("voice negotiates audio and releases microphone resources on hangup", async () => {
    const track = { stop: vi.fn() };
    const stream = { getTracks: () => [track] };
    vi.stubGlobal("navigator", {
        mediaDevices: { getUserMedia: vi.fn().mockResolvedValue(stream) },
    });
    const close = vi.fn();
    const remote = vi.fn().mockResolvedValue(undefined);
    let peer: { onconnectionstatechange: () => void };
    vi.stubGlobal(
        "RTCPeerConnection",
        class {
            constructor() {
                peer = this;
            }
            connectionState = "connected";
            onconnectionstatechange = () => {};
            iceGatheringState = "complete";
            localDescription = { sdp: "offer" };
            addTrack = vi.fn();
            createDataChannel = vi.fn();
            createOffer = vi
                .fn()
                .mockResolvedValue({ type: "offer", sdp: "offer" });
            setLocalDescription = vi.fn().mockResolvedValue(undefined);
            setRemoteDescription = remote;
            close = close;
        },
    );
    vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
    const send = vi.fn(() => true);
    const { rerender } = render(<Voice connected event={null} send={send} />);
    fireEvent.click(screen.getByRole("button", { name: "Start voice" }));
    await waitFor(() => expect(send).toHaveBeenCalledWith("start", "offer"));
    rerender(
        <Voice
            connected
            event={{ type: "voice", name: "answer", sdp: "answer" }}
            send={send}
        />,
    );
    await waitFor(() =>
        expect(remote).toHaveBeenCalledWith({ type: "answer", sdp: "answer" }),
    );
    expect(screen.queryByRole("status")).toBeNull();
    act(() => peer.onconnectionstatechange());
    expect(screen.queryByRole("status")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Stop voice" }));
    expect(screen.queryByRole("status")).toBeNull();
    expect(send).toHaveBeenLastCalledWith("stop");
    expect(track.stop).toHaveBeenCalledOnce();
    expect(close).toHaveBeenCalledOnce();
    expect(
        (
            screen.getByRole("button", {
                name: "Stopping voice",
            }) as HTMLButtonElement
        ).disabled,
    ).toBe(true);
    rerender(
        <Voice
            connected
            event={{ type: "voice", name: "closed" }}
            send={send}
        />,
    );
    expect(screen.getByRole("button", { name: "Start voice" })).toBeTruthy();
});

test("cancelling microphone permission setup never submits a late call", async () => {
    let grant: (stream: unknown) => void = () => {};
    const pending = new Promise((resolve) => {
        grant = resolve;
    });
    vi.stubGlobal("navigator", {
        mediaDevices: { getUserMedia: () => pending },
    });
    const send = vi.fn(() => true);
    render(<Voice connected event={null} send={send} />);
    fireEvent.click(screen.getByRole("button", { name: "Start voice" }));
    fireEvent.click(screen.getByRole("button", { name: "Stop voice" }));
    const track = { stop: vi.fn() };
    await act(async () => {
        grant({ getTracks: () => [track] });
    });
    expect(track.stop).toHaveBeenCalledOnce();
    expect(send).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Start voice" })).toBeTruthy();
});

test("microphone failures are visible and allow retry", async () => {
    vi.stubGlobal("navigator", {
        mediaDevices: {
            getUserMedia: vi
                .fn()
                .mockRejectedValue(new Error("Microphone access denied")),
        },
    });
    const send = vi.fn(() => true);
    render(<Voice connected event={null} send={send} />);
    fireEvent.click(screen.getByRole("button", { name: "Start voice" }));
    await waitFor(() =>
        expect(screen.getByRole("alert").textContent).toBe(
            "Microphone access denied",
        ),
    );
    expect(send).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Start voice" })).toBeTruthy();
});

test("a stale answer failure cannot stop a restarted voice attempt", async () => {
    const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }];
    vi.stubGlobal("navigator", {
        mediaDevices: {
            getUserMedia: vi
                .fn()
                .mockResolvedValueOnce({ getTracks: () => [tracks[0]] })
                .mockResolvedValueOnce({ getTracks: () => [tracks[1]] }),
        },
    });
    let rejectAnswer: (error: Error) => void = () => {};
    const answer = new Promise<void>((_, reject) => {
        rejectAnswer = reject;
    });
    const remote = vi.fn().mockReturnValue(answer);
    const close = vi.fn();
    vi.stubGlobal(
        "RTCPeerConnection",
        class {
            iceGatheringState = "complete";
            localDescription = { sdp: "offer" };
            addTrack = vi.fn();
            createDataChannel = vi.fn();
            createOffer = vi
                .fn()
                .mockResolvedValue({ type: "offer", sdp: "offer" });
            setLocalDescription = vi.fn().mockResolvedValue(undefined);
            setRemoteDescription = remote;
            close = close;
        },
    );
    vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
    const send = vi.fn(() => true);
    const { rerender } = render(<Voice connected event={null} send={send} />);
    fireEvent.click(screen.getByRole("button", { name: "Start voice" }));
    await waitFor(() => expect(send).toHaveBeenCalledWith("start", "offer"));
    rerender(
        <Voice
            connected
            event={{ type: "voice", name: "answer", sdp: "answer" }}
            send={send}
        />,
    );
    expect(remote).toHaveBeenCalledOnce();
    fireEvent.click(screen.getByRole("button", { name: "Stop voice" }));
    rerender(
        <Voice
            connected
            event={{ type: "voice", name: "closed" }}
            send={send}
        />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Start voice" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(3));
    await act(async () => {
        rejectAnswer(new Error("Old answer failed"));
    });
    expect(screen.queryByRole("alert")).toBeNull();
    expect(screen.getByRole("button", { name: "Stop voice" })).toBeTruthy();
    expect(send).toHaveBeenLastCalledWith("start", "offer");
    expect(send).toHaveBeenCalledTimes(3);
    expect(close).toHaveBeenCalledOnce();
    expect(tracks[0].stop).toHaveBeenCalledOnce();
    expect(tracks[1].stop).not.toHaveBeenCalled();
});
