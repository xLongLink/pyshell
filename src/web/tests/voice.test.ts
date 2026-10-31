import { afterEach, beforeEach, expect, test, vi } from "vitest";

// @ts-expect-error The sample browser module has no TypeScript declarations.
import { mountVoice } from "../../../sample/assets/voice.mjs";

function deferred<T>() {
    let resolve!: (value: T) => void;
    let reject!: (reason: Error) => void;
    const promise = new Promise<T>((yes, no) => {
        resolve = yes;
        reject = no;
    });
    return { promise, resolve, reject };
}

function setup({ gatherIce = false } = {}) {
    const sockets: Socket[] = [];
    const peers: Peer[] = [];
    const audios: Audio[] = [];
    const track = { stop: vi.fn() };
    const stream = { getTracks: () => [track] };
    const getUserMedia = vi.fn().mockResolvedValue(stream);

    class Socket {
        static OPEN = 1;
        readyState = 1;
        onmessage?: (event: { data: string }) => void;
        onclose?: () => void;
        send = vi.fn();
        close = vi.fn(() => {
            this.readyState = 3;
            this.onclose?.();
        });
        constructor(public url: string) {
            sockets.push(this);
        }
        receive(event: object) {
            this.onmessage?.({ data: JSON.stringify(event) });
        }
    }

    class Peer extends EventTarget {
        iceGatheringState = gatherIce ? "gathering" : "complete";
        connectionState = "new";
        localDescription = { type: "offer", sdp: "local-offer" };
        onconnectionstatechange?: () => void;
        ontrack?: (event: { streams: object[]; track?: object }) => void;
        addTrack = vi.fn();
        createDataChannel = vi.fn();
        createOffer = vi.fn().mockResolvedValue(this.localDescription);
        setLocalDescription = vi.fn().mockResolvedValue(undefined);
        setRemoteDescription = vi.fn().mockResolvedValue(undefined);
        close = vi.fn();
        constructor() {
            super();
            peers.push(this);
        }
        finishIce() {
            this.iceGatheringState = "complete";
            this.dispatchEvent(new Event("icegatheringstatechange"));
        }
    }

    class Audio {
        autoplay = false;
        srcObject: object | null = null;
        play = vi.fn().mockResolvedValue(undefined);
        pause = vi.fn();
        constructor() {
            audios.push(this);
        }
    }

    const document = window.document.implementation.createHTMLDocument();
    document.body.innerHTML = `
        <button id="voice"></button>
        <p id="voice-error" hidden></p>
        <p id="workspace-error" hidden></p>
    `;
    const page = Object.assign(new EventTarget(), {
        document,
        location: { protocol: "https:", host: "workspace.test" },
        navigator: { mediaDevices: { getUserMedia } },
        WebSocket: Socket,
        RTCPeerConnection: Peer,
        Audio,
        setTimeout: window.setTimeout.bind(window),
        clearTimeout: window.clearTimeout.bind(window),
    });
    const button = document.getElementById("voice") as HTMLButtonElement;
    const voiceError = document.getElementById("voice-error")!;
    const workspaceError = document.getElementById("workspace-error")!;
    mountVoice(page);
    const lifecycle = (type: string) =>
        page.dispatchEvent(new PageTransitionEvent(type, { persisted: true }));
    const connect = () => sockets.at(-1)!.receive({ type: "snapshot" });
    return {
        page,
        button,
        voiceError,
        workspaceError,
        sockets,
        peers,
        audios,
        track,
        stream,
        getUserMedia,
        lifecycle,
        connect,
    };
}

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

test("instruction events clear workspace errors without instruction text", () => {
    const app = setup();
    app.connect();
    app.sockets[0].receive({ type: "error", text: "Earlier task failed" });
    expect(app.workspaceError.hidden).toBe(false);
    app.sockets[0].receive({ type: "instruction" });
    expect(app.workspaceError.hidden).toBe(true);
    expect(app.workspaceError.textContent).toBe("");
});

test("BFCache suspension releases voice resources and resumes only once", async () => {
    const app = setup();
    expect(app.sockets[0].url).toBe("wss://workspace.test/ws");
    expect(app.button.disabled).toBe(true);
    app.lifecycle("pageshow");
    expect(app.sockets).toHaveLength(1);
    app.connect();
    app.button.click();
    await vi.advanceTimersByTimeAsync(0);
    const peer = app.peers[0];
    peer.ontrack?.({ streams: [app.stream] });
    peer.connectionState = "connected";
    peer.onconnectionstatechange?.();
    expect(app.button.dataset.state).toBe("on");

    app.lifecycle("pagehide");
    expect(app.track.stop).toHaveBeenCalledOnce();
    expect(peer.close).toHaveBeenCalledOnce();
    expect(app.audios[0].pause).toHaveBeenCalledOnce();
    expect(app.audios[0].srcObject).toBeNull();
    expect(app.sockets[0].close).toHaveBeenCalledOnce();
    expect(app.button.dataset.state).toBe("off");
    expect(app.button.disabled).toBe(true);
    expect(app.sockets[0].send).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(100000);
    expect(app.sockets).toHaveLength(1);

    app.lifecycle("pageshow");
    app.lifecycle("pageshow");
    expect(app.sockets).toHaveLength(2);
    expect(app.button.disabled).toBe(true);
    app.connect();
    expect(app.button.disabled).toBe(false);
    expect(app.workspaceError.hidden).toBe(true);
});

test("disconnects retry, suspension cancels retries, and old sockets stay ignored", async () => {
    const app = setup();
    app.connect();
    app.sockets[0].close();
    expect(app.workspaceError.textContent).toContain("Reconnecting");
    await vi.advanceTimersByTimeAsync(1499);
    expect(app.sockets).toHaveLength(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(app.sockets).toHaveLength(2);
    app.sockets[0].receive({ type: "snapshot" });
    expect(app.button.disabled).toBe(true);
    app.connect();
    app.sockets[0].receive({ type: "error", text: "stale" });
    app.sockets[0].onclose?.();
    expect(app.button.disabled).toBe(false);
    expect(app.workspaceError.hidden).toBe(true);
    await vi.advanceTimersByTimeAsync(1500);
    expect(app.sockets).toHaveLength(2);

    app.sockets[1].close();
    app.lifecycle("pagehide");
    app.sockets[1].receive({ type: "snapshot" });
    await vi.advanceTimersByTimeAsync(1500);
    expect(app.button.disabled).toBe(true);
    expect(app.sockets).toHaveLength(2);
    app.lifecycle("pageshow");
    expect(app.sockets).toHaveLength(3);
});

test("stopping during microphone permission releases the late stream without affecting a retry", async () => {
    const app = setup();
    const permission = deferred<typeof app.stream>();
    const lateTrack = { stop: vi.fn() };
    app.getUserMedia.mockReturnValueOnce(permission.promise);
    app.connect();
    app.button.click();
    expect(app.button.dataset.state).toBe("connecting");
    app.button.click();
    expect(app.button.dataset.state).toBe("off");
    app.button.click();
    await vi.advanceTimersByTimeAsync(0);
    permission.resolve({ getTracks: () => [lateTrack] });
    await vi.advanceTimersByTimeAsync(0);
    expect(lateTrack.stop).toHaveBeenCalledOnce();
    expect(app.track.stop).not.toHaveBeenCalled();
    expect(app.peers).toHaveLength(1);
    expect(app.button.dataset.state).toBe("connecting");
    expect(app.sockets[0].send).toHaveBeenCalledExactlyOnceWith(
        JSON.stringify({ type: "voice", action: "start", sdp: "local-offer" }),
    );
});

test.each(["stop", "pagehide"])(
    "%s cancels pending ICE without a timeout or stale offer",
    async (action) => {
        const app = setup({ gatherIce: true });
        app.connect();
        app.button.click();
        await vi.advanceTimersByTimeAsync(0);
        expect(vi.getTimerCount()).toBe(1);
        if (action === "stop") app.button.click();
        else app.lifecycle("pagehide");
        await vi.advanceTimersByTimeAsync(0);
        expect(vi.getTimerCount()).toBe(0);
        app.peers[0].finishIce();
        await vi.advanceTimersByTimeAsync(10000);
        expect(app.sockets[0].send).not.toHaveBeenCalled();
        expect(app.track.stop).toHaveBeenCalledOnce();
        expect(app.peers[0].close).toHaveBeenCalledOnce();
        expect(app.button.dataset.state).toBe("off");
        expect(app.voiceError.hidden).toBe(true);
    },
);

test("stale peer and answer callbacks cannot stop a new voice session", async () => {
    const app = setup();
    app.connect();
    app.button.click();
    await vi.advanceTimersByTimeAsync(0);
    const oldPeer = app.peers[0];
    const answer = deferred<void>();
    oldPeer.setRemoteDescription.mockReturnValueOnce(answer.promise);
    app.sockets[0].receive({ type: "voice", name: "answer", sdp: "answer" });
    app.button.click();
    expect(app.button.dataset.state).toBe("stopping");
    expect(app.button.disabled).toBe(true);
    app.sockets[0].receive({ type: "voice", name: "closed" });
    app.button.click();
    await vi.advanceTimersByTimeAsync(0);
    oldPeer.connectionState = "failed";
    oldPeer.onconnectionstatechange?.();
    oldPeer.ontrack?.({ streams: [app.stream] });
    answer.reject(new Error("late answer failure"));
    await vi.advanceTimersByTimeAsync(0);
    expect(app.peers[1].close).not.toHaveBeenCalled();
    expect(app.audios[0].play).not.toHaveBeenCalled();
    expect(app.button.dataset.state).toBe("connecting");
    expect(app.voiceError.hidden).toBe(true);
});

test("ICE timeout reports failure and allows a successful retry", async () => {
    const app = setup({ gatherIce: true });
    app.connect();
    app.button.click();
    await vi.advanceTimersByTimeAsync(10000);
    expect(app.voiceError.textContent).toBe("Voice network setup timed out.");
    expect(app.button.dataset.state).toBe("off");
    app.button.click();
    await vi.advanceTimersByTimeAsync(0);
    app.peers[1].finishIce();
    await vi.advanceTimersByTimeAsync(0);
    expect(app.voiceError.hidden).toBe(true);
    expect(app.sockets[0].send).toHaveBeenCalledOnce();
});
