import assert from "node:assert/strict";
import { test } from "node:test";
import { mountVoice } from "../assets/voice.mjs";

function browser(t) {
  const elements = Object.fromEntries(
    ["voice", "voice-error", "workspace-error"].map((id) => [
      id,
      {
        dataset: {},
        attributes: {},
        textContent: "",
        hidden: true,
        setAttribute(name, value) {
          this.attributes[name] = value;
        },
      },
    ]),
  );
  const sockets = [];
  const peers = [];
  const tracks = [];
  const audios = [];
  const sent = [];
  const timers = new Map();
  const listeners = new Map();
  let timerId = 0;
  const page = {
    document: { getElementById: (id) => elements[id] },
    location: { protocol: "http:", host: "localhost:8010" },
    setTimeout(callback, delay) {
      timers.set(++timerId, { callback, delay });
      return timerId;
    },
    clearTimeout(id) {
      timers.delete(id);
    },
    addEventListener(name, handler) {
      listeners.set(name, handler);
    },
    removeEventListener(name) {
      listeners.delete(name);
    },
    navigator: {
      mediaDevices: {
        async getUserMedia() {
          const track = {
            stops: 0,
            stop() {
              this.stops++;
            },
          };
          tracks.push(track);
          return { getTracks: () => [track] };
        },
      },
    },
    WebSocket: class {
      static OPEN = 1;
      readyState = 1;
      constructor(url) {
        this.url = url;
        sockets.push(this);
      }
      send(text) {
        sent.push(JSON.parse(text));
      }
      close() {
        this.readyState = 3;
        this.onclose?.();
      }
      receive(event) {
        this.onmessage?.({ data: JSON.stringify(event) });
      }
    },
    RTCPeerConnection: class {
      iceGatheringState = "complete";
      connectionState = "new";
      closed = 0;
      answers = [];
      events = new Map();
      constructor() {
        peers.push(this);
      }
      addTrack() {}
      createDataChannel() {}
      async createOffer() {
        return { type: "offer", sdp: "offer" };
      }
      async setLocalDescription(description) {
        this.localDescription = description;
      }
      async setRemoteDescription(answer) {
        this.answers.push(answer);
      }
      addEventListener(name, handler) {
        this.events.set(name, handler);
      }
      removeEventListener(name) {
        this.events.delete(name);
      }
      close() {
        this.closed++;
      }
    },
    Audio: class {
      pauses = 0;
      constructor() {
        audios.push(this);
      }
      async play() {}
      pause() {
        this.pauses++;
      }
    },
  };
  const dispose = mountVoice(page);
  t.after(dispose);
  return {
    page,
    button: elements.voice,
    voiceError: elements["voice-error"],
    workspaceError: elements["workspace-error"],
    sockets,
    peers,
    tracks,
    audios,
    sent,
    timers,
    listeners,
    dispose,
  };
}

function connected(t) {
  const state = browser(t);
  state.sockets[0].receive({ type: "snapshot" });
  return state;
}

test("voice negotiates audio and releases resources while waiting for hangup confirmation", async (t) => {
  const state = connected(t);
  await state.button.onclick();
  assert.deepEqual(state.sent, [
    { type: "voice", action: "start", sdp: "offer" },
  ]);
  state.sockets[0].receive({ type: "voice", name: "answer", sdp: "answer" });
  assert.deepEqual(state.peers[0].answers, [{ type: "answer", sdp: "answer" }]);
  state.peers[0].connectionState = "connected";
  state.peers[0].onconnectionstatechange();
  assert.equal(state.button.dataset.state, "on");
  assert.equal(state.timers.size, 0);
  const stream = { audio: true };
  state.peers[0].ontrack({ streams: [stream] });
  assert.equal(state.audios[0].srcObject, stream);
  await state.button.onclick();
  assert.deepEqual(state.sent.at(-1), { type: "voice", action: "stop" });
  assert.equal(state.tracks[0].stops, 1);
  assert.equal(state.peers[0].closed, 1);
  assert.equal(state.audios[0].pauses, 1);
  assert.equal(state.audios[0].srcObject, null);
  assert.equal(state.button.disabled, true);
  assert.equal(state.button.attributes["aria-label"], "Stopping voice");
  state.sockets[0].receive({ type: "voice", name: "closed" });
  assert.equal(state.button.disabled, false);
  assert.equal(state.button.attributes["aria-label"], "Start voice");
});

test("cancelling pending microphone permission never submits a late call", async (t) => {
  const state = connected(t);
  let grant;
  state.page.navigator.mediaDevices.getUserMedia = () =>
    new Promise((resolve) => {
      grant = resolve;
    });
  const pending = state.button.onclick();
  await state.button.onclick();
  const track = {
    stops: 0,
    stop() {
      this.stops++;
    },
  };
  grant({ getTracks: () => [track] });
  await pending;
  assert.equal(track.stops, 1);
  assert.deepEqual(state.sent, []);
  assert.equal(state.peers.length, 0);
  assert.equal(state.button.dataset.state, "off");
});

test("microphone failure is visible and allows retry", async (t) => {
  const state = connected(t);
  state.page.navigator.mediaDevices.getUserMedia = async () => {
    throw new Error("Microphone access denied");
  };
  await state.button.onclick();
  assert.equal(state.voiceError.textContent, "Microphone access denied");
  assert.equal(state.voiceError.hidden, false);
  assert.equal(state.button.disabled, false);
  assert.equal(state.button.dataset.state, "off");
  assert.deepEqual(state.sent, []);
});

test("a stale answer failure cannot stop a restarted voice attempt", async (t) => {
  const state = connected(t);
  await state.button.onclick();
  let reject;
  state.peers[0].setRemoteDescription = () =>
    new Promise((_, fail) => {
      reject = fail;
    });
  state.sockets[0].receive({
    type: "voice",
    name: "answer",
    sdp: "old answer",
  });
  await state.button.onclick();
  state.sockets[0].receive({ type: "voice", name: "closed" });
  await state.button.onclick();
  reject(new Error("Old answer failed"));
  await Promise.resolve();
  assert.equal(state.voiceError.hidden, true);
  assert.equal(state.button.dataset.state, "connecting");
  assert.equal(state.tracks[0].stops, 1);
  assert.equal(state.tracks[1].stops, 0);
  assert.deepEqual(
    state.sent.map((event) => event.action),
    ["start", "stop", "start"],
  );
});

test("disconnect releases audio and reconnects without replaying commands", async (t) => {
  const state = connected(t);
  await state.button.onclick();
  state.sockets[0].close();
  assert.equal(state.tracks[0].stops, 1);
  assert.equal(state.button.disabled, true);
  assert.match(
    state.workspaceError.textContent,
    /submitted work may still be running/,
  );
  const reconnect = [...state.timers.values()].find(
    (timer) => timer.delay === 1500,
  );
  reconnect.callback();
  assert.equal(state.sockets.length, 2);
  state.sockets[1].receive({ type: "snapshot" });
  state.sockets[0].receive({ type: "error", text: "stale" });
  assert.equal(state.workspaceError.hidden, true);
  assert.equal(state.button.disabled, false);
  assert.deepEqual(
    state.sent.map((event) => event.action),
    ["start"],
  );
});

test("invalid events do not enable voice and task errors remain plain text", (t) => {
  const state = browser(t);
  assert.equal(state.button.disabled, true);
  state.sockets[0].receive({ type: "voice", name: "answer", sdp: 42 });
  assert.match(state.workspaceError.textContent, /invalid workspace event/);
  assert.equal(state.button.disabled, true);
  state.sockets[0].receive({ type: "snapshot" });
  state.sockets[0].receive({ type: "error", text: "<script>bad()</script>" });
  assert.equal(state.workspaceError.textContent, "<script>bad()</script>");
  state.sockets[0].receive({ type: "instruction", text: "Next task" });
  assert.equal(state.workspaceError.hidden, true);
});

test("provider errors wait for close confirmation before allowing another call", async (t) => {
  const state = connected(t);
  await state.button.onclick();
  state.sockets[0].receive({
    type: "voice",
    name: "error",
    text: "Provider unavailable",
  });
  assert.equal(state.voiceError.textContent, "Provider unavailable");
  assert.equal(state.tracks[0].stops, 1);
  assert.equal(state.button.disabled, true);
  state.sockets[0].receive({ type: "voice", name: "closed" });
  assert.equal(state.button.disabled, false);
});

test("negotiation timeout releases resources and requests server hangup", async (t) => {
  const state = connected(t);
  await state.button.onclick();
  const timeout = [...state.timers.values()].find(
    (timer) => timer.delay === 90000,
  );
  timeout.callback();
  assert.equal(state.voiceError.textContent, "Voice connection timed out.");
  assert.equal(state.tracks[0].stops, 1);
  assert.equal(state.peers[0].closed, 1);
  assert.equal(state.button.dataset.state, "stopping");
  assert.deepEqual(state.sent.at(-1), { type: "voice", action: "stop" });
});

test("page suspension closes voice and restoration reconnects without replay", async (t) => {
  const state = connected(t);
  await state.button.onclick();
  state.listeners.get("pagehide")();
  assert.equal(state.tracks[0].stops, 1);
  assert.equal(state.button.disabled, true);
  state.listeners.get("pageshow")();
  assert.equal(state.sockets.length, 2);
  state.sockets[1].receive({ type: "snapshot" });
  assert.equal(state.button.disabled, false);
  assert.equal(state.sent.length, 1);
  state.dispose();
  assert.equal(state.listeners.size, 0);
  assert.equal(state.button.onclick, null);
});
