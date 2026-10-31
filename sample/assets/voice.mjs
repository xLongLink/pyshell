export function mountVoice(page = window) {
  const button = page.document.getElementById("voice");
  const voiceError = page.document.getElementById("voice-error");
  const workspaceError = page.document.getElementById("workspace-error");
  let socket;
  let reconnect;
  let connected = false;
  let phase = "off";
  let resources = {};
  let suspended = false;
  let disposed = false;

  function error(element, text) {
    element.textContent = text;
    element.hidden = !text;
  }

  function setPhase(value) {
    phase = value;
    button.dataset.state = value;
    button.disabled = !connected || value === "stopping";
    button.setAttribute("aria-pressed", String(value !== "off"));
    button.setAttribute(
      "aria-label",
      value === "off"
        ? "Start voice"
        : value === "stopping"
          ? "Stopping voice"
          : "Stop voice",
    );
  }

  function send(action, sdp) {
    if (!connected || socket?.readyState !== page.WebSocket.OPEN) return false;
    socket.send(
      JSON.stringify({ type: "voice", action, ...(sdp ? { sdp } : {}) }),
    );
    return true;
  }

  function stop(closed = false) {
    const current = resources;
    resources = {};
    page.clearTimeout(current.timer);
    current.cancelIce?.();
    current.peer?.close();
    current.stream?.getTracks().forEach((track) => track.stop());
    if (current.audio) {
      current.audio.pause();
      current.audio.srcObject = null;
    }
    setPhase(!closed && current.submitted && send("stop") ? "stopping" : "off");
  }

  async function start() {
    if (phase !== "off") {
      stop();
      return;
    }
    if (!connected) return;
    setPhase("connecting");
    error(voiceError, "");
    const current = {};
    resources = current;
    try {
      if (!page.navigator.mediaDevices?.getUserMedia)
        throw new Error(
          "Voice requires localhost or HTTPS and microphone access.",
        );
      const stream = await page.navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      if (current !== resources) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      current.stream = stream;
      const peer = new page.RTCPeerConnection();
      current.peer = peer;
      const audio = new page.Audio();
      audio.autoplay = true;
      current.audio = audio;
      peer.ontrack = ({ streams, track }) => {
        if (current !== resources) return;
        audio.srcObject = streams[0] ?? new page.MediaStream([track]);
        void audio.play().catch(() => {
          if (current !== resources) return;
          error(
            voiceError,
            "Audio playback was blocked. Restart voice to try again.",
          );
          stop();
        });
      };
      peer.onconnectionstatechange = () => {
        if (current !== resources) return;
        if (peer.connectionState === "connected") {
          page.clearTimeout(current.timer);
          setPhase("on");
        }
        if (["failed", "disconnected"].includes(peer.connectionState)) {
          error(voiceError, "Voice connection lost.");
          stop();
        }
      };
      stream.getTracks().forEach((track) => peer.addTrack(track, stream));
      peer.createDataChannel("oai-events");
      await peer.setLocalDescription(await peer.createOffer());
      await new Promise((resolve, reject) => {
        if (peer.iceGatheringState === "complete") {
          resolve();
          return;
        }
        function cleanup() {
          page.clearTimeout(timer);
          peer.removeEventListener("icegatheringstatechange", change);
          current.cancelIce = undefined;
        }
        function change() {
          if (peer.iceGatheringState === "complete") {
            cleanup();
            resolve();
          }
        }
        const timer = page.setTimeout(() => {
          cleanup();
          reject(new Error("Voice network setup timed out."));
        }, 10000);
        current.cancelIce = () => {
          cleanup();
          reject(new Error("Voice setup cancelled."));
        };
        peer.addEventListener("icegatheringstatechange", change);
        change();
      });
      if (current !== resources) return;
      const sdp = peer.localDescription?.sdp;
      if (!sdp || !send("start", sdp))
        throw new Error("The workspace disconnected before voice could start.");
      current.submitted = true;
      if (peer.connectionState !== "connected") {
        current.timer = page.setTimeout(() => {
          if (current !== resources) return;
          error(voiceError, "Voice connection timed out.");
          stop();
        }, 90000);
      }
    } catch (failure) {
      if (current !== resources) return;
      error(
        voiceError,
        failure instanceof Error ? failure.message : "Unable to start voice.",
      );
      stop();
    }
  }

  function receive(event) {
    if (event?.type === "snapshot") {
      connected = true;
      error(workspaceError, "");
      setPhase(phase);
    } else if (
      event?.type === "instruction" &&
      typeof event.text === "string"
    ) {
      error(workspaceError, "");
    } else if (event?.type === "error" && typeof event.text === "string") {
      error(workspaceError, event.text);
    } else if (event?.type === "voice") {
      if (event.name === "closed") {
        stop(true);
      } else if (event.name === "error" && typeof event.text === "string") {
        error(voiceError, event.text);
        stop(true);
        setPhase("stopping");
      } else if (
        event.name === "answer" &&
        typeof event.sdp === "string" &&
        event.sdp
      ) {
        const current = resources;
        void current.peer
          ?.setRemoteDescription({ type: "answer", sdp: event.sdp })
          .catch(() => {
            if (current !== resources) return;
            error(voiceError, "Unable to connect voice audio.");
            stop();
          });
      } else {
        throw new Error("Invalid voice event");
      }
    } else {
      throw new Error("Invalid workspace event");
    }
  }

  function connect() {
    const protocol = page.location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new page.WebSocket(`${protocol}//${page.location.host}/ws`);
    socket = ws;
    ws.onmessage = (message) => {
      if (disposed || suspended || socket !== ws) return;
      try {
        receive(JSON.parse(message.data));
      } catch {
        error(workspaceError, "The API sent an invalid workspace event.");
      }
    };
    ws.onclose = () => {
      if (disposed || suspended || socket !== ws) return;
      connected = false;
      stop(true);
      error(
        workspaceError,
        "Connection lost. Reconnecting; submitted work may still be running.",
      );
      reconnect = page.setTimeout(connect, 1500);
    };
  }

  function suspend() {
    suspended = true;
    connected = false;
    page.clearTimeout(reconnect);
    stop(true);
    socket?.close();
  }

  function resume() {
    if (disposed || !suspended) return;
    suspended = false;
    connect();
  }

  button.onclick = start;
  page.addEventListener("pagehide", suspend);
  page.addEventListener("pageshow", resume);
  setPhase("off");
  connect();
  return () => {
    disposed = true;
    suspend();
    button.onclick = null;
    page.removeEventListener("pagehide", suspend);
    page.removeEventListener("pageshow", resume);
  };
}
