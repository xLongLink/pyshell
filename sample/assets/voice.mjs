export function mountVoice(page = window) {
  const button = page.document.getElementById("voice");
  const voiceError = page.document.getElementById("voice-error");
  const workspaceError = page.document.getElementById("workspace-error");
  const orb = page.document.getElementById("voice-orb");
  let socket;
  let reconnect;
  let connected = false;
  let phase = "off";
  let resources = {};
  let suspended = false;

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
    if (orb) {
      orb.state = !connected
        ? "disabled"
        : value === "off"
          ? "idle"
          : value === "on"
            ? orb.level > 0.02 ? "speaking" : "listening"
            : "connecting";
    }
  }

  function send(action, sdp) {
    if (!connected || socket?.readyState !== page.WebSocket.OPEN) return false;
    socket.send(
      JSON.stringify({ type: "voice", action, ...(sdp ? { sdp } : {}) }),
    );
    return true;
  }

  function setLevel(level) {
    if (!orb) return;
    orb.level = level;
    if (phase === "on") orb.state = level > 0.02 ? "speaking" : "listening";
  }

  function animateSpeech(current, stream) {
    if (!current.audioContext) return;
    try {
      current.source?.disconnect();
      page.cancelAnimationFrame(current.frame);
      const analyser = current.audioContext.createAnalyser();
      analyser.fftSize = 256;
      current.source = current.audioContext.createMediaStreamSource(stream);
      current.source.connect(analyser);
      const samples = new Float32Array(analyser.fftSize);

      function draw() {
        if (current !== resources) return;
        analyser.getFloatTimeDomainData(samples);
        const rms = Math.sqrt(
          samples.reduce((sum, sample) => sum + sample * sample, 0) /
            samples.length,
        );
        setLevel(Math.min(1, Math.max(0, rms - 0.005) * 8));
        current.frame = page.requestAnimationFrame(draw);
      }

      draw();
    } catch (failure) {
      page.console?.warn("Voice visualization unavailable.", failure);
    }
  }

  function stop(closed = false) {
    const current = resources;
    resources = {};
    page.cancelAnimationFrame(current.frame);
    current.source?.disconnect();
    void current.audioContext?.close().catch((failure) => {
      page.console?.warn("Unable to close voice visualization.", failure);
    });
    setLevel(0);
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

  function failVoice(text) {
    error(voiceError, text);
    stop();
    if (orb) orb.state = "error";
  }

  function waitForIce(current) {
    const { peer } = current;
    return new Promise((resolve, reject) => {
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
      if (page.AudioContext) {
        current.audioContext = new page.AudioContext();
        void current.audioContext.resume().catch((failure) => {
          page.console?.warn("Voice visualization unavailable.", failure);
        });
      }
    } catch (failure) {
      page.console?.warn("Voice visualization unavailable.", failure);
    }
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
        animateSpeech(current, audio.srcObject);
        void audio.play().catch(() => {
          if (current !== resources) return;
          failVoice("Audio playback was blocked. Restart voice to try again.");
        });
      };
      peer.onconnectionstatechange = () => {
        if (current !== resources) return;
        if (peer.connectionState === "connected") {
          page.clearTimeout(current.timer);
          setPhase("on");
        } else if (["failed", "disconnected"].includes(peer.connectionState)) {
          failVoice("Voice connection lost.");
        }
      };
      stream.getTracks().forEach((track) => peer.addTrack(track, stream));
      peer.createDataChannel("oai-events");
      await peer.setLocalDescription(await peer.createOffer());
      await waitForIce(current);
      if (current !== resources) return;
      const sdp = peer.localDescription?.sdp;
      if (!sdp || !send("start", sdp))
        throw new Error("The workspace disconnected before voice could start.");
      current.submitted = true;
      if (peer.connectionState !== "connected") {
        current.timer = page.setTimeout(() => {
          if (current !== resources) return;
          failVoice("Voice connection timed out.");
        }, 90000);
      }
    } catch (failure) {
      if (current !== resources) return;
      failVoice(
        failure instanceof Error ? failure.message : "Unable to start voice.",
      );
    }
  }

  function receive(event) {
    if (event?.type === "snapshot") {
      connected = true;
      error(workspaceError, "");
      setPhase(phase);
    } else if (event?.type === "instruction") {
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
        if (orb) orb.state = "error";
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
            failVoice("Unable to connect voice audio.");
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
      if (suspended || socket !== ws) return;
      try {
        receive(JSON.parse(message.data));
      } catch {
        error(workspaceError, "The API sent an invalid workspace event.");
      }
    };
    ws.onclose = () => {
      if (suspended || socket !== ws) return;
      connected = false;
      stop(true);
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
    if (!suspended) return;
    suspended = false;
    connect();
  }

  button.onclick = start;
  page.addEventListener("pagehide", suspend);
  page.addEventListener("pageshow", resume);
  setPhase("off");
  connect();
}
