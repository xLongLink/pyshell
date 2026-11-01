export function mountVoice() {
  const button = document.getElementById("voice");
  const voiceError = document.getElementById("voice-error");
  const workspaceError = document.getElementById("workspace-error");
  const orb = document.getElementById("voice-orb");
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

  function speechState() {
    return orb.level > 0.02 ? "speaking" : "listening";
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
    orb.state = !connected
      ? "disabled"
      : value === "off"
        ? "idle"
        : value === "on"
          ? speechState()
          : "connecting";
  }

  function send(action, sdp) {
    if (!connected || socket?.readyState !== WebSocket.OPEN) return false;
    socket.send(
      JSON.stringify({ type: "voice", action, ...(sdp ? { sdp } : {}) }),
    );
    return true;
  }

  function setLevel(level) {
    orb.level = level;
    if (phase === "on") orb.state = speechState();
  }

  function animateSpeech(current, stream) {
    if (!current.audioContext) return;
    try {
      current.source?.disconnect();
      cancelAnimationFrame(current.frame);
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
        current.frame = requestAnimationFrame(draw);
      }

      draw();
    } catch (failure) {
      console.warn("Voice visualization unavailable.", failure);
    }
  }

  function stop(closed = false) {
    const current = resources;
    if (!closed && current.submitted && send("stop")) {
      current.stream?.getTracks().forEach((track) => {
        track.enabled = false;
      });
      current.audio?.pause();
      clearTimeout(current.timer);
      current.timer = setTimeout(() => stop(true), 20000);
      setPhase("stopping");
      return;
    }
    resources = {};
    cancelAnimationFrame(current.frame);
    current.source?.disconnect();
    void current.audioContext?.close().catch((failure) => {
      console.warn("Unable to close voice visualization.", failure);
    });
    setLevel(0);
    clearTimeout(current.timer);
    current.cancelIce?.();
    current.peer?.close();
    current.stream?.getTracks().forEach((track) => track.stop());
    if (current.audio) {
      current.audio.pause();
      current.audio.srcObject = null;
    }
    setPhase("off");
  }

  function failVoice(text) {
    error(voiceError, text);
    stop();
    orb.state = "error";
  }

  function waitForIce(current) {
    const { peer } = current;
    return new Promise((resolve, reject) => {
      function cleanup() {
        clearTimeout(timer);
        peer.removeEventListener("icegatheringstatechange", change);
        current.cancelIce = undefined;
      }
      function change() {
        if (peer.iceGatheringState === "complete") {
          cleanup();
          resolve();
        }
      }
      const timer = setTimeout(() => {
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
      if (window.AudioContext) {
        current.audioContext = new AudioContext();
        void current.audioContext.resume().catch((failure) => {
          console.warn("Voice visualization unavailable.", failure);
        });
      }
    } catch (failure) {
      console.warn("Voice visualization unavailable.", failure);
    }
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error(
          "Voice requires localhost or HTTPS and microphone access.",
        );
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      if (current !== resources) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      current.stream = stream;
      const peer = new RTCPeerConnection();
      current.peer = peer;
      const audio = new Audio();
      audio.autoplay = true;
      current.audio = audio;
      peer.ontrack = ({ streams, track }) => {
        if (current !== resources || phase === "stopping") return;
        audio.srcObject = streams[0] ?? new MediaStream([track]);
        animateSpeech(current, audio.srcObject);
        void audio.play().catch(() => {
          if (current !== resources) return;
          failVoice("Audio playback was blocked. Restart voice to try again.");
        });
      };
      peer.onconnectionstatechange = () => {
        if (current !== resources || phase === "stopping") return;
        if (peer.connectionState === "connected") {
          if (current.started) {
            clearTimeout(current.timer);
            setPhase("on");
          }
        } else if (["failed", "disconnected"].includes(peer.connectionState)) {
          failVoice("Voice connection lost.");
        }
      };
      stream.getTracks().forEach((track) => peer.addTrack(track, stream));
      const events = peer.createDataChannel("oai-events");
      events.onmessage = ({ data }) => {
        if (current !== resources) return;
        try {
          const event = JSON.parse(data);
          if (event.type === "session.started") {
            current.started = true;
            if (peer.connectionState === "connected" && phase !== "stopping") {
              clearTimeout(current.timer);
              setPhase("on");
            }
          } else if (event.type === "session.closed") {
            stop(true);
          }
        } catch (failure) {
          console.warn("Invalid Live event.", failure);
        }
      };
      await peer.setLocalDescription(await peer.createOffer());
      await waitForIce(current);
      if (current !== resources) return;
      const sdp = peer.localDescription?.sdp;
      if (!sdp || !send("start", sdp))
        throw new Error("The workspace disconnected before voice could start.");
      current.submitted = true;
      if (!current.started || peer.connectionState !== "connected") {
        current.timer = setTimeout(() => {
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
        orb.state = "error";
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
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${protocol}//${location.host}/ws`);
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
      reconnect = setTimeout(connect, 1500);
    };
  }

  function suspend() {
    suspended = true;
    connected = false;
    clearTimeout(reconnect);
    stop(true);
    socket?.close();
  }

  function resume() {
    if (!suspended) return;
    suspended = false;
    connect();
  }

  button.onclick = start;
  window.addEventListener("pagehide", suspend);
  window.addEventListener("pageshow", resume);
  setPhase("off");
  connect();
}
