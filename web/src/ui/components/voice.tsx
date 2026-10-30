import { useCallback, useEffect, useRef, useState } from "react";
import type { VoiceEvent } from "@/lib/workspace";

export function Voice({
    connected,
    event,
    send,
}: {
    connected: boolean;
    event: VoiceEvent | null;
    send: (action: "start" | "stop", sdp?: string) => boolean;
}) {
    const [phase, setPhase] = useState<
        "off" | "connecting" | "on" | "stopping"
    >("off");
    const [error, setError] = useState("");
    const generation = useRef(0);
    const resources = useRef<{
        peer?: RTCPeerConnection;
        stream?: MediaStream;
        audio?: HTMLAudioElement;
        timer?: ReturnType<typeof setTimeout>;
        submitted?: boolean;
        cancelIce?: () => void;
    }>({});

    const stop = useCallback(
        (closed = false) => {
            generation.current++;
            const current = resources.current;
            resources.current = {};
            clearTimeout(current.timer);
            current.cancelIce?.();
            current.peer?.close();
            current.stream?.getTracks().forEach((track) => track.stop());
            if (current.audio) {
                current.audio.pause();
                current.audio.srcObject = null;
            }
            const stopping = !closed && current.submitted && send("stop");
            setPhase(stopping ? "stopping" : "off");
        },
        [send],
    );

    useEffect(() => {
        if (!connected) stop(true);
        return () => stop();
    }, [connected, stop]);

    useEffect(() => {
        if (!event) return;
        if (event.name === "error") {
            setError(event.text);
            stop(true);
            setPhase("stopping");
        } else if (event.name === "closed") {
            stop(true);
        } else {
            const id = generation.current;
            void resources.current.peer
                ?.setRemoteDescription({ type: "answer", sdp: event.sdp })
                .catch(() => {
                    if (id === generation.current) {
                        setError("Unable to connect voice audio.");
                        stop();
                    }
                });
        }
    }, [event, stop]);

    async function start() {
        if (phase !== "off") {
            stop();
            return;
        }
        if (!connected) return;
        setPhase("connecting");
        setError("");
        const id = ++generation.current;
        const current = resources.current;
        try {
            if (!navigator.mediaDevices?.getUserMedia)
                throw new Error(
                    "Voice requires localhost or HTTPS and microphone access.",
                );
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { echoCancellation: true, noiseSuppression: true },
            });
            if (id !== generation.current) {
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
                if (id !== generation.current) return;
                audio.srcObject = streams[0] ?? new MediaStream([track]);
                void audio.play().catch(() => {
                    if (id === generation.current) {
                        setError(
                            "Audio playback was blocked. Restart voice to try again.",
                        );
                        stop();
                    }
                });
            };
            peer.onconnectionstatechange = () => {
                if (id !== generation.current) return;
                if (peer.connectionState === "connected") {
                    clearTimeout(current.timer);
                    setPhase("on");
                }
                if (["failed", "disconnected"].includes(peer.connectionState)) {
                    setError("Voice connection lost.");
                    stop();
                }
            };
            stream.getTracks().forEach((track) => peer.addTrack(track, stream));
            peer.createDataChannel("oai-events");
            await peer.setLocalDescription(await peer.createOffer());
            await new Promise<void>((resolve, reject) => {
                if (peer.iceGatheringState === "complete") {
                    resolve();
                    return;
                }
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
            });
            if (id !== generation.current) return;
            const sdp = peer.localDescription?.sdp;
            if (!sdp || !send("start", sdp))
                throw new Error(
                    "The workspace disconnected before voice could start.",
                );
            current.submitted = true;
            current.timer = setTimeout(() => {
                setError("Voice connection timed out.");
                stop();
            }, 90000);
        } catch (failure) {
            if (id !== generation.current) return;
            setError(
                failure instanceof Error
                    ? failure.message
                    : "Unable to start voice.",
            );
            stop();
        }
    }

    return (
        <div className="fixed right-5 bottom-5 z-20 flex max-w-xs flex-row-reverse items-center gap-3">
            <button
                type="button"
                className="voice-orb relative grid size-18 shrink-0 cursor-pointer place-items-center rounded-full border border-[#ffd45c] focus-visible:outline-3 focus-visible:outline-offset-8 focus-visible:outline-primary disabled:cursor-wait disabled:opacity-50 motion-reduce:transition-none"
                data-state={phase}
                disabled={!connected || phase === "stopping"}
                aria-label={
                    phase === "off"
                        ? "Start voice"
                        : phase === "stopping"
                          ? "Stopping voice"
                          : "Stop voice"
                }
                aria-pressed={phase !== "off"}
                onClick={() => void start()}
            >
                <span
                    className={`pointer-events-none absolute -inset-[9%] rounded-[46%_54%_51%_49%] border-2 border-[#ffd45c80] ${phase === "on" ? "voice-ring-active motion-reduce:animate-none" : ""}`}
                    aria-hidden="true"
                />
            </button>
            {error ? (
                <p role="alert" className="text-sm text-destructive">
                    {error}
                </p>
            ) : (
                phase !== "off" && (
                    <p role="status" className="text-sm text-muted-foreground">
                        {phase === "on"
                            ? "Listening"
                            : phase === "stopping"
                              ? "Stopping…"
                              : "Connecting voice…"}
                    </p>
                )
            )}
        </div>
    );
}
