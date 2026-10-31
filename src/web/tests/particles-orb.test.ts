import { afterEach, beforeEach, expect, test, vi } from "vitest";

// @ts-expect-error The sample browser module has no TypeScript declarations.
import { ParticlesOrb } from "../../../sample/assets/particles-orb.mjs";

type Orb = HTMLElement & { state: string; level: number };

let frames: Map<number, FrameRequestCallback>;
let motion: EventTarget & { matches: boolean };
let context: ReturnType<typeof makeContext>;

function makeContext() {
    return {
        setTransform: vi.fn(),
        clearRect: vi.fn(),
        beginPath: vi.fn(),
        moveTo: vi.fn(),
        arc: vi.fn(),
        fill: vi.fn(),
        fillStyle: "",
        globalAlpha: 1,
        globalCompositeOperation: "source-over",
    };
}

function mount(attributes: Record<string, string> = {}) {
    const orb = document.createElement("particles-orb") as Orb;
    for (const [name, value] of Object.entries(attributes))
        orb.setAttribute(name, value);
    document.body.append(orb);
    return orb;
}

function advance(now: number) {
    for (const [id, callback] of [...frames]) {
        frames.delete(id);
        callback(now);
    }
}

beforeEach(() => {
    frames = new Map();
    let id = 0;
    motion = Object.assign(new EventTarget(), { matches: false });
    context = makeContext();
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(
        context as unknown as CanvasRenderingContext2D,
    );
    vi.stubGlobal("matchMedia", () => motion);
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => {
        frames.set(++id, callback);
        return id;
    });
    vi.stubGlobal("cancelAnimationFrame", (frame: number) =>
        frames.delete(frame),
    );
    vi.stubGlobal("devicePixelRatio", 2);
});

afterEach(() => {
    document.body.replaceChildren();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

test("registers a native canvas element with configurable size and palette", () => {
    const orb = mount({
        size: "192",
        "color-from": "#ff0000",
        "color-to": "#ff0000",
    });
    expect(orb).toBeInstanceOf(ParticlesOrb);
    expect(orb.getAttribute("role")).toBe("img");
    expect(orb.getAttribute("aria-label")).toBe("Assistant orb");
    const canvas = orb.shadowRoot!.querySelector("canvas")!;
    expect(canvas.width).toBe(384);
    expect(canvas.height).toBe(384);
    expect(orb.style.width).toBe("192px");
    expect(context.fillStyle).toBe("rgb(255,0,0)");
    expect(context.arc).toHaveBeenCalled();
    expect(context.arc.mock.calls.flat().every(Number.isFinite)).toBe(true);
    orb.setAttribute("size", "96");
    expect(canvas.width).toBe(192);
    expect(orb.style.width).toBe("96px");
    expect(frames.size).toBe(1);
});

test("speech volume smoothly changes the particle positions and stays normalized", () => {
    const orb = mount({ state: "speaking" });
    advance(0);
    const resting = context.arc.mock.calls.slice(-720);
    orb.level = 5;
    expect(orb.level).toBe(1);
    for (let time = 50; time <= 1000; time += 50) advance(time);
    expect(Number(orb.style.getPropertyValue("--orb-level"))).toBeGreaterThan(
        0.99,
    );
    expect(context.arc.mock.calls.slice(-720)).not.toEqual(resting);
    orb.level = -1;
    expect(orb.level).toBe(0);
    for (let time = 1050; time <= 3000; time += 50) advance(time);
    expect(Number(orb.style.getPropertyValue("--orb-level"))).toBeLessThan(
        0.01,
    );
    expect(context.arc.mock.calls.flat().every(Number.isFinite)).toBe(true);
});

test("removal cancels animation and reconnection owns a single animation loop", () => {
    const orb = mount();
    expect(frames.size).toBe(1);
    orb.remove();
    expect(frames.size).toBe(0);
    const draws = context.clearRect.mock.calls.length;
    window.dispatchEvent(new Event("pageshow"));
    motion.dispatchEvent(new Event("change"));
    expect(context.clearRect).toHaveBeenCalledTimes(draws);
    expect(frames.size).toBe(0);
    document.body.append(orb);
    window.dispatchEvent(new Event("pageshow"));
    expect(frames.size).toBe(1);
    advance(0);
    expect(frames.size).toBe(1);
});

test("reduced motion renders static states and disabled orbs never animate", () => {
    motion.matches = true;
    const orb = mount({ state: "connecting" });
    expect(context.arc).toHaveBeenCalled();
    expect(frames.size).toBe(0);
    const draws = context.clearRect.mock.calls.length;
    orb.state = "speaking";
    orb.level = 1;
    expect(context.clearRect.mock.calls.length).toBeGreaterThan(draws);
    expect(frames.size).toBe(0);
    motion.matches = false;
    motion.dispatchEvent(new Event("change"));
    expect(frames.size).toBe(1);
    orb.state = "disabled";
    expect(frames.size).toBe(0);
    orb.state = "idle";
    expect(frames.size).toBe(1);
});

test("hidden and suspended pages pause animation until they resume", () => {
    const hidden = vi.spyOn(document, "hidden", "get");
    hidden.mockReturnValue(false);
    const orb = mount();
    hidden.mockReturnValue(true);
    document.dispatchEvent(new Event("visibilitychange"));
    expect(frames.size).toBe(0);
    orb.state = "connecting";
    expect(frames.size).toBe(0);
    hidden.mockReturnValue(false);
    document.dispatchEvent(new Event("visibilitychange"));
    expect(frames.size).toBe(1);
    window.dispatchEvent(new Event("pagehide"));
    orb.state = "speaking";
    expect(frames.size).toBe(0);
    window.dispatchEvent(new Event("pageshow"));
    expect(frames.size).toBe(1);
});

test("instances keep independent levels and invalid states fall back to idle", () => {
    const first = mount();
    const second = mount({ state: "__proto__" });
    first.level = 0.8;
    expect(second.level).toBe(0);
    expect(second.state).toBe("idle");
    expect(frames.size).toBe(2);
    first.remove();
    expect(frames.size).toBe(1);
});

test("unavailable canvas rendering does not start an animation loop", () => {
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
    const orb = mount();
    orb.state = "speaking";
    expect(frames.size).toBe(0);
});
