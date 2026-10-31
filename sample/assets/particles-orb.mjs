const COUNT = 720;
const TAU = Math.PI * 2;
const TONES = 6;
const ALPHAS = 10;
const BUCKETS = TONES * ALPHAS;
const clamp = (value) => Math.max(0, Math.min(1, Number(value) || 0));
const mix = (from, to, amount) => from + (to - from) * amount;
const rgb = (hex) => {
  if (!/^#[\da-f]{6}$/i.test(hex)) return [240, 171, 252];
  return [1, 3, 5].map((start) => parseInt(hex.slice(start, start + 2), 16));
};

const STATES = {
  idle: { spin: 0.14, breathe: 0.05, drift: 1, ripple: 0, swell: 0.04, flow: 0, swirl: 0, pulse: 0, pulseRate: 1, ring: 0, jitter: 0, shake: 0, alpha: 0.72, rest: 0 },
  connecting: { spin: 0.3, breathe: 0.02, drift: 0.2, ripple: 0, swell: 0, flow: 0, swirl: 0, pulse: 0.06, pulseRate: 0.5, ring: 1, jitter: 0, shake: 0, alpha: 0.8, rest: 0.12 },
  listening: { spin: 0.55, breathe: 0.012, drift: 0, ripple: 1, swell: 0.12, flow: 0, swirl: 0, pulse: 0, pulseRate: 1, ring: 0, jitter: 0, shake: 0, alpha: 0.92, rest: 0.55 },
  thinking: { spin: 0.32, breathe: 0.01, drift: 0, ripple: 0, swell: 0, flow: 0, swirl: 0, pulse: 1, pulseRate: 1, ring: 0, jitter: 0, shake: 0, alpha: 0.78, rest: 0.3 },
  speaking: { spin: 0.24, breathe: 0.01, drift: 0, ripple: 0, swell: 0.06, flow: 1, swirl: 1, pulse: 0, pulseRate: 1, ring: 0, jitter: 0.6, shake: 0, alpha: 0.94, rest: 0.55 },
  error: { spin: 0.08, breathe: 0, drift: 0, ripple: 0, swell: 0, flow: 0, swirl: 0, pulse: 0, pulseRate: 1, ring: 0, jitter: 0.7, shake: 1, alpha: 0.85, rest: 0.2 },
  disabled: { spin: 0, breathe: 0, drift: 0, ripple: 0, swell: 0, flow: 0, swirl: 0, pulse: 0, pulseRate: 1, ring: 0, jitter: 0, shake: 0, alpha: 0.45, rest: 0 },
};

const sphere = Array.from({ length: COUNT }, (_, index) => {
  const y = 1 - (index / (COUNT - 1)) * 2;
  const radius = Math.sqrt(1 - y * y);
  const angle = Math.PI * (3 - Math.sqrt(5)) * index;
  const tone = (index * 0.5436890126) % 1;
  return {
    x: Math.cos(angle) * radius,
    y,
    z: Math.sin(angle) * radius,
    ring: (index * 0.61803398875) % 1,
    seed: ((index * 0.7548776662) % 1) * TAU,
    tone,
    bucket: Math.floor(tone * TONES),
  };
});

export class ParticlesOrb extends HTMLElement {
  static observedAttributes = ["state", "size", "speed", "color-from", "color-to"];

  constructor() {
    super();
    this.attachShadow({ mode: "open" }).innerHTML = `
      <style>
        :host { display: inline-grid; width: 168px; height: 168px; place-items: center; pointer-events: none; }
        canvas { display: block; width: 100%; height: 100%; }
      </style>
      <canvas aria-hidden="true"></canvas>
    `;
    this.canvas = this.shadowRoot.querySelector("canvas");
    this.params = { ...STATES.idle };
    this.targetLevel = 0;
    this.energy = 0;
    this.errorWeight = 0;
    this.clock = 0;
    this.pulseClock = 0;
    this.angle = 0;
    this.ringPhase = 0;
    this.frame = null;
    this.last = null;
    this.positions = new Float32Array(COUNT * 3);
    this.buckets = new Uint8Array(COUNT);
    this.order = new Uint16Array(COUNT);
    this.counts = new Uint16Array(BUCKETS);
    this.starts = new Uint16Array(BUCKETS);
    this.palette = [];
    this.tick = (now) => {
      this.frame = null;
      const dt = this.last === null ? 0 : Math.min(0.05, (now - this.last) / 1000);
      this.last = now;
      this.draw(dt);
      this.schedule();
    };
    this.pause = () => {
      this.ownerDocument.defaultView.cancelAnimationFrame(this.frame);
      this.frame = null;
      this.last = null;
    };
    this.suspend = () => {
      this.suspended = true;
      this.pause();
    };
    this.updateVisibility = () => {
      this.pause();
      if (!this.ownerDocument.hidden) this.refresh();
    };
    this.resume = () => {
      this.suspended = false;
      this.refresh();
    };
    this.motionChanged = () => {
      this.pause();
      this.refresh();
    };
  }

  get state() {
    const state = this.getAttribute("state") || "idle";
    return Object.hasOwn(STATES, state) ? state : "idle";
  }

  set state(value) {
    if (this.getAttribute("state") !== value) this.setAttribute("state", value);
  }

  get level() {
    return this.targetLevel;
  }

  set level(value) {
    this.targetLevel = clamp(value);
  }

  connectedCallback() {
    const page = this.ownerDocument.defaultView;
    this.context = this.canvas.getContext("2d");
    this.motion = page.matchMedia("(prefers-reduced-motion: reduce)");
    this.motion.addEventListener("change", this.motionChanged);
    this.ownerDocument.addEventListener("visibilitychange", this.updateVisibility);
    page.addEventListener("pagehide", this.suspend);
    page.addEventListener("pageshow", this.resume);
    if (!this.hasAttribute("role")) this.setAttribute("role", "img");
    if (!this.hasAttribute("aria-label")) this.setAttribute("aria-label", "Assistant orb");
    this.resize();
    this.refresh();
  }

  disconnectedCallback() {
    const page = this.ownerDocument.defaultView;
    this.pause();
    this.motion.removeEventListener("change", this.motionChanged);
    this.ownerDocument.removeEventListener("visibilitychange", this.updateVisibility);
    page.removeEventListener("pagehide", this.suspend);
    page.removeEventListener("pageshow", this.resume);
  }

  attributeChangedCallback(name) {
    if (!this.isConnected || !this.context) return;
    if (name === "size") this.resize();
    this.refresh();
  }

  resize() {
    this.size = Math.max(16, Math.min(1024, Number(this.getAttribute("size")) || 168));
    this.dpr = Math.min(this.ownerDocument.defaultView.devicePixelRatio || 1, 2);
    this.style.width = `${this.size}px`;
    this.style.height = `${this.size}px`;
    this.canvas.width = Math.round(this.size * this.dpr);
    this.canvas.height = Math.round(this.size * this.dpr);
  }

  refresh() {
    if (!this.context) return;
    this.draw(0, this.motion.matches || this.state === "disabled");
    if (this.motion.matches || this.state === "disabled") this.pause();
    else this.schedule();
  }

  schedule() {
    if (this.frame !== null || !this.isConnected || this.suspended || this.ownerDocument.hidden || this.motion.matches || this.state === "disabled") return;
    this.frame = this.ownerDocument.defaultView.requestAnimationFrame(this.tick);
  }

  draw(dt, immediate = false) {
    const target = STATES[this.state];
    const blend = immediate ? 1 : 1 - Math.exp(-dt * 8);
    for (const key of Object.keys(target)) this.params[key] = mix(this.params[key], target[key], blend);
    const p = this.params;
    this.energy = mix(this.energy, this.targetLevel, 1 - Math.exp(-dt * (this.targetLevel > this.energy ? 16 : 7.5)));
    this.errorWeight = mix(this.errorWeight, Number(this.state === "error"), blend);
    const level = this.motion.matches ? p.rest : this.energy;
    const speed = Math.max(0, Math.min(10, Number(this.getAttribute("speed") ?? 1) || 0));
    const elapsed = this.motion.matches ? 0 : dt * speed;
    this.clock += elapsed;
    this.pulseClock += elapsed * p.pulseRate;
    this.angle += elapsed * p.spin * (1 + p.ripple * level * 1.8);
    this.ringPhase = (this.ringPhase + elapsed * 0.7) % TAU;
    this.style.setProperty("--orb-level", this.energy.toFixed(3));

    const t = this.clock + 1.7;
    const pt = this.pulseClock + 1.7;
    const center = this.size / 2;
    const smallness = Math.min(1, this.size / 140);
    const beat = Math.sin(pt * 2.6) * 0.5 + 0.5;
    const radius = center * 0.62 * (1 + p.breathe * Math.sin(t * 1.1) + level * p.swell - p.pulse * (0.06 + 0.12 * beat ** 3));
    const jitter = p.jitter * radius * (0.012 + level * 0.07);
    const drift = p.drift * radius * 0.055;
    const swirl = p.swirl * (0.35 + level * 0.9);
    const shakeX = p.shake * radius * 0.05 * (Math.sin(t * 26) + 0.5 * Math.sin(t * 15.7));
    const shakeY = p.shake * radius * 0.05 * (Math.cos(t * 22.5) + 0.5 * Math.sin(t * 13.1));
    this.counts.fill(0);

    for (let i = 0; i < COUNT; i++) {
      const s = sphere[i];
      const twist = this.angle + swirl * Math.sin(s.y * 2.4 + t * 1.6);
      const x = s.x * Math.cos(twist) - s.z * Math.sin(twist);
      const z = s.x * Math.sin(twist) + s.z * Math.cos(twist);
      const y = s.y * Math.cos(0.32) - z * Math.sin(0.32);
      const depth = (s.y * Math.sin(0.32) + z * Math.cos(0.32) + 1) / 2;
      const perspective = 0.65 + depth * 0.45;
      let r = radius * (1 + p.ripple * (0.04 + level * 0.22) * (0.5 + 0.5 * Math.sin(s.y * 4.5 - t * 6.5)));
      r *= 1 - p.pulse * 0.16 * (0.4 + 0.6 * beat) * (0.5 + 0.5 * Math.sin(s.ring * TAU + pt * 3.1));
      r *= 1 - p.flow * (0.18 + level * 0.4) * (0.5 + 0.5 * Math.sin(s.seed * 3 - t * 3.4)) ** 2;
      let px = center + x * r * perspective + shakeX;
      let py = center + y * r * perspective + shakeY;
      px += drift * (Math.sin(t * 0.55 + s.seed * 3.7) + 0.5 * Math.sin(t * 1.3 + s.seed * 1.3)) + jitter * Math.sin(t * 14 + s.seed * 9.3);
      py += drift * (Math.cos(t * 0.62 + s.seed * 2.9) + 0.5 * Math.sin(t * 1.05 + s.seed * 5.1)) + jitter * Math.cos(t * 17 + s.seed * 6.1);
      let alpha = 0.12 + depth * depth * 0.78;
      let dot = 0.6 + depth * 1.5;
      const ringAngle = (i / COUNT) * TAU + this.ringPhase + 0.05 * Math.sin(t * 1.3 + s.seed);
      const ringRadius = center * (0.58 + 0.13 * s.ring) * (1 + 0.05 * Math.sin(t + s.seed * 1.7)) * (1 + p.pulse * 0.4 * Math.sin(pt * 2.6));
      px = mix(px, center + Math.cos(ringAngle) * ringRadius, p.ring);
      py = mix(py, center + Math.sin(ringAngle) * ringRadius, p.ring);
      alpha = mix(alpha, 0.35 + s.tone * 0.5, p.ring) * smallness ** 2;
      dot = mix(dot, 0.75 + s.tone * 0.9, p.ring) * (0.55 + 0.45 * smallness);
      this.positions[i * 3] = px;
      this.positions[i * 3 + 1] = py;
      this.positions[i * 3 + 2] = dot;
      const bucket = s.bucket * ALPHAS + Math.min(ALPHAS - 1, Math.floor(alpha * ALPHAS));
      this.buckets[i] = bucket;
      this.counts[bucket]++;
    }

    let total = 0;
    for (let b = 0; b < BUCKETS; b++) {
      this.starts[b] = total;
      total += this.counts[b];
    }
    for (let i = 0; i < COUNT; i++) this.order[this.starts[this.buckets[i]]++] = i;

    const paletteKey = `${this.getAttribute("color-from")}|${this.getAttribute("color-to")}|${Math.round(this.errorWeight * 64)}`;
    if (paletteKey !== this.paletteKey) {
      this.paletteKey = paletteKey;
      const from = rgb(this.getAttribute("color-from") || "#f0abfc");
      const to = rgb(this.getAttribute("color-to") || "#818cf8");
      const errorFrom = [248, 113, 113];
      const errorTo = [251, 146, 60];
      this.palette = Array.from({ length: TONES }, (_, b) => {
        const channels = from.map((value, channel) => mix(mix(value, errorFrom[channel], this.errorWeight), mix(to[channel], errorTo[channel], this.errorWeight), (b + 0.5) / TONES));
        return `rgb(${channels.join(",")})`;
      });
    }

    const ctx = this.context;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.size, this.size);
    ctx.globalCompositeOperation = "lighter";
    let cursor = 0;
    for (let b = 0; b < BUCKETS; b++) {
      const count = this.counts[b];
      if (!count) continue;
      ctx.globalAlpha = clamp(((b % ALPHAS + 0.5) / ALPHAS) * p.alpha);
      ctx.fillStyle = this.palette[Math.floor(b / ALPHAS)];
      ctx.beginPath();
      for (let k = cursor; k < cursor + count; k++) {
        const i = this.order[k] * 3;
        const x = this.positions[i], y = this.positions[i + 1], r = this.positions[i + 2];
        ctx.moveTo(x + r, y);
        ctx.arc(x, y, r, 0, TAU);
      }
      ctx.fill();
      cursor += count;
    }
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
  }
}

if (!customElements.get("particles-orb")) customElements.define("particles-orb", ParticlesOrb);
