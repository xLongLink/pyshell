import { z } from "zod";
import { a2uiBatchSchema, identifier } from "./a2ui.ts";

export const placementsSchema = z.record(
    identifier,
    z.strictObject({ pinned: z.boolean() }),
);
export type Placements = z.infer<typeof placementsSchema>;

const a2uiEventSchema = z.object({
    type: z.literal("a2ui"),
    messages: a2uiBatchSchema,
});

export const eventSchema = z.discriminatedUnion("type", [
    a2uiEventSchema,
    z.object({ type: z.literal("placement"), widgets: placementsSchema }),
    z.object({ type: z.literal("instruction"), text: z.string() }),
    z.object({ type: z.literal("error"), text: z.string() }),
]);

export const voiceEventSchema = z.discriminatedUnion("name", [
    z.object({
        type: z.literal("voice"),
        name: z.literal("answer"),
        sdp: z.string().min(1),
    }),
    z.object({ type: z.literal("voice"), name: z.literal("closed") }),
    z.object({
        type: z.literal("voice"),
        name: z.literal("error"),
        text: z.string(),
    }),
]);
export type VoiceEvent = z.infer<typeof voiceEventSchema>;

export const messageSchema = z.union([
    eventSchema,
    voiceEventSchema,
    z.object({
        type: z.literal("snapshot"),
        events: z.array(a2uiEventSchema),
        placements: placementsSchema.default({}),
    }),
]);
