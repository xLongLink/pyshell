import { z } from "zod";
import { a2uiBatchSchema, identifier } from "./a2ui.ts";

export const placementsSchema = z.record(
    identifier,
    z.object({ pinned: z.boolean() }).strict(),
);
export type Placements = z.infer<typeof placementsSchema>;

export const eventSchema = z.discriminatedUnion("type", [
    z.object({ type: z.literal("a2ui"), messages: a2uiBatchSchema }),
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
        events: z.array(eventSchema),
        placements: placementsSchema.default({}),
    }),
]);
