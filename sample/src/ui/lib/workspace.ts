import { z } from "zod";

export const eventSchema = z.discriminatedUnion("type", [
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
    }),
]);
