import { z } from "zod";

const identifier = z.string().regex(/^[A-Za-z0-9_-]{1,64}$/);
export type WorkspaceMessage =
    | { version: string; createSurface: { surfaceId: string } }
    | { version: string; updateComponents: { surfaceId: string } }
    | { version: string; deleteSurface: { surfaceId: string } };
const uiMessageSchema: z.ZodType<WorkspaceMessage> = z.union([
    z
        .object({
            version: z.string(),
            createSurface: z.object({ surfaceId: identifier }).passthrough(),
        })
        .passthrough(),
    z
        .object({
            version: z.string(),
            updateComponents: z.object({ surfaceId: identifier }).passthrough(),
        })
        .passthrough(),
    z
        .object({
            version: z.string(),
            deleteSurface: z.object({ surfaceId: identifier }).passthrough(),
        })
        .passthrough(),
]);

export const placementsSchema = z.record(
    identifier,
    z.object({ pinned: z.boolean() }).strict(),
);
export type Placements = z.infer<typeof placementsSchema>;

export const eventSchema = z.discriminatedUnion("type", [
    z.object({
        type: z.literal("a2ui"),
        messages: z.array(uiMessageSchema).max(100),
    }),
    z.object({ type: z.literal("placement"), widgets: placementsSchema }),
    z.object({ type: z.literal("instruction"), text: z.string() }),
    z.object({ type: z.literal("error"), text: z.string() }),
    z.object({ type: z.literal("state"), busy: z.boolean() }),
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
        busy: z.boolean(),
        placements: placementsSchema.default({}),
    }),
]);
