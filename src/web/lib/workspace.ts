import { z } from "zod";
import { a2uiBatchSchema, identifier } from "./a2ui.ts";

export const placementsSchema = z.record(
    identifier,
    z.strictObject({ pinned: z.boolean() }),
);
export type Placements = z.infer<typeof placementsSchema>;

export const messageSchema = z
    .object({
        type: z.literal("snapshot"),
        events: z
            .array(
                z.object({
                    type: z.literal("a2ui"),
                    messages: a2uiBatchSchema,
                }),
            )
            .max(20),
        placements: placementsSchema.default({}),
    })
    .superRefine(({ events }, context) => {
        const ids = events.map(
            ({ messages }) => messages[0].createSurface.surfaceId,
        );
        if (new Set(ids).size !== ids.length)
            context.addIssue({
                code: "custom",
                message: "Duplicate surface ID",
            });
    })
    .transform(({ events, placements }) => ({
        surfaces: events.map(({ messages: [creation, update] }) => {
            const component = update?.updateComponents.components[0];
            return {
                id: creation.createSurface.surfaceId,
                props: component && {
                    data: component.data,
                    layout: component.layout,
                },
            };
        }),
        placements,
    }));

export type Surface = z.infer<typeof messageSchema>["surfaces"][number];
