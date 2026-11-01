import { z } from "zod";
import { a2uiBatchSchema, identifier } from "./a2ui.ts";

export const placementsSchema = z.record(
    identifier,
    z.strictObject({ pinned: z.boolean() }),
);

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
        placements: placementsSchema,
    })
    .superRefine(({ events, placements }, context) => {
        const ids = events.map(
            ({ messages }) => messages[0].createSurface.surfaceId,
        );
        if (new Set(ids).size !== ids.length)
            context.addIssue({
                code: "custom",
                message: "Duplicate surface ID",
            });
        const placementIds = Object.keys(placements);
        if (
            placementIds.length !== ids.length ||
            ids.some((id) => !Object.hasOwn(placements, id))
        )
            context.addIssue({
                code: "custom",
                message:
                    "Placements must match the current surface IDs exactly",
            });
        if (
            Object.values(placements).filter(({ pinned }) => !pinned).length > 4
        )
            context.addIssue({
                code: "custom",
                message: "At most four surfaces may be unpinned",
            });
    })
    .transform(({ events, placements }) => ({
        surfaces: events.map(({ messages: [creation, update] }) => {
            const component = update.updateComponents.components[0];
            return {
                id: creation.createSurface.surfaceId,
                pinned: placements[creation.createSurface.surfaceId].pinned,
                props: {
                    data: component.data,
                    layout: component.layout,
                },
            };
        }),
    }));

export type Surface = z.infer<typeof messageSchema>["surfaces"][number];
