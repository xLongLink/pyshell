import { z } from "zod";
import { identifier } from "@/lib/a2ui";

export const placementsSchema = z.record(
    identifier,
    z.object({ pinned: z.boolean() }).strict(),
);
export type Placements = z.infer<typeof placementsSchema>;
