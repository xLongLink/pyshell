import { expect, test } from "vitest";
import { placementsSchema } from "../lib/placements";

test("placements accept pinned flags and reject invalid identifiers and extra state", () => {
    const placements = { sales: { pinned: true }, forecast: { pinned: false } };
    expect(placementsSchema.parse(placements)).toEqual(placements);
    for (const value of [
        { "invalid/id": { pinned: true } },
        { sales: { pinned: "true" } },
        { sales: { pinned: true, focused: true } },
    ]) {
        expect(placementsSchema.safeParse(value).success).toBe(false);
    }
});
