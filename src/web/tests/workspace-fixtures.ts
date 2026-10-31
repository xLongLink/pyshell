import { catalogId } from "../lib/a2ui.ts";

export function creation(surfaceId: string) {
    return { version: "v0.9", createSurface: { surfaceId, catalogId } };
}

export function update(surfaceId: string) {
    return {
        version: "v0.9",
        updateComponents: {
            surfaceId,
            components: [
                {
                    id: "root",
                    component: "Figure",
                    data: [{ type: "bar", x: ["A"], y: [1] }],
                },
            ],
        },
    };
}
