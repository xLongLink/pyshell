import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
    act,
    cleanup,
    fireEvent,
    render,
    screen,
    waitFor,
} from "@testing-library/react";
import { figureProps, a2uiBatchSchema } from "../lib/a2ui";
import GeographicMap from "../components/plot/map";
import { mapTraceSchema } from "../lib/map";
import * as geography from "../components/plot/geography";

const sdk = vi.hoisted(() => {
    class FakeMap {
        static instances: FakeMap[] = [];
        handlers: Record<string, (event?: unknown) => void> = {};
        sources: Record<string, { setData: ReturnType<typeof vi.fn> }> = {};
        layers: Record<string, unknown> = {};
        constructor() {
            FakeMap.instances.push(this);
        }
        on = vi.fn((event: string, handler: (event?: unknown) => void) => {
            this.handlers[event] = handler;
        });
        addControl = vi.fn();
        resize = vi.fn();
        remove = vi.fn();
        flyTo = vi.fn();
        fitBounds = vi.fn();
        getStyle = () => ({ sources: { basemap: {}, ...this.sources } });
        getSource = (id: string) => this.sources[id];
        getLayer = (id: string) => this.layers[id];
        addSource = vi.fn((id: string) => {
            this.sources[id] = { setData: vi.fn() };
        });
        addLayer = vi.fn((layer: { id: string }) => {
            this.layers[layer.id] = layer;
        });
        removeLayer = vi.fn((id: string) => {
            delete this.layers[id];
        });
        removeSource = vi.fn((id: string) => {
            delete this.sources[id];
        });
        moveLayer = vi.fn();
        setPaintProperty = vi.fn();
        queryRenderedFeatures = vi.fn();
    }
    const content = vi.fn();
    const removePopup = vi.fn();
    class FakePopup {
        setLngLat() {
            return this;
        }
        setDOMContent(element: HTMLElement) {
            content(element);
            return this;
        }
        addTo() {
            return this;
        }
        remove = removePopup;
    }
    return { FakeMap, FakePopup, content, removePopup };
});

vi.mock("maplibre-gl", () => ({
    Map: sdk.FakeMap,
    Popup: sdk.FakePopup,
    NavigationControl: class {},
    setWorkerUrl: vi.fn(),
}));

const input = {
    type: "map",
    view: { center: [8.54, 47.37], zoom: 12 },
    layers: [
        {
            id: "buildings",
            name: "Renovated before 2025",
            popup_fields: ["name", "renovation_year"],
            data: {
                type: "FeatureCollection",
                features: [
                    {
                        type: "Feature",
                        id: "one",
                        geometry: { type: "Point", coordinates: [8.54, 47.37] },
                        properties: { name: "Building", renovation_year: 2024 },
                    },
                ],
            },
        },
    ],
};

beforeEach(() => {
    sdk.FakeMap.instances = [];
    vi.stubGlobal(
        "ResizeObserver",
        class {
            observe() {}
            disconnect() {}
        },
    );
});

afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
});

test("map schemas accept empty results and preserve scalar feature properties on the wire", () => {
    const figure = { id: "root", component: "Figure", data: [input] };
    const batch = a2uiBatchSchema.parse([
        { version: "v0.9", createSurface: { surfaceId: "buildings" } },
        {
            version: "v0.9",
            updateComponents: { surfaceId: "buildings", components: [figure] },
        },
    ]);
    const trace = batch[1].updateComponents.components[0].data[0];
    expect(
        trace.type === "map" &&
            trace.layers[0].data.features[0].properties.renovation_year,
    ).toBe(2024);
    expect(mapTraceSchema.parse({ type: "map", layers: [] }).layers).toEqual(
        [],
    );
    expect(
        figureProps.safeParse({
            data: [input, { type: "bar", x: [1], y: [1] }],
        }).success,
    ).toBe(false);
});

test.each([
    { center: [181, 0] },
    { zoom: 2 },
    { country: "Switzerland" },
    { country: "CHE", center: [8, 47] },
    { center: [8, 47], zoom: 23 },
    {
        bounds: [
            [5, 48],
            [11, 45],
        ],
    },
    {},
])("invalid camera is rejected: %j", (view) => {
    expect(mapTraceSchema.safeParse({ ...input, view }).success).toBe(false);
});

test("duplicate IDs, non-scalar properties, and aggregate feature budgets are rejected", () => {
    expect(
        mapTraceSchema.safeParse({
            ...input,
            layers: [input.layers[0], input.layers[0]],
        }).success,
    ).toBe(false);
    const duplicate = structuredClone(input);
    duplicate.layers[0].data.features.push(
        duplicate.layers[0].data.features[0],
    );
    expect(mapTraceSchema.safeParse(duplicate).success).toBe(false);
    const feature = input.layers[0].data.features[0];
    expect(
        mapTraceSchema.safeParse({
            ...input,
            layers: [
                {
                    ...input.layers[0],
                    data: {
                        type: "FeatureCollection",
                        features: [
                            {
                                ...feature,
                                properties: { nested: { value: 1 } },
                            },
                        ],
                    },
                },
            ],
        }).success,
    ).toBe(false);
    const layers = ["first", "second"].map((id) => ({
        ...input.layers[0],
        id,
        data: {
            type: "FeatureCollection",
            features: Array.from({ length: 101 }, (_, index) => ({
                ...feature,
                id: String(index),
            })),
        },
    }));
    expect(mapTraceSchema.safeParse({ ...input, layers }).success).toBe(false);
});

test("data and title updates preserve the camera; explicit repeated navigation moves it", () => {
    const trace = mapTraceSchema.parse(input);
    const { rerender, unmount } = render(
        <GeographicMap trace={trace} showlegend />,
    );
    const map = sdk.FakeMap.instances[0];
    act(() => map.handlers.load());
    expect(map.flyTo).toHaveBeenCalledTimes(1);
    expect(map.addSource).toHaveBeenCalledWith(
        "data-buildings",
        expect.objectContaining({ type: "geojson" }),
    );
    const updated = mapTraceSchema.parse(input);
    updated.layers[0].data.features[0].properties.renovation_year = 2023;
    rerender(<GeographicMap trace={updated} showlegend={false} />);
    expect(map.flyTo).toHaveBeenCalledTimes(1);
    expect(map.sources["data-buildings"].setData).toHaveBeenCalledWith(
        updated.layers[0].data,
    );
    expect(screen.queryByLabelText("Map layers")).toBeNull();
    rerender(
        <GeographicMap
            trace={mapTraceSchema.parse({
                ...input,
                view: { ...input.view, revision: 1 },
            })}
            showlegend
        />,
    );
    expect(map.flyTo).toHaveBeenCalledTimes(2);
    rerender(
        <GeographicMap
            trace={mapTraceSchema.parse({ type: "map", layers: [] })}
            showlegend
        />,
    );
    expect(map.flyTo).toHaveBeenCalledTimes(2);
    expect(map.removeSource).toHaveBeenCalledWith("data-buildings");
    expect(screen.getByRole("status").textContent).toContain("No features");
    unmount();
    expect(map.remove).toHaveBeenCalledOnce();
});

test("feature details are rendered as text, not HTML, and only selected properties are shown", () => {
    const trace = mapTraceSchema.parse(input);
    render(<GeographicMap trace={trace} showlegend />);
    const map = sdk.FakeMap.instances[0];
    act(() => map.handlers.load());
    map.queryRenderedFeatures.mockReturnValue([
        {
            source: "data-buildings",
            properties: {
                name: '<img src=x onerror="alert(1)">',
                renovation_year: null,
                other: "not shown",
            },
        },
    ]);
    act(() => map.handlers.click({ point: [0, 0], lngLat: [8, 47] }));
    const content = sdk.content.mock.calls.at(-1)![0] as HTMLElement;
    expect(content.querySelector("img")).toBeNull();
    expect(content.textContent).toContain('<img src=x onerror="alert(1)">');
    expect(content.textContent).toContain("renovation_year: Unknown");
    expect(content.textContent).not.toContain("not shown");
});

test("country navigation fits downloaded bounds without replacing layers", async () => {
    vi.spyOn(geography, "loadGeography").mockResolvedValue({
        type: "FeatureCollection",
        geography: "world",
        name: "World",
        level: "country",
        features: [
            {
                type: "Feature",
                id: "CHE",
                properties: { name: "Switzerland" },
                geometry: {
                    type: "Polygon",
                    coordinates: [
                        [
                            [6, 46],
                            [6, 48],
                            [10, 48],
                            [10, 46],
                            [6, 46],
                        ],
                    ],
                },
            },
        ],
    });
    render(
        <GeographicMap
            trace={mapTraceSchema.parse({ ...input, view: { country: "CHE" } })}
            showlegend
        />,
    );
    const map = sdk.FakeMap.instances[0];
    act(() => map.handlers.load());
    await waitFor(() => expect(map.fitBounds).toHaveBeenCalledOnce());
    const bounds = map.fitBounds.mock.calls[0][0];
    expect(bounds[0][0]).toBeCloseTo(6);
    expect(bounds[1][0]).toBeCloseTo(10);
    expect(map.sources["data-buildings"]).toBeDefined();
});

test("unavailable country bounds report an error and allow retry with a fresh map", async () => {
    vi.spyOn(geography, "loadGeography").mockRejectedValue(
        new Error("Offline"),
    );
    render(
        <GeographicMap
            trace={mapTraceSchema.parse({ ...input, view: { country: "CHE" } })}
            showlegend
        />,
    );
    const map = sdk.FakeMap.instances[0];
    act(() => map.handlers.load());
    expect((await screen.findByRole("alert")).textContent).toContain(
        "Unable to load country bounds",
    );
    fireEvent.click(screen.getByRole("button", { name: "Retry" }));
    expect(map.remove).toHaveBeenCalledOnce();
    expect(sdk.FakeMap.instances).toHaveLength(2);
});
