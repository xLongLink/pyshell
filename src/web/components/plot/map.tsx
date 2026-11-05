import { useEffect, useRef, useState } from "react";
import { geoBounds } from "d3-geo";
import {
    Map,
    NavigationControl,
    Popup,
    setWorkerUrl,
    type GeoJSONSource,
    type MapMouseEvent,
} from "maplibre-gl";
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url";
import "maplibre-gl/dist/maplibre-gl.css";
import type { MapTrace } from "@/lib/map";
import { loadGeography, orientBoundaries } from "./geography";

setWorkerUrl(workerUrl);

export default function GeographicMap({
    trace,
    showlegend,
}: {
    trace: MapTrace;
    showlegend: boolean;
}) {
    const container = useRef<HTMLDivElement>(null);
    const current = useRef(trace);
    const popup = useRef<Popup | null>(null);
    const [map, setMap] = useState<Map | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [attempt, setAttempt] = useState(0);
    current.current = trace;

    useEffect(() => {
        if (!container.current) return;
        let instance: Map;
        setError(null);
        try {
            instance = new Map({
                container: container.current,
                center: [0, 20],
                zoom: 1,
                style: {
                    version: 8,
                    sources: {
                        basemap: {
                            type: "raster",
                            tiles: [
                                "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                            ],
                            tileSize: 256,
                            maxzoom: 19,
                            attribution:
                                '© <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
                        },
                    },
                    layers: [
                        { id: "basemap", type: "raster", source: "basemap" },
                    ],
                },
            });
        } catch {
            setError(
                "Unable to initialize the map. WebGL support is required.",
            );
            return;
        }
        instance.addControl(new NavigationControl(), "top-right");
        instance.on("load", () => setMap(instance));
        instance.on("error", () =>
            setError(
                "Some map resources could not be loaded. Check your connection or retry.",
            ),
        );
        const click = (event: MapMouseEvent) => {
            const layers = current.current.layers;
            const ids = layers
                .flatMap((layer) => [`${layer.id}-fill`, `${layer.id}-point`])
                .filter((id) => instance.getLayer(id));
            if (!ids.length) return;
            const feature = instance.queryRenderedFeatures(event.point, {
                layers: ids,
            })[0];
            popup.current?.remove();
            if (!feature) return;
            const layer = layers.find(
                (item) => `data-${item.id}` === feature.source,
            );
            if (!layer) return;
            const content = document.createElement("div");
            content.className = "text-sm text-black";
            const heading = document.createElement("strong");
            heading.textContent = layer.name || layer.id;
            content.append(heading);
            const fields = layer.popup_fields.length
                ? layer.popup_fields
                : Object.keys(feature.properties ?? {});
            for (const field of fields) {
                const row = document.createElement("div");
                const value = feature.properties?.[field];
                row.textContent = `${field}: ${value == null ? "Unknown" : String(value)}`;
                content.append(row);
            }
            const detail = new Popup();
            detail
                .setLngLat(event.lngLat)
                .setDOMContent(content)
                .addTo(instance);
            popup.current = detail;
        };
        instance.on("click", click);
        const observer = new ResizeObserver(() => instance.resize());
        observer.observe(container.current);
        return () => {
            observer.disconnect();
            popup.current?.remove();
            instance.remove();
            setMap(null);
        };
    }, [attempt]);

    useEffect(() => {
        if (!map) return;
        popup.current?.remove();
        const keep = new Set(trace.layers.map((layer) => `data-${layer.id}`));
        for (const id of Object.keys(map.getStyle().sources)) {
            if (id === "basemap" || keep.has(id)) continue;
            for (const suffix of ["fill", "line", "point"])
                map.removeLayer(`${id.slice(5)}-${suffix}`);
            map.removeSource(id);
        }
        for (const layer of trace.layers) {
            const sourceId = `data-${layer.id}`;
            const source = map.getSource(sourceId) as GeoJSONSource | undefined;
            if (source) source.setData(layer.data);
            else {
                map.addSource(sourceId, { type: "geojson", data: layer.data });
                map.addLayer({
                    id: `${layer.id}-fill`,
                    type: "fill",
                    source: sourceId,
                    filter: ["==", ["geometry-type"], "Polygon"],
                    paint: { "fill-color": layer.color, "fill-opacity": 0.55 },
                });
                map.addLayer({
                    id: `${layer.id}-line`,
                    type: "line",
                    source: sourceId,
                    filter: ["==", ["geometry-type"], "Polygon"],
                    paint: { "line-color": layer.color, "line-width": 1.5 },
                });
                map.addLayer({
                    id: `${layer.id}-point`,
                    type: "circle",
                    source: sourceId,
                    filter: ["==", ["geometry-type"], "Point"],
                    paint: {
                        "circle-color": layer.color,
                        "circle-radius": 6,
                        "circle-stroke-color": "#ffffff",
                        "circle-stroke-width": 1,
                    },
                });
            }
            map.setPaintProperty(`${layer.id}-fill`, "fill-color", layer.color);
            map.setPaintProperty(`${layer.id}-line`, "line-color", layer.color);
            map.setPaintProperty(
                `${layer.id}-point`,
                "circle-color",
                layer.color,
            );
        }
        trace.layers.forEach((layer) => {
            for (const suffix of ["fill", "line", "point"])
                map.moveLayer(`${layer.id}-${suffix}`);
        });
    }, [map, trace.layers]);

    const viewKey = JSON.stringify(trace.view ?? null);
    useEffect(() => {
        if (!map) return;
        const view = JSON.parse(viewKey) as MapTrace["view"];
        if (!view) return;
        setError(null);
        let cancelled = false;
        if (view.center)
            map.flyTo({ center: view.center, zoom: view.zoom ?? 10 });
        else if (view.bounds) map.fitBounds(view.bounds, { padding: 32 });
        else if (view.country) {
            loadGeography("world")
                .then((geography) => {
                    if (cancelled) return;
                    const feature = geography.features.find(
                        (item) => item.id === view.country,
                    );
                    if (!feature)
                        throw new Error("Country boundaries are unavailable");
                    const bounds = geoBounds(
                        orientBoundaries({
                            type: "FeatureCollection",
                            features: [feature],
                        }),
                    );
                    if (bounds[1][0] < bounds[0][0]) bounds[1][0] += 360;
                    map.fitBounds(bounds, { padding: 32 });
                })
                .catch(() => {
                    if (!cancelled)
                        setError(
                            `Unable to load country bounds for ${view.country}. Use explicit bounds or retry.`,
                        );
                });
        }
        return () => {
            cancelled = true;
        };
    }, [map, viewKey]);

    const count = trace.layers.reduce(
        (total, layer) => total + layer.data.features.length,
        0,
    );
    return (
        <div className="flex h-full min-h-0 flex-col gap-2">
            <div
                ref={container}
                aria-label="Interactive geographic map"
                className="relative min-h-32 w-full flex-1 overflow-hidden rounded-md"
            />
            {error && (
                <div role="alert" className="text-xs text-destructive">
                    {error}{" "}
                    <button
                        className="underline"
                        onClick={() => setAttempt((value) => value + 1)}
                    >
                        Retry
                    </button>
                </div>
            )}
            <p role="status" className="text-xs text-muted-foreground">
                {!map && !error
                    ? "Loading map…"
                    : count
                      ? `${count} features. Click a feature for details.`
                      : "No features to display."}
            </p>
            {showlegend && trace.layers.length > 0 && (
                <ul
                    aria-label="Map layers"
                    className="flex flex-wrap gap-3 text-xs"
                >
                    {trace.layers.map((layer) => (
                        <li key={layer.id} className="flex items-center gap-1">
                            <span
                                className="size-3 rounded"
                                style={{ backgroundColor: layer.color }}
                            />
                            {layer.name || layer.id} (
                            {layer.data.features.length})
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
