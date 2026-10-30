import { useEffect, useMemo, useState } from "react";
import { ComposableMap, Geographies, Geography } from "react-simple-maps/core";
import { plotClassName, type Trace } from "./common";
import {
    loadGeography,
    mapProjection,
    type Geography as GeographyData,
} from "./geography";

type MapProps = {
    trace: Extract<Trace, { type: "choropleth" }>;
    showlegend: boolean;
};

const lowColor = "color-mix(in oklch, var(--chart-1) 20%, var(--card))";
const highColor = "var(--chart-1)";
const format = new Intl.NumberFormat(undefined, {
    maximumSignificantDigits: 6,
});

export default function Choropleth(props: MapProps) {
    const geography = props.trace.geography;
    const [result, setResult] = useState<{
        boundaries?: GeographyData;
        error?: boolean;
    }>();
    const [attempt, setAttempt] = useState(0);
    useEffect(() => {
        let cancelled = false;
        loadGeography(geography).then(
            (boundaries) => {
                if (!cancelled) setResult({ boundaries });
            },
            () => {
                if (!cancelled) setResult({ error: true });
            },
        );
        return () => {
            cancelled = true;
        };
    }, [geography, attempt]);
    if (result?.error) {
        return (
            <div role="alert" className="py-8 text-sm text-muted-foreground">
                Unable to load map boundaries.{" "}
                <button
                    className="underline"
                    onClick={() => {
                        setResult(undefined);
                        setAttempt(attempt + 1);
                    }}
                >
                    Retry
                </button>
            </div>
        );
    }
    if (!result?.boundaries) {
        return (
            <p role="status" className="py-8 text-sm text-muted-foreground">
                Loading map…
            </p>
        );
    }
    const locations = new Set(
        result.boundaries.features.map((feature) => feature.id),
    );
    if (
        result.boundaries.level !== props.trace.level ||
        props.trace.locations.some((code) => !locations.has(code))
    ) {
        return (
            <p role="alert" className="py-8 text-sm text-muted-foreground">
                Map locations do not match the selected geography.
            </p>
        );
    }
    return <BoundaryMap {...props} boundaries={result.boundaries} />;
}

function BoundaryMap({
    trace,
    showlegend,
    boundaries,
}: MapProps & { boundaries: GeographyData }) {
    const [active, setActive] = useState<string | null>(null);
    const source = boundaries.source;
    const projection = useMemo(
        () => mapProjection(boundaries, trace.geography === "world"),
        [trace.geography, boundaries],
    );
    const values = new Map(
        trace.locations.map((code, index) => [code, trace.values[index]]),
    );
    const present = trace.values.filter((value) => value !== null);
    const min = present.length ? Math.min(...present) : 0;
    const max = present.length ? Math.max(...present) : 0;
    const scale = Math.max(Math.abs(min), Math.abs(max)) || 1;
    const color = (value: number | null | undefined) => {
        if (value == null) return "var(--muted)";
        const position =
            min === max
                ? 0.5
                : (value / scale - min / scale) / (max / scale - min / scale);
        return `color-mix(in oklch, var(--chart-1) ${20 + position * 80}%, var(--card))`;
    };
    const describe = (code: string, name: string) => {
        const value = values.get(code);
        return `${name}: ${value == null ? "No data" : format.format(value)}`;
    };
    const activeFeature = boundaries.features.find(
        (feature) => feature.id === active,
    );

    return (
        <div className={`${plotClassName} flex min-h-0 flex-col`}>
            <ComposableMap
                width={800}
                height={450}
                projection={projection}
                role="group"
                aria-label={
                    trace.name ||
                    (trace.geography === "world"
                        ? "World statistical map"
                        : `${boundaries.name} first-level divisions statistical map`)
                }
                className="min-h-0 w-full flex-1"
            >
                <Geographies geography={boundaries}>
                    {({ geographies }) =>
                        geographies.map((geography) => {
                            const code = String(geography.id);
                            const label = describe(
                                code,
                                geography.properties?.name ?? code,
                            );
                            return (
                                <Geography
                                    key={code}
                                    geography={geography}
                                    fill={color(values.get(code))}
                                    stroke="var(--border)"
                                    strokeWidth={0.6}
                                    vectorEffect="non-scaling-stroke"
                                    tabIndex={0}
                                    role="img"
                                    aria-label={label}
                                    className="outline-none hover:stroke-foreground focus-visible:stroke-foreground focus-visible:stroke-2"
                                    onMouseEnter={() => setActive(code)}
                                    onMouseLeave={() => setActive(null)}
                                    onFocus={() => setActive(code)}
                                    onBlur={() => setActive(null)}
                                >
                                    <title>{label}</title>
                                </Geography>
                            );
                        })
                    }
                </Geographies>
            </ComposableMap>
            <p
                role="status"
                className="min-h-5 shrink-0 text-center text-xs text-muted-foreground"
            >
                {activeFeature
                    ? describe(
                          String(activeFeature.id),
                          activeFeature.properties?.name ?? active,
                      )
                    : present.length
                      ? "Hover or focus a region for details."
                      : "No values to display."}
            </p>
            {showlegend && (
                <div
                    aria-label="Map legend"
                    className="mx-auto mt-2 flex w-full max-w-sm shrink-0 items-center gap-2 text-xs text-muted-foreground"
                >
                    {present.length > 0 && (
                        <>
                            <span>{format.format(min)}</span>
                            <span
                                aria-hidden="true"
                                className="h-2 flex-1 rounded"
                                style={{
                                    background:
                                        min === max
                                            ? color(min)
                                            : `linear-gradient(to right, ${lowColor}, ${highColor})`,
                                }}
                            />
                            <span>{format.format(max)}</span>
                        </>
                    )}
                    <span
                        aria-hidden="true"
                        className="ml-2 size-3 rounded border bg-muted"
                    />
                    <span>No data</span>
                </div>
            )}
            <div className="mt-2 shrink-0 text-right text-[10px] break-words text-muted-foreground">
                <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                >
                    {source.source}
                </a>
                {source.year && ` (${source.year})`}
                {" · "}
                <a
                    href={source.licenseUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                >
                    {source.license}
                </a>
            </div>
        </div>
    );
}
