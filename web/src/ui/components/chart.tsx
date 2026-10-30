import type { z } from "zod";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { chartProps } from "@/lib/a2ui";

const colors = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];

export function Chart({
  kind,
  title,
  points,
}: z.infer<typeof chartProps> & { kind: "bar" | "line" | "pie" }) {
  const caption =
    title || { bar: "Bar chart", line: "Line chart", pie: "Pie chart" }[kind];
  const slices = points.filter((point) => point.value !== null);

  return (
    <figure className="min-w-0 rounded-lg border bg-card p-5">
      <figcaption className="mb-4 font-semibold">{caption}</figcaption>
      {kind === "pie" &&
      !slices.some((point) => point.value !== null && point.value > 0) ? (
        <p className="py-8 text-sm text-muted-foreground">
          No positive values to display.
        </p>
      ) : (
        <div className="h-72 w-full" aria-hidden="true">
          <ResponsiveContainer width="100%" height="100%">
            {kind === "pie" ? (
              <PieChart accessibilityLayer={false}>
                <Pie
                  data={slices}
                  dataKey="value"
                  nameKey="label"
                  isAnimationActive={false}
                >
                  {slices.map((_, index) => (
                    <Cell key={index} fill={colors[index % colors.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            ) : kind === "line" ? (
              <LineChart data={points} accessibilityLayer={false}>
                <CartesianGrid vertical={false} stroke="var(--border)" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: "var(--muted-foreground)" }}
                />
                <YAxis tick={{ fill: "var(--muted-foreground)" }} />
                <Tooltip />
                <Line
                  dataKey="value"
                  type="linear"
                  stroke="var(--chart-1)"
                  connectNulls={false}
                  isAnimationActive={false}
                />
              </LineChart>
            ) : (
              <BarChart data={points} accessibilityLayer={false}>
                <CartesianGrid vertical={false} stroke="var(--border)" />
                <XAxis
                  dataKey="label"
                  tick={{ fill: "var(--muted-foreground)" }}
                />
                <YAxis tick={{ fill: "var(--muted-foreground)" }} />
                <Tooltip />
                <Bar
                  dataKey="value"
                  fill="var(--chart-1)"
                  isAnimationActive={false}
                />
              </BarChart>
            )}
          </ResponsiveContainer>
        </div>
      )}
      <details className="mt-4 text-sm">
        <summary className="cursor-pointer text-muted-foreground">
          View chart data
        </summary>
        <table className="mt-2 w-full text-left">
          <caption className="sr-only">{caption} data</caption>
          <thead>
            <tr>
              <th scope="col">Label</th>
              <th scope="col">Value</th>
            </tr>
          </thead>
          <tbody>
            {points.map((point, index) => (
              <tr key={index}>
                <th scope="row" className="font-normal">
                  {point.label}
                </th>
                <td>{point.value ?? "Missing"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </figure>
  );
}
