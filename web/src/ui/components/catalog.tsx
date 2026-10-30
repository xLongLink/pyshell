import { Fragment } from "react";
import { createComponentImplementation } from "@a2ui/react/v0_9";
import { Catalog, childList } from "@a2ui/web_core/v0_9";
import { Chart } from "@/components/chart";
import {
  catalogId,
  chartProps,
  pieChartProps,
  layoutProps,
  metricProps,
  tableProps,
  textProps,
} from "@/lib/a2ui";

const Text = createComponentImplementation(
  { name: "Text", schema: textProps },
  ({ props }) => (
    <section>
      {props.title && <h3 className="mb-2 font-semibold">{props.title}</h3>}
      <p className="whitespace-pre-wrap text-sm leading-relaxed">
        {props.text}
      </p>
    </section>
  ),
);

const layoutSchema = layoutProps.extend({ children: childList() });

const Column = createComponentImplementation(
  { name: "Column", schema: layoutSchema },
  ({ props, buildChild }) => (
    <div className="flex min-w-0 flex-col gap-4">
      {props.children.map((child) => (
        <Fragment key={typeof child === "string" ? child : child.id}>
          {typeof child === "string"
            ? buildChild(child)
            : buildChild(child.id, child.basePath)}
        </Fragment>
      ))}
    </div>
  ),
);

const Row = createComponentImplementation(
  { name: "Row", schema: layoutSchema },
  ({ props, buildChild }) => (
    <div className="flex min-w-0 flex-wrap items-start gap-4">
      {props.children.map((child) => (
        <Fragment key={typeof child === "string" ? child : child.id}>
          {typeof child === "string"
            ? buildChild(child)
            : buildChild(child.id, child.basePath)}
        </Fragment>
      ))}
    </div>
  ),
);

const Card = createComponentImplementation(
  { name: "Card", schema: layoutSchema },
  ({ props, buildChild }) => (
    <section className="min-w-0 space-y-4 rounded-lg border bg-card p-5">
      {props.children.map((child) => (
        <Fragment key={typeof child === "string" ? child : child.id}>
          {typeof child === "string"
            ? buildChild(child)
            : buildChild(child.id, child.basePath)}
        </Fragment>
      ))}
    </section>
  ),
);

const Metric = createComponentImplementation(
  { name: "Metric", schema: metricProps },
  ({ props }) => (
    <dl className="min-w-40 rounded-lg border bg-card p-5">
      <dt className="text-sm text-muted-foreground">{props.label}</dt>
      <dd className="mt-2 text-3xl font-semibold tracking-tight">
        {props.value}
      </dd>
      {props.detail && (
        <dd className="mt-2 text-xs text-muted-foreground">{props.detail}</dd>
      )}
    </dl>
  ),
);

const Table = createComponentImplementation(
  { name: "Table", schema: tableProps },
  ({ props }) => (
    <div className="max-w-full overflow-x-auto rounded-lg border bg-card">
      <table className="w-full text-left text-sm">
        <caption className="px-4 py-3 text-left font-semibold">
          {props.title || "Data table"}
        </caption>
        <thead>
          <tr className="border-y bg-accent">
            {props.columns.map((column, index) => (
              <th key={index} scope="col" className="px-4 py-3 font-medium">
                {String(column)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {props.rows.map((row, index) => (
            <tr key={index} className="border-b last:border-0">
              {row.map((value, column) => (
                <td key={column} className="px-4 py-3">
                  {value === null ? "—" : String(value)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  ),
);

const BarChart = createComponentImplementation(
  { name: "BarChart", schema: chartProps },
  ({ props }) => <Chart kind="bar" {...props} />,
);

const LineChart = createComponentImplementation(
  { name: "LineChart", schema: chartProps },
  ({ props }) => <Chart kind="line" {...props} />,
);

const PieChart = createComponentImplementation(
  { name: "PieChart", schema: pieChartProps },
  ({ props }) => <Chart kind="pie" {...props} />,
);

export const resultCatalog = new Catalog(catalogId, "v0.9", [
  Text,
  Column,
  Row,
  Card,
  Metric,
  Table,
  BarChart,
  LineChart,
  PieChart,
]);
