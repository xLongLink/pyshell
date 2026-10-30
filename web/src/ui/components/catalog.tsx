import { createComponentImplementation } from "@a2ui/react/v0_9";
import { Catalog } from "@a2ui/web_core/v0_9";
import { BarChart } from "@/components/chart/bar";
import { LineChart } from "@/components/chart/line";
import { PieChart } from "@/components/chart/pie";
import { catalogId, chartProps, pieChartProps } from "@/lib/a2ui";

const Bar = createComponentImplementation(
  { name: "BarChart", schema: chartProps },
  ({ props }) => <BarChart {...props} />,
);

const Line = createComponentImplementation(
  { name: "LineChart", schema: chartProps },
  ({ props }) => <LineChart {...props} />,
);

const Pie = createComponentImplementation(
  { name: "PieChart", schema: pieChartProps },
  ({ props }) => <PieChart {...props} />,
);

export const resultCatalog = new Catalog(catalogId, "v0.9", [Bar, Line, Pie]);
