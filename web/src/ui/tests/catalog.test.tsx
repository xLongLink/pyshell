import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { Results } from "../components/results";
import { a2uiBatchSchema, catalogId } from "../lib/a2ui";

beforeEach(() => {
  vi.stubGlobal(
    "ResizeObserver",
    class {
      constructor(private callback: ResizeObserverCallback) {}
      observe(target: Element) {
        this.callback(
          [
            {
              target,
              contentRect: { width: 600, height: 300 },
            } as ResizeObserverEntry,
          ],
          this as unknown as ResizeObserver,
        );
      }
      unobserve() {}
      disconnect() {}
    },
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function surface(components: unknown[], surfaceId = "sales") {
  return a2uiBatchSchema.parse([
    { version: "v0.9", createSurface: { surfaceId, catalogId } },
    { version: "v0.9", updateComponents: { surfaceId, components } },
  ]);
}

function bar(title: string) {
  return {
    id: "root",
    component: "BarChart",
    title,
    points: [{ label: "A", value: 42 }],
  };
}

test("replacement, removal, and reset do not duplicate result surfaces", async () => {
  const messages = surface([bar("Before")]);
  const { rerender } = render(<Results messages={messages} />);
  await waitFor(() => expect(screen.getByText("Before")).toBeTruthy());
  const replaced = a2uiBatchSchema.parse([
    ...messages,
    { version: "v0.9", deleteSurface: { surfaceId: "sales" } },
    ...surface([bar("After")]),
  ]);
  rerender(<Results messages={replaced} />);
  await waitFor(() => expect(screen.getByText("After")).toBeTruthy());
  expect(screen.queryByText("Before")).toBeNull();
  expect(screen.getAllByText("After")).toHaveLength(1);
  rerender(
    <Results
      messages={[
        ...replaced,
        { version: "v0.9", deleteSurface: { surfaceId: "sales" } },
      ]}
    />,
  );
  await waitFor(() => expect(screen.queryByText("After")).toBeNull());
  rerender(<Results messages={messages} />);
  await waitFor(() => expect(screen.getByText("Before")).toBeTruthy());
  rerender(<Results messages={[]} />);
  await waitFor(() => expect(screen.queryByLabelText("Results")).toBeNull());
});

test("chart titles remain inert even when they contain HTML", async () => {
  const text = '<img src=x onerror="alert(1)">';
  const { container } = render(<Results messages={surface([bar(text)])} />);
  await waitFor(() => expect(screen.getByText(text)).toBeTruthy());
  expect(container.querySelector("img")).toBeNull();
});

test.each(["BarChart", "LineChart", "PieChart"] as const)(
  "%s renders its plot without a data table while preserving missing values",
  async (component) => {
    const { container } = render(
      <Results
        messages={surface([
          {
            id: "root",
            component,
            title: "Revenue",
            points: [
              { label: "Apples", value: 42 },
              { label: "Pears", value: null },
              { label: "Oranges", value: 84 },
            ],
          },
        ])}
      />,
    );
    await waitFor(() => expect(screen.getByText("Revenue")).toBeTruthy());
    expect(screen.queryByText("View chart data")).toBeNull();
    expect(screen.queryByRole("table")).toBeNull();
    const plot = {
      BarChart: ".recharts-bar",
      LineChart: ".recharts-line",
      PieChart: ".recharts-pie",
    }[component];
    await waitFor(() => expect(container.querySelector(plot!)).not.toBeNull());
    expect(container.querySelector('[data-slot="chart"]')).not.toBeNull();
    expect(screen.getByRole("application").getAttribute("tabindex")).toBe("0");
    if (component === "LineChart") {
      const path = container
        .querySelector(".recharts-line-curve")
        ?.getAttribute("d");
      expect(path?.match(/M/g)).toHaveLength(2);
    }
    if (component === "PieChart") {
      expect(screen.getAllByText("Apples")).toHaveLength(1);
      expect(screen.queryByText("Pears")).toBeNull();
    }
  },
);

test("pie labels are rendered as text, never as generated CSS keys", async () => {
  const label = '</style><img src=x onerror="alert(1)">';
  const { container } = render(
    <Results
      messages={surface([
        { id: "root", component: "PieChart", points: [{ label, value: 42 }] },
      ])}
    />,
  );
  await waitFor(() => expect(screen.getAllByText(label)).toHaveLength(1));
  expect(container.querySelector("img")).toBeNull();
  const style = container.querySelector(
    '[data-slot="chart"] style',
  )?.textContent;
  expect(style).toContain("--color-slice_0");
  expect(style).not.toContain(label);
});

test("a pie with zero or missing values reports an empty plot", async () => {
  render(
    <Results
      messages={surface([
        {
          id: "root",
          component: "PieChart",
          points: [
            { label: "Zero", value: 0 },
            { label: "Unknown", value: null },
          ],
        },
      ])}
    />,
  );
  await waitFor(() =>
    expect(screen.getByText("No positive values to display.")).toBeTruthy(),
  );
  expect(screen.queryByRole("table")).toBeNull();
});

test("backend placement moves widgets between center and sidebar without controls or visible IDs", async () => {
  const messages = [
    ...surface([bar("First result")], "first"),
    ...surface([bar("Second result")], "second"),
  ];
  const { rerender } = render(<Results messages={messages} />);
  await waitFor(() => expect(screen.getByText("Second result")).toBeTruthy());
  const second = screen.getByRole("article", { name: "Widget second" });
  expect(
    screen.getByRole("region", { name: "Center widgets" }).contains(second),
  ).toBe(true);
  expect(screen.queryByRole("button")).toBeNull();
  expect(screen.queryByText("second")).toBeNull();
  rerender(
    <Results
      messages={messages}
      placements={{
        first: { pinned: false, focused: true },
        second: { pinned: true, focused: false },
      }}
    />,
  );
  expect(
    screen
      .getByRole("complementary", { name: "Side widgets" })
      .contains(screen.getByRole("article", { name: "Widget second" })),
  ).toBe(true);
  rerender(
    <Results
      messages={messages}
      placements={{
        first: { pinned: false, focused: true },
        second: { pinned: false, focused: true },
      }}
    />,
  );
  expect(screen.queryByRole("complementary")).toBeNull();
  expect(screen.getAllByRole("article")[0].getAttribute("aria-label")).toBe(
    "Widget first",
  );
});

test("an in-place widget update retains its sidebar placement", async () => {
  const messages = surface([bar("Original")]);
  const placements = { sales: { pinned: true, focused: false } };
  const { rerender } = render(
    <Results messages={messages} placements={placements} />,
  );
  await waitFor(() => expect(screen.getByText("Original")).toBeTruthy());
  const updated = a2uiBatchSchema.parse([
    {
      version: "v0.9",
      updateComponents: {
        surfaceId: "sales",
        components: [bar("Updated")],
      },
    },
  ]);
  rerender(
    <Results messages={[...messages, ...updated]} placements={placements} />,
  );
  await waitFor(() => expect(screen.getByText("Updated")).toBeTruthy());
  expect(screen.queryByText("Original")).toBeNull();
  expect(screen.getAllByRole("article")).toHaveLength(1);
  expect(
    screen.getByRole("complementary").contains(screen.getByRole("article")),
  ).toBe(true);
});

test("the center holds at most four widgets and overflow stays visible on the side", async () => {
  const messages = [1, 2, 3, 4, 5].flatMap((index) =>
    surface([bar(`Chart ${index}`)], `chart_${index}`),
  );
  render(<Results messages={messages} />);
  await waitFor(() => expect(screen.getByText("Chart 5")).toBeTruthy());
  expect(
    screen
      .getByRole("region", { name: "Center widgets" })
      .querySelectorAll("article"),
  ).toHaveLength(4);
  expect(
    screen.getByRole("complementary").querySelectorAll("article"),
  ).toHaveLength(1);
  expect(screen.getByRole("complementary").textContent).toContain("Chart 1");
});
