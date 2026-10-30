import { afterEach, beforeEach, expect, test, vi } from "vitest";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
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
  "%s renders its plot and accessible data while preserving missing values",
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
    await waitFor(() =>
      expect(screen.getByRole("table", { name: "Revenue data" })).toBeTruthy(),
    );
    expect(screen.getByRole("cell", { name: "Missing" })).toBeTruthy();
    expect(screen.getByRole("cell", { name: "42" })).toBeTruthy();
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
      expect(screen.getAllByText("Apples")).toHaveLength(2);
      expect(screen.getAllByText("Pears")).toHaveLength(1);
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
  await waitFor(() => expect(screen.getAllByText(label)).toHaveLength(2));
  expect(container.querySelector("img")).toBeNull();
  const style = container.querySelector(
    '[data-slot="chart"] style',
  )?.textContent;
  expect(style).toContain("--color-slice_0");
  expect(style).not.toContain(label);
});

test("a pie with zero or missing values reports an empty plot without losing data", async () => {
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
  expect(screen.getByRole("cell", { name: "0" })).toBeTruthy();
  expect(screen.getByRole("cell", { name: "Missing" })).toBeTruthy();
});

test("widget controls emit actions and authoritative placement orders and focuses results", async () => {
  const messages = [
    ...surface([bar("First result")], "first"),
    ...surface([bar("Second result")], "second"),
  ];
  const onAction = vi.fn();
  const scroll = vi.fn();
  const { rerender } = render(
    <Results messages={messages} onAction={onAction} />,
  );
  await waitFor(() => expect(screen.getByText("Second result")).toBeTruthy());
  const second = screen.getByRole("article", { name: "Widget second" });
  second.scrollIntoView = scroll;
  fireEvent.click(screen.getByRole("button", { name: "Pin second" }));
  fireEvent.click(screen.getByRole("button", { name: "Focus second" }));
  expect(onAction.mock.calls).toEqual([
    ["second", "pin"],
    ["second", "focus"],
  ]);
  rerender(
    <Results
      messages={messages}
      placements={{
        first: { pinned: false, focused: false },
        second: { pinned: true, focused: true },
      }}
      onAction={onAction}
    />,
  );
  await waitFor(() => expect(document.activeElement).toBe(second));
  expect(scroll).toHaveBeenCalled();
  expect(screen.getAllByRole("article")[0]).toBe(second);
  fireEvent.click(screen.getByRole("button", { name: "Unpin second" }));
  fireEvent.click(screen.getByRole("button", { name: "Unfocus second" }));
  expect(onAction.mock.calls.slice(-2)).toEqual([
    ["second", "unpin"],
    ["second", "unfocus"],
  ]);
  rerender(
    <Results
      messages={messages}
      placements={{
        first: { pinned: false, focused: false },
        second: { pinned: false, focused: false },
      }}
      onAction={onAction}
    />,
  );
  await waitFor(() => expect(document.activeElement).not.toBe(second));
  expect(screen.getAllByRole("article")[0].getAttribute("aria-label")).toBe(
    "Widget first",
  );
});

test("an in-place widget update retains its pin and focus placement", async () => {
  const messages = surface([bar("Original")]);
  const placements = { sales: { pinned: true, focused: true } };
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
    screen
      .getByRole("button", { name: "Unpin sales" })
      .getAttribute("aria-pressed"),
  ).toBe("true");
  expect(document.activeElement).toBe(
    screen.getByRole("article", { name: "Widget sales" }),
  );
});
