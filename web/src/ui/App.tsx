import { useEffect, useRef, useState, type FormEvent } from "react";
import { ArrowUp } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useWorkspace } from "@/lib/hooks/useWorkspace";
import { Results } from "@/components/results";
import { ErrorBoundary } from "react-error-boundary";

export default function App() {
  const { connected, busy, entries, uiMessages, placements, error, send } =
    useWorkspace();
  const [instruction, setInstruction] = useState("");
  const bottom = useRef<HTMLDivElement>(null);
  const hasFocus = Object.values(placements).some(
    (placement) => placement.focused,
  );

  useEffect(() => {
    if (!hasFocus) bottom.current?.scrollIntoView({ block: "end" });
  }, [entries, hasFocus]);

  function submit(event: FormEvent) {
    event.preventDefault();
    if (
      instruction.trim() &&
      send({ type: "run", instruction: instruction.trim() })
    )
      setInstruction("");
  }

  return (
    <div className="flex h-dvh flex-col">
      <main className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-6 px-5 py-8">
          {entries.map((entry, index) => (
            <section
              key={index}
              className={
                entry.kind === "instruction"
                  ? "border-l-2 border-primary pl-4"
                  : ""
              }
            >
              <h2 className="mb-2 text-xs font-medium uppercase tracking-wider text-muted-foreground">
                {
                  {
                    instruction: "You",
                    code: "Python",
                    output: "Output",
                    error: "Error",
                    status: "Agent",
                  }[entry.kind]
                }
              </h2>
              {entry.kind === "code" ||
              entry.kind === "output" ||
              entry.kind === "error" ? (
                <pre
                  className={`overflow-x-auto whitespace-pre-wrap break-words rounded-lg border p-4 font-mono text-sm ${entry.kind === "error" ? "border-destructive/30 bg-destructive/5 text-destructive" : "bg-card"}`}
                >
                  <code>{entry.text || "Generating Python…"}</code>
                </pre>
              ) : (
                <p className="whitespace-pre-wrap text-sm leading-relaxed">
                  {entry.text}
                </p>
              )}
            </section>
          ))}
          <ErrorBoundary
            resetKeys={[uiMessages]}
            fallback={
              <p role="alert" className="text-sm text-destructive">
                Unable to render these results. Execution output is still
                available above.
              </p>
            }
          >
            <Results
              messages={uiMessages}
              placements={placements}
              onAction={
                connected
                  ? (surface_id, action) => {
                      send({ type: "widget", surface_id, action });
                    }
                  : undefined
              }
            />
          </ErrorBoundary>
          <div ref={bottom} />
        </div>
      </main>
      <footer className="border-t bg-background px-5 py-4">
        <form className="mx-auto max-w-4xl space-y-3" onSubmit={submit}>
          {error && (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          )}
          <label htmlFor="instruction" className="sr-only">
            Instruction
          </label>
          <div className="flex items-end gap-3">
            <Textarea
              id="instruction"
              value={instruction}
              onChange={(event) => setInstruction(event.target.value)}
              maxLength={8000}
              placeholder="Message…"
              disabled={!connected || busy}
              onKeyDown={(event) => {
                if (
                  event.key === "Enter" &&
                  !event.shiftKey &&
                  !event.nativeEvent.isComposing
                ) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
            />
            <Button
              type="submit"
              size="icon"
              aria-label="Send instruction"
              disabled={!connected || busy || !instruction.trim()}
            >
              <ArrowUp />
            </Button>
          </div>
        </form>
      </footer>
    </div>
  );
}
