import { useEffect, useRef, useState, type FormEvent } from "react";
import { ArrowUp, LoaderCircle, RotateCcw, Terminal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useWorkspace } from "@/lib/hooks/useWorkspace";

export default function App() {
  const { connected, busy, model, entries, error, send } = useWorkspace();
  const [instruction, setInstruction] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [entries]);

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
      <header className="flex items-center justify-between gap-4 border-b px-5 py-4">
        <div className="flex items-center gap-3">
          <Terminal className="size-5 text-primary" aria-hidden="true" />
          <h1 className="font-semibold tracking-tight">PyShell</h1>
          <span role="status" className="text-xs text-muted-foreground">
            {connected ? (busy ? "Working" : "Connected") : "Connecting"}
          </span>
        </div>
        <Button
          variant="outline"
          disabled={!connected || busy}
          onClick={() => {
            if (
              window.confirm(
                "Reset the shared Python namespace and conversation for all tabs? Files will remain.",
              )
            )
              send({ type: "reset" });
          }}
        >
          <RotateCcw />
          Reset
        </Button>
      </header>
      <main className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-4xl space-y-6 px-5 py-8">
          {entries.length === 0 && (
            <section className="py-16">
              <p className="mb-3 text-sm font-medium text-primary">
                Your Python workspace
              </p>
              <h2 className="mb-3 text-3xl font-semibold tracking-tight">
                What would you like to do?
              </h2>
              <p className="max-w-lg text-muted-foreground">
                Give an instruction. Watch the agent write Python, run it in the
                container, and use the results to finish your task.
              </p>
            </section>
          )}
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
          {busy && (
            <p
              role="status"
              className="flex items-center gap-2 text-sm text-muted-foreground"
            >
              <LoaderCircle className="size-4 animate-spin" />
              Agent working…
            </p>
          )}
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
          <Textarea
            id="instruction"
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            maxLength={8000}
            placeholder="Ask the agent to work with Python…"
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
          <div className="flex items-center justify-between gap-4">
            <p className="text-xs text-muted-foreground">
              One shared container · {model || "Connecting"}
              <br />
              Generated code runs without approval. Files survive reset.
            </p>
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
