import { useState, type FormEvent } from "react";
import { ArrowUp, MessageCircle, X } from "lucide-react";
import { Popover } from "radix-ui";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useWorkspace } from "@/lib/hooks/useWorkspace";
import { Results } from "@/components/results";
import { ErrorBoundary } from "react-error-boundary";
import { Voice } from "@/components/voice";

export default function App() {
    const {
        connected,
        busy,
        uiMessages,
        placements,
        error,
        send,
        sendVoice,
        voiceEvent,
        output,
    } = useWorkspace();
    const [instruction, setInstruction] = useState("");
    const [open, setOpen] = useState(false);

    function submit(event: FormEvent) {
        event.preventDefault();
        if (instruction.trim() && send(instruction.trim())) {
            setInstruction("");
            setOpen(false);
        }
    }

    return (
        <div className="flex h-dvh flex-col">
            <main className="min-h-0 flex-1 overflow-y-auto">
                <div className="space-y-6 p-4 md:p-8">
                    {error && (
                        <p role="alert" className="text-sm text-destructive">
                            {error}
                        </p>
                    )}
                    {!uiMessages.length && (
                        <p className="py-16 text-center text-sm text-muted-foreground">
                            Your generated widgets will appear here. Use the
                            microphone or chat button to describe what you want.
                        </p>
                    )}
                    <ErrorBoundary
                        resetKeys={[uiMessages]}
                        fallback={
                            <p
                                role="alert"
                                className="text-sm text-destructive"
                            >
                                Unable to render these widgets. Check the server
                                console for execution details.
                            </p>
                        }
                    >
                        <Results
                            messages={uiMessages}
                            placements={placements}
                        />
                    </ErrorBoundary>
                    {output && (
                        <details className="text-sm">
                            <summary>Execution output</summary>
                            <pre className="max-h-64 overflow-auto whitespace-pre-wrap pt-2">
                                {output}
                            </pre>
                        </details>
                    )}
                </div>
            </main>
            <Voice connected={connected} event={voiceEvent} send={sendVoice} />
            <Popover.Root open={open} onOpenChange={setOpen}>
                <Popover.Trigger asChild>
                    <Button
                        className="fixed right-5 bottom-5 size-12 rounded-full shadow-lg"
                        aria-label="Open chat"
                    >
                        <MessageCircle />
                    </Button>
                </Popover.Trigger>
                <Popover.Portal>
                    <Popover.Content
                        aria-label="Chat"
                        side="top"
                        align="end"
                        sideOffset={12}
                        collisionPadding={20}
                        className="z-50 w-[calc(100vw-2.5rem)] max-w-sm rounded-xl border bg-background p-4 shadow-lg"
                    >
                        <div className="mb-3 flex items-center justify-between">
                            <h2 className="text-sm font-medium">
                                Ask the agent
                            </h2>
                            <Popover.Close asChild>
                                <Button
                                    variant="outline"
                                    aria-label="Close chat"
                                >
                                    <X />
                                </Button>
                            </Popover.Close>
                        </div>
                        {(!connected || busy) && (
                            <p
                                role="status"
                                className="mb-3 text-sm text-muted-foreground"
                            >
                                {!connected ? "Connecting…" : "Working…"}
                            </p>
                        )}
                        <form className="space-y-3" onSubmit={submit}>
                            <label htmlFor="instruction" className="sr-only">
                                Instruction
                            </label>
                            <div className="flex items-end gap-3">
                                <Textarea
                                    id="instruction"
                                    value={instruction}
                                    onChange={(event) =>
                                        setInstruction(event.target.value)
                                    }
                                    maxLength={8000}
                                    placeholder="Describe what you want to create or change…"
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
                                    aria-label="Send instruction"
                                    disabled={
                                        !connected ||
                                        busy ||
                                        !instruction.trim()
                                    }
                                >
                                    <ArrowUp />
                                </Button>
                            </div>
                        </form>
                    </Popover.Content>
                </Popover.Portal>
            </Popover.Root>
        </div>
    );
}
