import { useWorkspace } from "@/lib/hooks/useWorkspace";
import { Renderer } from "@/components/renderer";
import { Voice } from "@/components/voice";

export default function App() {
    const { connected, uiMessages, placements, error, sendVoice, voiceEvent } =
        useWorkspace();

    return (
        <div className="h-dvh">
            <Renderer messages={uiMessages} placements={placements} />
            {error && (
                <p
                    role="alert"
                    className="fixed top-4 left-4 z-20 rounded bg-background p-3 text-sm text-destructive"
                >
                    {error}
                </p>
            )}
            <Voice
                docked={uiMessages.length > 0}
                connected={connected}
                event={voiceEvent}
                send={sendVoice}
            />
        </div>
    );
}
