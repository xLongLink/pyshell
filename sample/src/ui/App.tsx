import { useWorkspace } from "@/lib/hooks/useWorkspace";
import { Voice } from "@/components/voice";

export default function App() {
    const { connected, error, sendVoice, voiceEvent } = useWorkspace();
    const widgets = new URL(location.href);
    widgets.port = "8000";
    widgets.pathname = "/";
    widgets.search = "";
    widgets.hash = "";

    return (
        <div className="flex h-dvh flex-col">
            {error && (
                <p role="alert" className="p-4 text-sm text-destructive">
                    {error}
                </p>
            )}
            <iframe
                title="Python workspace"
                src={widgets.href}
                sandbox="allow-scripts allow-same-origin"
                referrerPolicy="no-referrer"
                className="min-h-0 w-full flex-1 border-0"
            />
            <Voice connected={connected} event={voiceEvent} send={sendVoice} />
        </div>
    );
}
