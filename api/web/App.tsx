import { ErrorBoundary } from "react-error-boundary";
import { Results } from "@/components/results";
import { useRenderer } from "@/lib/useRenderer";

export default function App() {
    const { messages, placements, error } = useRenderer();
    return (
        <main className="min-h-dvh space-y-6 p-4 md:p-8">
            {error && (
                <p role="alert" className="text-sm text-destructive">
                    {error}
                </p>
            )}
            <ErrorBoundary
                resetKeys={[messages]}
                fallback={
                    <p role="alert" className="text-sm text-destructive">
                        Unable to render these widgets. Check the server console
                        for execution details.
                    </p>
                }
            >
                <Results messages={messages} placements={placements} />
            </ErrorBoundary>
        </main>
    );
}
