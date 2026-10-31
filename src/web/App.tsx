import { useWorkspace } from "@/lib/hooks/useWorkspace";
import { Results } from "@/components/results";
import { ErrorBoundary } from "react-error-boundary";

export default function App() {
    const { surfaces, placements, error } = useWorkspace();

    return (
        <div className="flex h-dvh flex-col">
            <main className="min-h-0 flex-1 overflow-y-auto">
                <div className="space-y-6 p-4 md:p-8">
                    {error && (
                        <p role="alert" className="text-sm text-destructive">
                            {error}
                        </p>
                    )}
                    <ErrorBoundary
                        resetKeys={[surfaces]}
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
                        <Results surfaces={surfaces} placements={placements} />
                    </ErrorBoundary>
                </div>
            </main>
        </div>
    );
}
