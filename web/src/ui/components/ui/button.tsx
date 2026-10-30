import * as React from "react";
import { cn } from "@/lib/utils";

function Button({
    className,
    variant = "default",
    ...props
}: React.ComponentProps<"button"> & { variant?: "default" | "outline" }) {
    return (
        <button
            data-slot="button"
            className={cn(
                "inline-flex shrink-0 items-center justify-center gap-2 rounded-md text-sm font-medium whitespace-nowrap transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 [&_svg]:size-4 [&_svg]:shrink-0",
                variant === "outline"
                    ? "border bg-background hover:bg-accent hover:text-accent-foreground"
                    : "bg-primary text-primary-foreground hover:bg-primary/90",
                "size-10",
                className,
            )}
            {...props}
        />
    );
}

export { Button };
