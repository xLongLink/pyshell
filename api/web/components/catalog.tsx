import { createComponentImplementation } from "@a2ui/react/v0_9";
import { Catalog } from "@a2ui/web_core/v0_9";
import { Figure } from "@/components/plot/figure";
import { catalogId, figureProps } from "@/lib/a2ui";

const Plot = createComponentImplementation(
    { name: "Figure", schema: figureProps },
    ({ props }) => <Figure {...props} />,
);
export const resultCatalog = new Catalog(catalogId, "v0.9", [Plot]);
