from typing import TYPE_CHECKING
from pydantic import PrivateAttr
from src.ui.model import UIModel
from src.ui.types import Identifier

if TYPE_CHECKING:
    from src.ui.session import UI


class Widget(UIModel):
    """A typed result with shared content and placement operations."""

    id: Identifier = "root"

    # Publication metadata is private and excluded from the wire model.
    _owner: "UI | None" = PrivateAttr(default=None)
    _surface_id: str | None = PrivateAttr(default=None)

    @property
    def surface_id(self) -> str:
        """The stable ID assigned when the widget was published."""
        if self._surface_id is None:
            raise RuntimeError("Publish the widget before using its handle")
        return self._surface_id

    def _publisher(self) -> "UI":
        if self._owner is None:
            raise RuntimeError("Publish the widget before using its handle")
        return self._owner

    def update(self, **attributes: object) -> None:
        """Publish validated changes without replacing the widget's identity."""
        self._publisher().update(self, attributes)

    def focus(self) -> None:
        """Move this widget into the center, which holds up to four widgets."""
        self._publisher().action(self, "focus")

    def unfocus(self) -> None:
        """Move this widget out of the center into the sidebar."""
        self._publisher().action(self, "unfocus")

    def pin(self) -> None:
        """Move this widget into the sidebar."""
        self._publisher().action(self, "pin")

    def unpin(self) -> None:
        """Move this widget back into the center."""
        self._publisher().action(self, "unpin")

    def remove(self) -> None:
        """Remove this result and invalidate its handle."""
        self._publisher().remove_widget(self)
