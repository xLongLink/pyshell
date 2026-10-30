from typing import ClassVar, Protocol
from pydantic import BaseModel, ConfigDict, PrivateAttr
from src.ui.types import Action, Identifier


class Publisher(Protocol):
    """Publish widget mutations through their owning UI session."""

    def update(self, widget: "Widget", attributes: dict[str, object]) -> None:
        """Validate and publish attributes before changing the widget."""
        ...

    def action(self, widget: "Widget", action: Action) -> None:
        """Publish a workspace placement action."""
        ...

    def remove_widget(self, widget: "Widget") -> None:
        """Remove the widget's surface."""
        ...


class Widget(BaseModel):
    """A typed result with shared content and placement operations."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    arguments: ClassVar[tuple[str, ...]] = ()
    id: Identifier = "root"
    _owner: Publisher | None = PrivateAttr(default=None)
    _surface_id: str | None = PrivateAttr(default=None)

    @property
    def surface_id(self) -> str:
        """The stable ID assigned when the widget was published."""
        if self._surface_id is None:
            raise RuntimeError("Publish the widget before using its handle")
        return self._surface_id

    def _publisher(self) -> Publisher:
        if self._owner is None:
            raise RuntimeError("Publish the widget before using its handle")
        return self._owner

    def update(self, **attributes: object) -> None:
        """Publish validated changes without replacing the widget's identity."""
        self._publisher().update(self, attributes)

    def focus(self) -> None:
        """Bring this result into view and emphasize it."""
        self._publisher().action(self, "focus")

    def unfocus(self) -> None:
        """Remove this result's focus without affecting another result."""
        self._publisher().action(self, "unfocus")

    def pin(self) -> None:
        """Keep this result in the workspace's pinned section."""
        self._publisher().action(self, "pin")

    def unpin(self) -> None:
        """Return this result to normal workspace placement."""
        self._publisher().action(self, "unpin")

    def remove(self) -> None:
        """Remove this result and invalidate its handle."""
        self._publisher().remove_widget(self)
