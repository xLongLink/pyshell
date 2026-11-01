from api.ui.state import UIState


def creation(identifier):
    return {"createSurface": {"surfaceId": identifier}}


def content(identifier, title="Initial"):
    return {"updateComponents": {"surfaceId": identifier, "components": [{"layout": {"title": {"text": title}}, "data": []}]}}


class RecordingUIState(UIState):
    def __init__(self):
        super().__init__()
        self.messages = []

    def publish(self, messages):
        super().publish(messages)
        self.messages.extend(messages)
