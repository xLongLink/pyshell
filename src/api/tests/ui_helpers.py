from api.ui.state import UIState


class RecordingUIState(UIState):
    def __init__(self):
        super().__init__()
        self.messages = []

    def publish(self, messages):
        super().publish(messages)
        self.messages.extend(messages)
