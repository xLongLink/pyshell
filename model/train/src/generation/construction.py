class TaskRecord(dict):
    expression: str = ""
    extra: int | None = None
    alias_code: str = ""

    def __init__(self, record: dict, *, name: str, other: str | None = None):
        super().__init__(record)
        self.name = name
        self.other = other if other is not None else name + "_other"
