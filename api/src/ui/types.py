from typing import Literal, Annotated
from pydantic import Field

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
TextValue = Annotated[str, Field(max_length=10000)]
Number = Annotated[float, Field(allow_inf_nan=False)]
Cell = TextValue | int | Number | bool | None
Action = Literal["focus", "unfocus", "pin", "unpin"]
