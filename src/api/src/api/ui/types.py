from typing import Literal, TypeVar, Annotated, get_args
from pydantic import Field, GetPydanticSchema, GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

T = TypeVar("T")


def _immutable_list_schema(source: object, handler: GetCoreSchemaHandler) -> CoreSchema:
    schema = handler(list[get_args(source)[0]])
    return core_schema.no_info_after_validator_function(
        tuple, schema, serialization=core_schema.plain_serializer_function_ser_schema(list, return_schema=schema)
    )


ImmutableList = Annotated[tuple[T, ...], GetPydanticSchema(_immutable_list_schema)]

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
TextValue = Annotated[str, Field(max_length=10000)]
Number = Annotated[float, Field(allow_inf_nan=False)]
Action = Literal["focus", "pin"]
