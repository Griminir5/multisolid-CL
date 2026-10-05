"""Field locations shared by validation and interactive input editors."""

from dataclasses import dataclass

from pydantic_core import PydanticCustomError


@dataclass(frozen=True)
class InputIssue:
    message: str
    paths: tuple[tuple, ...]


def field_error(message, *paths):
    """Attach relative field paths to rules involving more than one field."""
    return PydanticCustomError(
        "value_error", "Value error, {error}", {"error": message, "fields": paths}
    )


def model_issues(error, prefix=()):
    for detail in error.errors():
        # Pydantic includes discriminated-union tags in locations; these are
        # schema branches, not keys in an authored step.
        location = tuple(part for i, part in enumerate(detail["loc"])
                         if not (part in ("hold", "ramp") and i > 0
                                 and isinstance(detail["loc"][i - 1], int)))
        fields = detail.get("ctx", {}).get("fields", ((),))
        message = "Required value is missing." if detail["type"] == "missing" else detail["msg"]
        if detail.get("input") == "" and detail["type"] in ("float_type", "int_type"):
            message = "Enter a number."
        yield InputIssue(message, tuple(prefix + location + tuple(path) for path in fields))
