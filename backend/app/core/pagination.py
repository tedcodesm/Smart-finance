"""Generic pagination + sorting envelopes."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")

MAX_PAGE_SIZE = 200
DEFAULT_PAGE_SIZE = 25


class PageParams(BaseModel):
    """Query parameters for every list endpoint.

    Two interchangeable styles are accepted:

    * page based  - ``?page=2&page_size=50``
    * offset based - ``?skip=50&limit=50``

    They are exposed under the ``skip``/``limit`` aliases because the offset form
    is the documented public contract. Supplying both is allowed; the offset
    values win, as they are the more specific instruction.
    """

    model_config = ConfigDict(populate_by_name=True)

    page: int = Field(default=1, ge=1, description="1-based page number.")
    page_size: int = Field(
        default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE, description=f"Rows per page (max {MAX_PAGE_SIZE})."
    )
    offset_in: int | None = Field(
        default=None, alias="skip", ge=0, description="Row offset. Overrides `page` when supplied."
    )
    limit_in: int | None = Field(
        default=None,
        alias="limit",
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Rows to return. Overrides `page_size` when supplied. Max {MAX_PAGE_SIZE}.",
    )

    @property
    def skip(self) -> int:
        if self.offset_in is not None:
            return self.offset_in
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        return self.page_size if self.limit_in is None else self.limit_in


class PageMeta(BaseModel):
    total: int = 0
    skip: int = 0
    limit: int = DEFAULT_PAGE_SIZE
    page: int = 1
    page_size: int = DEFAULT_PAGE_SIZE
    pages: int = 0
    has_next: bool = False
    has_prev: bool = False


class Page(BaseModel, Generic[T]):
    items: list[T] = Field(default_factory=list)
    meta: PageMeta = Field(default_factory=PageMeta)

    @classmethod
    def build(cls, items: list[T], total: int, params: PageParams) -> "Page[T]":
        size = params.limit
        pages = (total + size - 1) // size if size else 0
        skip = params.skip
        return cls(
            items=items,
            meta=PageMeta(
                total=total,
                skip=skip,
                limit=size,
                page=params.page,
                page_size=size,
                pages=pages,
                has_next=skip + size < total,
                has_prev=skip > 0,
            ),
        )


def build_sort(sort: str | None, allowed: dict[str, str], default: str = "-created_at") -> list[tuple[str, int]]:
    """Translate `"name, -created_at"` into a Mongo sort spec.

    `allowed` maps the public field name to the stored document path, which lets
    us keep internal fields (e.g. `period_key`) out of the public contract.
    """
    if not sort:
        return [(allowed.get(default.lstrip("-"), default.lstrip("-")), -1 if default.startswith("-") else 1)]

    spec: list[tuple[str, int]] = []
    for token in sort.split(","):
        token = token.strip()
        if not token:
            continue
        descending = token.startswith("-")
        field = token.lstrip("-+")
        path = allowed.get(field)
        if path is None:
            continue
        spec.append((path, -1 if descending else 1))
    return spec or [(allowed.get(default.lstrip("-"), "created_at"), -1)]


def to_camel(text: str) -> str:  # pragma: no cover - trivial
    head, *tail = text.split("_")
    return head + "".join(part.capitalize() for part in tail)


def public_keys(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    return doc
