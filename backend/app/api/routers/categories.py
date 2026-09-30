"""Category CRUD."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, status as http_status

from app.api.deps import Services, get_current_user, get_services, require_doc
from app.core.pagination import Page, PageParams
from app.models.base import serialize_doc, to_object_id
from app.models.category import CategoryCreate, CategoryOut, CategoryUpdate

router = APIRouter(prefix="/categories", tags=["categories"])


@router.get("", response_model=Page[CategoryOut])
def list_categories(
    params: PageParams = Depends(),
    category_type: str | None = Query(default=None, alias="type"),
    include_archived: bool = Query(default=False),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> Page[CategoryOut]:
    service = services.category_service()
    rows = service.list_categories(user["_id"], type=category_type, include_archived=include_archived)
    page_rows = [serialize_doc(row) for row in rows[params.skip : params.skip + params.limit]]
    return Page.build(items=[r for r in page_rows if r is not None], total=len(rows), params=params)


@router.post("", response_model=CategoryOut, status_code=http_status.HTTP_201_CREATED)
def create_category(
    payload: CategoryCreate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    return services.category_service().serialize(
        require_doc(
            services.category_service().create_category(user["_id"], payload.model_dump(mode="json")),
            "Category could not be created",
        )
    )


@router.get("/{category_id}", response_model=CategoryOut)
def get_category(
    category_id: str,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    return services.category_service().serialize(
        require_doc(
            services.category_service().require(user["_id"], to_object_id(category_id, field="category_id")),
            "Category not found",
        )
    )


@router.patch("/{category_id}", response_model=CategoryOut)
def update_category(
    category_id: str,
    changes: CategoryUpdate,
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> dict:
    cat = to_object_id(category_id, field="category_id")
    updated = services.category_service().update_category(user["_id"], cat, changes.model_dump(exclude_none=True))
    return services.category_service().serialize(require_doc(updated, "Category not found"))


@router.delete("/{category_id}", status_code=http_status.HTTP_204_NO_CONTENT)
def delete_category(
    category_id: str,
    force: bool = Query(default=False, description="Also unassign transactions that still use this category"),
    user: dict = Depends(get_current_user),
    services: Services = Depends(get_services),
) -> None:
    cat = to_object_id(category_id, field="category_id")
    services.category_service().delete_category(user["_id"], cat, force=force)
    return None