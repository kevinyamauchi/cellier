"""The discriminated union over every concrete transform kind."""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import Field

from cellier.transform._affine import AffineTransform
from cellier.transform._by_dimension import ByDimensionTransform
from cellier.transform._nonuniform import NonUniformAxisTransform

__all__ = ["TransformType"]

TransformType = Annotated[
    Union[
        AffineTransform,
        ByDimensionTransform,
        NonUniformAxisTransform,
    ],
    Field(discriminator="transform_type"),
]
"""Discriminated union over every concrete transform kind.

**This, not ``BaseTransform``, is what a pydantic model field should be
annotated with.**  ``BaseTransform`` is abstract: a field typed with it
loses the concrete subclass's validator and serializer, and a round-trip
through JSON raises ``PydanticSerializationError`` on the wrapped
``transformnd`` object.  The ``transform_type`` discriminator each concrete
class carries is what lets pydantic pick the right one back out again.

Use ``BaseTransform`` for ordinary function annotations, where it says
"anything satisfying the contract" and no serialization is involved.
"""
