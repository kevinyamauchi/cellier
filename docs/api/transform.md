# Transform

Coordinate systems and the transforms between them.

A transform names the two coordinate systems it maps between, so it can say
*from where, to where* and not only *by how much*. That is what lets the
viewer refuse a transform built against a look-alike pair of spaces, and what
makes a pyramid's per-level maps composable without a downsampling
assumption.

::: cellier.transform
