# Paint

Paint controllers apply label edits to data stores, with undo/redo history and
write buffering for multiscale data.

## Controllers

::: cellier.paint.AbstractPaintController
::: cellier.paint.SyncPaintController
::: cellier.paint.MultiscalePaintController

## History

::: cellier.paint.ActiveStroke
::: cellier.paint.CommandHistory
::: cellier.paint.PaintStrokeCommand

## Write buffers & layers

::: cellier.paint.WriteBuffer
::: cellier.paint.TensorStoreWriteBuffer
::: cellier.paint.WriteLayer
::: cellier.paint.BrickKey
