from qtpy.QtWidgets import QGridLayout, QWidget


class QtQuadview(QWidget):
    """Qt widget for displaying 4 canvases.

    The quandrant indices correspond with the following:
    +-------------------+-------------------+
    |                   |                   |
    |         1         |         0         |
    |                   |                   |
    +-------------------+-------------------+
    |                   |                   |
    |         2         |         3         |
    |                   |                   |
    +-------------------+-------------------+


    """

    def __init__(
        self,
        widget_0: QWidget | None = None,
        widget_1: QWidget | None = None,
        widget_2: QWidget | None = None,
        widget_3: QWidget | None = None,
        parent=None,
    ):
        super().__init__(parent)

        # make the layout
        layout = QGridLayout()

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        if widget_0 is not None:
            widget_0.setParent(self)
            layout.addWidget(widget_0, 0, 1)
        if widget_1 is not None:
            widget_1.setParent(self)
            layout.addWidget(widget_1, 0, 0)
        if widget_2 is not None:
            widget_2.setParent(self)
            layout.addWidget(widget_2, 1, 0)
        if widget_3 is not None:
            widget_3.setParent(self)
            layout.addWidget(widget_3, 1, 1)

        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        self.setLayout(layout)
