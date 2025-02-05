from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

import json

from labelme import utils
from labelme.logger import logger


class ScrollAreaPreview(QtWidgets.QScrollArea):
    def __init__(self, *args, **kwargs):
        super(ScrollAreaPreview, self).__init__(*args, **kwargs)

        self.setWidgetResizable(True)

        content = QtWidgets.QWidget(self)
        self.setWidget(content)

        lay = QtWidgets.QVBoxLayout(content)

        self.label = QtWidgets.QLabel(content)
        self.label.setWordWrap(True)

        lay.addWidget(self.label)

    def setText(self, text):
        self.label.setText(text)

    def setPixmap(self, pixmap):
        self.label.setPixmap(pixmap)

    def clear(self):
        self.label.clear()


class FileDialogPreview(QtWidgets.QFileDialog):
    def __init__(self, *args, **kwargs):
        super(FileDialogPreview, self).__init__(*args, **kwargs)
        self.setOption(self.DontUseNativeDialog, True)

        self.setLabelText(QtWidgets.QFileDialog.LookIn, self.tr("Look in:"))
        self.setLabelText(QtWidgets.QFileDialog.FileName, self.tr("File name:"))
        self.setLabelText(QtWidgets.QFileDialog.FileType, self.tr("Files of type:"))
        self.setLabelText(QtWidgets.QFileDialog.Accept, self.tr("Open"))
        self.setLabelText(QtWidgets.QFileDialog.Reject, self.tr("Cancel"))

        self.labelPreview = ScrollAreaPreview(self)
        self.labelPreview.setFixedSize(300, 300)
        self.labelPreview.setHidden(True)

        box = QtWidgets.QVBoxLayout()
        box.addWidget(self.labelPreview)
        box.addStretch()

        self.setFixedSize(self.width() + 300, self.height())
        self.layout().addLayout(box, 1, 3, 1, 1)
        self.currentChanged.connect(self.onChange)

    def onChange(self, path):
        if path.lower().endswith(".json"):
            with open(path, "r") as f:
                data = json.load(f)
                self.labelPreview.setText(
                    json.dumps(data, indent=4, sort_keys=False)
                )
            self.labelPreview.label.setAlignment(
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop
            )
            self.labelPreview.setHidden(False)
        else:
            pixmap = None
            try:
                if utils.is_supported_image(path):
                    imagePil = next(utils.load_image(path))
                    imageBytes = utils.preprocess_img(imagePil, path)
                    pixmap = QtGui.QPixmap.fromImage(QtGui.QImage.fromData(imageBytes))
            except Exception as err:
                logger.exception(f"Load image {path} got err: {str(err)}")

            if pixmap is None or pixmap.isNull():
                self.labelPreview.clear()
                self.labelPreview.setHidden(True)
            else:
                self.labelPreview.setPixmap(
                    pixmap.scaled(
                        self.labelPreview.width() - 30,
                        self.labelPreview.height() - 30,
                        QtCore.Qt.KeepAspectRatio,
                        QtCore.Qt.SmoothTransformation,
                    )
                )
                self.labelPreview.label.setAlignment(QtCore.Qt.AlignCenter)
                self.labelPreview.setHidden(False)
