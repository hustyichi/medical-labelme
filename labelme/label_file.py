import base64
import contextlib
import io
import json
import os
import os.path as osp
from typing import Optional, List

import numpy as np
import PIL.Image
import pydicom

from labelme import __version__
from labelme.logger import logger
from labelme import PY2
from labelme import QT4
from labelme import utils
from labelme.shape import Shape
from labelme.shape import make_shape


PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)
    return


class LabelFileError(Exception):
    pass


# the mininal image unit
class FrameLabel(object):
    def __init__(
        self,
        image_path: str,
        image_pil: PIL.Image.Image,
        frame: int = 0,
        image_data: Optional[bytes] = None,
        shapes: Optional[List[Shape]] = None,
        app_config: Optional[dict] = None,
    ) -> None:
        self.image_path = image_path
        self.frame = frame
        self.image_data_bytes = image_data
        self.image_pil = image_pil
        self.shapes = shapes or []
        self.app_config = app_config or {}

    @property
    def image_data(self) -> bytes:
        if self.image_data_bytes:
            return self.image_data_bytes

        self.image_data_bytes = utils.preprocess_img(self.image_pil, self.image_path)
        return self.image_data_bytes

    def add_shape(self, shape: Shape):
        self.shapes.append(shape)

    def remove_shapes(self, shapes: List[Shape]):
        for shape in shapes:
            self.shapes.remove(shape)

    def format(self):
        return {
            "frame": self.frame,
            "shapes": [shape.format() for shape in self.shapes],
            "imageWidth": self.image_pil.width,
            "imageHeight": self.image_pil.height,
        }

    def load(self, format_data: dict):
        if (
            self.image_pil.width != format_data["imageWidth"]
            or self.image_pil.height != format_data["imageHeight"]
        ):
            raise LabelFileError(f"Frame {self.frame} width or height not match")

        for shape_data in format_data["shapes"]:
            shape = make_shape(shape_data, self.app_config.get("label_flags", {}))
            self.add_shape(shape)


class ImageLabel(object):
    suffix = ".json"

    def __init__(
        self,
        file_path: Optional[str] = None,
        app_config: Optional[dict] = None,
    ) -> None:
        self.image_path = ""
        self.label_path = ""
        self.frame_labels: List[FrameLabel] = []
        self.current_frame = 0
        self.total_frame = 0
        self.other_data: dict = {}
        self.app_config: dict = app_config or {}
        self.flags = {k: False for k in self.app_config.get("flags") or []}

        if file_path:
            self.load(file_path)

    def load_image(self, image_path: str):
        if not osp.exists(image_path):
            raise LabelFileError(f"Image {image_path} not exist")

        frame = 0
        for image_pil in utils.load_image(image_path):
            self.frame_labels.append(
                FrameLabel(
                    image_path,
                    image_pil,
                    frame=frame,
                    app_config=self.app_config,
                )
            )
            frame += 1

        self.image_path = image_path
        self.total_frame = len(self.frame_labels)

    def load_label_file(self, label_path: str):
        if not osp.exists(label_path):
            raise LabelFileError("Label file not exist")

        try:
            with open(label_path, "r") as f:
                data = json.load(f)
                relative_image_path = data.pop("imagePath")
        except Exception as e:
            raise LabelFileError(f"Parsed label file {label_path} failed")

        self.label_path = label_path
        image_path = osp.join(osp.dirname(label_path), relative_image_path)
        self.load_image(image_path)

        self.flags.update(data.pop("flags", {}))

        frame_shapes_data = data.pop("frames", [])
        version = data.pop("version", None)
        totalFrame = data.pop("totalFrames", 0)
        self.other_data = data.copy()

        logger.info(
            f"Load label file {label_path} with version {version} parsed successfully, got {totalFrame} frames"
        )

        # parse frame shape
        for frame_data in frame_shapes_data:
            frame_idx = int(frame_data.get("frame", -1))

            if frame_idx < 0 or frame_idx >= self.total_frame:
                raise LabelFileError(f"Frame shape data {frame_data} parse failed")

            self.frame_labels[frame_idx].load(frame_data)

    def next_frame(self):
        if self.current_frame >= self.total_frame - 1:
            return False

        self.current_frame += 1
        return True

    def prev_frame(self):
        if self.current_frame <= 0:
            return False

        self.current_frame -= 1
        return True

    @property
    def current_frame_label(self) -> Optional[FrameLabel]:
        if self.current_frame >= len(self.frame_labels):
            return None

        return self.frame_labels[self.current_frame]

    @property
    def current_frame_image_data(self) -> Optional[bytes]:
        current_label = self.current_frame_label
        if not current_label:
            return None

        return current_label.image_data

    @property
    def current_frame_shapes(self) -> list[Shape]:
        current_label = self.current_frame_label
        if not current_label:
            return []

        return current_label.shapes

    def add_shape(self, shape: Shape):
        current_label = self.current_frame_label
        if not current_label:
            return

        current_label.add_shape(shape)

    def remove_shapes(self, shapes: List[Shape]):
        current_label = self.current_frame_label
        if not current_label:
            return

        current_label.remove_shapes(shapes)

    # load a image file or label file
    def load(self, file_path: str):
        if utils.is_supported_image(file_path):
            self.load_image(file_path)
        elif ImageLabel.is_label_file(file_path):
            self.load_label_file(file_path)
        else:
            raise LabelFileError("Not support file type")

    def format(self) -> dict:
        data = self.other_data
        data.update(
            dict(
                version=__version__,
                flags=self.flags,
                imagePath=osp.relpath(self.image_path, osp.dirname(self.label_path)),
                totalFrames=self.total_frame,
                frames=[frame.format() for frame in self.frame_labels if frame.shapes],
            )
        )
        return data

    # save image label to local file
    def save(self, label_path: str = ""):
        if not label_path:
            if not self.label_path:
                raise LabelFileError("Label path not specified")
        else:
            self.label_path = label_path

        label_parent = osp.dirname(self.label_path)
        if not osp.exists(label_parent):
            os.makedirs(label_parent)

        try:
            with open(self.label_path, "w") as f:
                json.dump(self.format(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(file_path):
        return osp.splitext(file_path)[1].lower() == ImageLabel.suffix


class LabelFile(object):
    suffix = ".json"

    def __init__(self, filename=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_dicom_file(filename: str, frame: int = 0):
        def handle_intercept(dicom_data: pydicom.FileDataset, img_data: np.ndarray):
            if "RescaleIntercept" in dicom_data:
                img_data += int(dicom_data.RescaleIntercept)

            return img_data

        def normalize_img(img_data: np.ndarray):
            min_, max_ = float(np.min(img_data)), float(np.max(img_data))
            normalize_img_data = (img_data - min_) / (max_ - min_) * 255
            trans_img_data = np.uint8(normalize_img_data)
            return trans_img_data

        try:
            dicom_data = pydicom.dcmread(filename)
            img = np.array(dicom_data.pixel_array).astype("float32")

            total_frame = utils.count_dicom_frame(dicom_data)
            if total_frame > 1:
                img = img[frame]

            img = handle_intercept(dicom_data, img)
            img = normalize_img(img)

            image_pil = PIL.Image.fromarray(img)
            return image_pil, total_frame
        except IOError:
            logger.error(f"Failed to load dicom file: {filename}")
            return None, 0

    @staticmethod
    def load_common_image_file(filename: str):
        try:
            image_pil = PIL.Image.open(filename)
            return image_pil
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

    @staticmethod
    def load_image_file(filename: str, frame: int = 0):
        total_frame = 0

        if utils.is_dicom_file(filename):
            image_pil, total_frame = LabelFile.load_dicom_file(filename, frame)
        else:
            image_pil = LabelFile.load_common_image_file(filename)
            total_frame = 1

        if not image_pil:
            return None, 0

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read(), total_frame

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
        ]
        shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
            "description",
        ]
        try:
            with open(filename, "r") as f:
                data = json.load(f)

            if data["imageData"] is not None:
                imageData = base64.b64decode(data["imageData"])
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            else:
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                imageData, _ = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            imagePath = data["imagePath"]
            self._check_image_height_and_width(
                base64.b64encode(imageData).decode("utf-8"),
                data.get("imageHeight"),
                data.get("imageWidth"),
            )
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    description=s.get("description"),
                    group_id=s.get("group_id"),
                    other_data={
                        k: v for k, v in s.items() if k not in shape_keys
                    },
                )
                for s in data["shapes"]
            ]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.filename = filename
        self.otherData = otherData

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
        )
        for key, value in otherData.items():
            assert key not in data
            data[key] = value
        try:
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
