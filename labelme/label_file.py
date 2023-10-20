import contextlib
import io
import json
import os
import os.path as osp
from typing import Optional, List

import PIL.Image

from labelme import __version__
from labelme.logger import logger
from labelme import PY2
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
            try:
                self.shapes.remove(shape)
            except ValueError:
                logger.exception(f"Remove not exist shape {shape.label} with points {shape.points}")

    def load_shapes(self, shapes: List[Shape], replace:bool = True):
        if replace:
            self.shapes = []

        for shape in shapes:
            self.add_shape(shape)

    def format(self):
        return {
            "frame": self.frame,
            "shapes": [shape.format() for shape in self.shapes],
            "width": self.image_pil.width,
            "height": self.image_pil.height,
        }

    def load(self, format_data: dict):
        if (
            self.image_pil.width != format_data["width"]
            or self.image_pil.height != format_data["height"]
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
                relative_image_path = data.pop("image_path")
        except Exception as e:
            raise LabelFileError(f"Parsed label file {label_path} failed")

        self.label_path = label_path
        image_path = osp.join(osp.dirname(label_path), relative_image_path)
        self.load_image(image_path)

        self.flags.update(data.pop("flags", {}))

        frame_shapes_data = data.pop("frames", [])
        version = data.pop("version", None)
        total_frames = data.pop("total_frames", 0)
        self.other_data = data.copy()

        logger.info(
            f"Load label file {label_path} with version {version} parsed successfully, got {total_frames} frames"
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

    def load_shapes(self, shapes: List[Shape], replace: bool = True):
        current_label = self.current_frame_label
        if not current_label:
            return

        current_label.load_shapes(shapes, replace)

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
                image_path=osp.relpath(self.image_path, osp.dirname(self.label_path)),
                total_frames=self.total_frame,
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

    @staticmethod
    def is_matched_label_file(image_path: str, label_path: str):
        if not (osp.exists(image_path) and osp.exists(label_path)):
            return False

        try:
            with open(label_path, "r") as f:
                data = json.load(f)
                relative_image_path = data.pop("image_path")
        except Exception as e:
            return False

        matched_image_path = osp.join(osp.dirname(label_path), relative_image_path)
        return osp.samefile(image_path, matched_image_path)
