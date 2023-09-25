# -*- coding: utf-8 -*-

import numpy as np
import PIL
import pydicom
from qtpy import QtGui

DICOM_TYPES = [
    "dcm",
    "dic",
    "dicom",
]

FRAME_TAG = [0x0028,0x0008]


def is_dicom_file(file_name: str):
    if not file_name:
        return False

    suffix = file_name.lower().split(".")[-1]
    return suffix in DICOM_TYPES


def get_all_supported_image_types():
    common_image_types = [
        fmt.data().decode().lower()
        for fmt in QtGui.QImageReader.supportedImageFormats()
    ]

    return common_image_types + DICOM_TYPES


def is_supported_image(file_name: str):
    if not file_name:
        return False

    suffix = file_name.lower().split(".")[-1]
    return suffix in get_all_supported_image_types()


def count_dicom_frame(dicom_data: pydicom.FileDataset):
    frame_data = dicom_data.get_item(FRAME_TAG)  # type: ignore
    if frame_data:
        frame_value = frame_data.value
        if frame_value:
            if isinstance(frame_value, bytes):
                return frame_value.decode()
            return frame_value

    pixel_shape = dicom_data.pixel_array.shape
    # frame, width, height, channel
    if len(pixel_shape) >= 4:
        return pixel_shape[0]

    # width, height
    if len(pixel_shape) <= 2:
        return 1

    # width, height, channel
    if pixel_shape[-1] <= 3:
        return 1

    # frame, width, height
    return pixel_shape[0]


def load_dicom_file(image_path: str):
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
        dicom_data = pydicom.dcmread(image_path)
        img = np.array(dicom_data.pixel_array).astype("float32")

        total_frame = count_dicom_frame(dicom_data)

        img = handle_intercept(dicom_data, img)
        img = normalize_img(img)

        # single frame
        if total_frame <= 1:
            yield PIL.Image.fromarray(img)
            return

        # multiple frames
        for frame in range(total_frame):
            yield PIL.Image.fromarray(img[frame])

    except IOError:
        return


def load_common_image(image_path: str):
    try:
        yield PIL.Image.open(image_path)
    except IOError:
        return


def load_image(image_path: str):
    if is_dicom_file(image_path):
        yield from load_dicom_file(image_path)
    else:
        yield from load_common_image(image_path)
