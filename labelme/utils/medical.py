# -*- coding: utf-8 -*-

from qtpy import QtGui

DICOM_TYPES = [
    "dcm",
    "dic",
    "dicom",
]


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
