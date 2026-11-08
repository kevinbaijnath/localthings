"""Prepare inspected, full-resolution landscape artwork without enlargement."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import io
import struct
import warnings
import zlib
from dataclasses import dataclass

from PIL import Image, ImageCms, ImageOps, JpegImagePlugin

_TARGET_SIZE = (3840, 2160)
_MAX_SOURCE_BYTES = 30 * 1024 * 1024
_MAX_SOURCE_PIXELS = 60_000_000
_MAX_OUTPUT_BYTES = 20 * 1024 * 1024
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_BLOCK_SIZE = 65536
_SRGB_PROFILE = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
_SRGB_BYTES = _SRGB_PROFILE.tobytes()
_HDR_MARKERS = (
    b"hdrgm:",
    b"hdr-gain-map",
    b"hdrgainmap",
    b"hdr_gain_map",
    b"urn:iso:std:iso:ts:21496",
)


class ImageValidationError(ValueError):
    """The source cannot meet the artwork preparation policy."""


@dataclass(frozen=True)
class PreparedArtwork:
    """Exact upload bytes and a small local gallery preview."""

    data: bytes
    thumbnail: bytes
    source_width: int
    source_height: int
    crop_width: int
    crop_height: int
    crop_axis: str
    digest: str


def _jpeg(image: Image.Image) -> bytes:
    with io.BytesIO() as output:
        image.save(output, format="JPEG", quality=95, subsampling=0, icc_profile=_SRGB_BYTES)
        return output.getvalue()


def _png_raster_size(payload: memoryview, size: tuple[int, int]) -> int:
    if len(payload) != 13:
        raise ImageValidationError("The image file is damaged or incomplete.")
    width, height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", payload
    )
    if depth > 8:
        raise ImageValidationError("Export a standard 8-bit SDR JPEG or PNG.")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}.get(color)
    if (
        (width, height) != size
        or channels is None
        or depth not in (1, 2, 4, 8)
        or (color in (2, 4, 6) and depth != 8)
        or compression != 0
        or filtering != 0
        or interlace not in (0, 1)
    ):
        raise ImageValidationError("The image file is damaged or incomplete.")
    passes = ((0, 0, 1, 1),)
    if interlace:
        passes = (
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        )
    expected = 0
    for x, y, dx, dy in passes:
        columns = max(0, (width - x + dx - 1) // dx)
        rows = max(0, (height - y + dy - 1) // dy)
        if columns and rows:
            expected += rows * (1 + (columns * channels * depth + 7) // 8)
    return expected


def _check_png(source: bytes, size: tuple[int, int]) -> None:
    # Pillow exposes 16-bit truecolor PNGs as RGB, so decoded mode loses this check.
    offset = 8
    has_color_profile = False
    unknown_color = False
    view = memoryview(source)
    inflater = zlib.decompressobj()
    expected = None
    inflated = 0
    while offset + 12 <= len(source):
        length = struct.unpack_from(">I", source, offset)[0]
        kind = source[offset + 4 : offset + 8]
        end = offset + 8 + length
        if end + 4 > len(source):
            raise ImageValidationError("The image file is damaged or incomplete.")
        # Pillow's verify stops before checking IEND's checksum.
        if (
            zlib.crc32(memoryview(source)[offset + 4 : end])
            != struct.unpack_from(">I", source, end)[0]
        ):
            raise ImageValidationError("The image file is damaged or incomplete.")
        if kind == b"IHDR":
            if expected is not None:
                raise ImageValidationError("The image file is damaged or incomplete.")
            expected = _png_raster_size(view[offset + 8 : end], size)
        if kind == b"IDAT":
            if expected is None:
                raise ImageValidationError("The image file is damaged or incomplete.")
            for start in range(offset + 8, end, _PNG_BLOCK_SIZE):
                pending = view[start : min(start + _PNG_BLOCK_SIZE, end)]
                while True:
                    budget = min(_PNG_BLOCK_SIZE, expected - inflated + 1)
                    block = inflater.decompress(pending, budget)
                    inflated += len(block)
                    if inflated > expected or inflater.unused_data:
                        raise ImageValidationError("The image file is damaged or incomplete.")
                    pending = inflater.unconsumed_tail
                    if not pending and len(block) < budget:
                        break
        if kind == b"acTL":
            raise ImageValidationError("Choose a single-frame JPEG or PNG.")
        if kind in (b"mDCv", b"cLLi") or (
            kind == b"cICP" and source[offset + 8 : end] != bytes((1, 13, 0, 1))
        ):
            raise ImageValidationError("Export a standard sRGB SDR JPEG or PNG.")
        if kind in (b"iCCP", b"sRGB"):
            has_color_profile = True
        if kind == b"gAMA" and source[offset + 8 : end] != struct.pack(">I", 45455):
            unknown_color = True
        if kind == b"cHRM" and source[offset + 8 : end] != struct.pack(
            ">8I", 31270, 32900, 64000, 33000, 30000, 60000, 15000, 6000
        ):
            unknown_color = True
        offset = end + 4
        if kind == b"IEND":
            break
    if not inflater.eof or inflated != expected:
        raise ImageValidationError("The image file is damaged or incomplete.")
    if unknown_color and not has_color_profile:
        raise ImageValidationError("Export the image with a standard sRGB color profile.")


def _check_metadata(image: Image.Image) -> None:
    values = [value for value in image.info.values() if isinstance(value, (str, bytes))]
    if isinstance(image, JpegImagePlugin.JpegImageFile):
        if image.bits != 8:
            raise ImageValidationError("Export a standard 8-bit SDR JPEG or PNG.")
        for _kind, value in image.applist:
            if value.startswith(b"MPF\x00"):
                raise ImageValidationError("Choose a single-frame JPEG or PNG.")
            values.append(value)
        if any(value.startswith(b"ICC_PROFILE\x00") for _kind, value in image.applist) and not (
            image.info.get("icc_profile")
        ):
            raise ImageValidationError("Export the image with a valid sRGB color profile.")
    for value in values:
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        if any(marker in encoded.lower() for marker in _HDR_MARKERS):
            raise ImageValidationError("Export a standard sRGB SDR JPEG or PNG.")


def _validate_jpeg(source: bytes, header: Image.Image) -> None:
    # Pillow suppresses libjpeg warnings and can pad incomplete scans with gray.
    try:
        library = ctypes.util.find_library("turbojpeg")
        if not library:
            raise OSError
        decoder = ctypes.CDLL(library)
        decoder.tjInitDecompress.argtypes = []
        decoder.tjInitDecompress.restype = ctypes.c_void_p
        decoder.tjDecompressHeader3.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
            *([ctypes.POINTER(ctypes.c_int)] * 4),
        ]
        decoder.tjDecompressHeader3.restype = ctypes.c_int
        decoder.tjDecompress2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_void_p,
            *([ctypes.c_int] * 5),
        ]
        decoder.tjDecompress2.restype = ctypes.c_int
        decoder.tjDestroy.argtypes = [ctypes.c_void_p]
        decoder.tjDestroy.restype = ctypes.c_int
    except (OSError, AttributeError):
        raise ImageValidationError("JPEG validation is unavailable. Try a PNG image.") from None
    handle = decoder.tjInitDecompress()
    if not handle:
        raise ImageValidationError("JPEG validation could not start.")
    error = "The JPEG is damaged or cannot be decoded completely."
    try:
        width, height, sampling, colorspace = (ctypes.c_int() for _ in range(4))
        data = ctypes.c_char_p(source)
        if decoder.tjDecompressHeader3(
            handle,
            data,
            len(source),
            ctypes.byref(width),
            ctypes.byref(height),
            ctypes.byref(sampling),
            ctypes.byref(colorspace),
        ) != 0 or (
            (width.value, height.value) != header.size
            or width.value <= 0
            or height.value <= 0
            or width.value * height.value > _MAX_SOURCE_PIXELS
        ):
            raise ImageValidationError(error)
        pixel_format, channels = {"L": (6, 1), "CMYK": (11, 4)}.get(header.mode, (0, 3))
        pixels = ctypes.create_string_buffer(width.value * height.value * channels)
        # STOPONWARNING rejects recoverable damage; LIMITSCANS caps progressive work at 500 scans.
        if (
            decoder.tjDecompress2(
                handle,
                data,
                len(source),
                pixels,
                width.value,
                0,
                height.value,
                pixel_format,
                8192 | 32768,
            )
            != 0
        ):
            raise ImageValidationError(error)
    except (MemoryError, OverflowError):
        raise ImageValidationError(
            "The JPEG could not be decoded within the image limits."
        ) from None
    finally:
        if decoder.tjDestroy(handle) != 0:
            raise ImageValidationError("JPEG validation could not finish.")


def _load(source: bytes) -> Image.Image:
    if not source.startswith((_PNG_SIGNATURE, b"\xff\xd8")):
        raise ImageValidationError("Choose a single-frame JPEG or PNG.")
    if source.startswith(b"\xff\xd8") and not source.endswith(b"\xff\xd9"):
        raise ImageValidationError("The image file is damaged or incomplete.")
    with Image.open(io.BytesIO(source), formats=("JPEG", "PNG")) as header:
        if header.format not in ("JPEG", "PNG"):
            raise ImageValidationError("Choose a single-frame JPEG or PNG.")
        if header.width * header.height > _MAX_SOURCE_PIXELS:
            raise ImageValidationError("The source must contain at most 60 million pixels.")
        if getattr(header, "n_frames", 1) != 1:
            raise ImageValidationError("Choose a single-frame JPEG or PNG.")
        if header.format == "PNG":
            _check_png(source, header.size)
        _check_metadata(header)
        header.verify()
        if header.format == "JPEG":
            _validate_jpeg(source, header)
    with Image.open(io.BytesIO(source), formats=("JPEG", "PNG")) as original:
        original.load()
        _check_metadata(original)
        return ImageOps.exif_transpose(original)


def _to_srgb(image: Image.Image) -> Image.Image:
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "CMYK"):
        raise ImageValidationError("Export the image as a standard sRGB JPEG or PNG.")
    if ("A" in image.getbands() or "transparency" in image.info) and image.convert(
        "RGBA"
    ).getchannel("A").getextrema() != (255, 255):
        raise ImageValidationError("Export the image without transparency.")
    if "icc_profile" in image.info:
        profile = image.info["icc_profile"]
        if not isinstance(profile, bytes) or not profile:
            raise ImageValidationError("Export the image with a valid sRGB color profile.")
        try:
            input_profile = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            mode = image.mode
            if mode in ("1", "LA"):
                mode = "L"
            elif mode in ("P", "RGBA"):
                mode = "RGB"
            converted = ImageCms.profileToProfile(
                image.convert(mode), input_profile, _SRGB_PROFILE, outputMode="RGB"
            )
        except (OSError, ValueError, TypeError, ImageCms.PyCMSError):
            raise ImageValidationError(
                "Export the image with a valid sRGB color profile."
            ) from None
        assert converted is not None
        return converted
    if image.mode == "CMYK":
        raise ImageValidationError("Export CMYK images with a valid sRGB color profile.")
    return image.convert("RGB")


def _crop(image: Image.Image, position: str) -> tuple[Image.Image, int, int, str]:
    width, height = image.size
    factor = min(width // 16, height // 9)
    crop_width, crop_height = factor * 16, factor * 9
    if crop_width < 3840 or crop_height < 2160:
        raise ImageValidationError("The crop must contain at least 3840 x 2160 source pixels.")
    axis = "none"
    positions = ("center",)
    if width * 9 > height * 16:
        axis, positions = "horizontal", ("left", "center", "right")
    elif width * 9 < height * 16:
        axis, positions = "vertical", ("top", "center", "bottom")
    if position not in positions:
        raise ImageValidationError("Choose a crop position for this image's available axis.")
    left, top = (width - crop_width) // 2, (height - crop_height) // 2
    if position == "left":
        left = 0
    elif position == "right":
        left = width - crop_width
    elif position == "top":
        top = 0
    elif position == "bottom":
        top = height - crop_height
    return (
        image.crop((left, top, left + crop_width, top + crop_height)),
        crop_width,
        crop_height,
        axis,
    )


def prepare_image(source: bytes, position: str = "center") -> PreparedArtwork:
    """Decode and prepare one exact 4K JPEG for preview and upload in an executor."""
    if len(source) > _MAX_SOURCE_BYTES:
        raise ImageValidationError("The original file must be at most 30 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            original = _load(source)
            width, height = original.size
            image, crop_width, crop_height, axis = _crop(_to_srgb(original), position)
            if image.size != _TARGET_SIZE:
                image = image.resize(_TARGET_SIZE, Image.Resampling.LANCZOS)
            image.info.clear()
            data = _jpeg(image)
            if len(data) > _MAX_OUTPUT_BYTES:
                raise ImageValidationError("The prepared JPEG exceeds the 20 MiB limit.")
            with Image.open(io.BytesIO(data), formats=("JPEG",)) as verified:
                verified.load()
                if verified.format != "JPEG" or verified.size != _TARGET_SIZE:
                    raise ImageValidationError("The prepared image could not be verified.")
            thumbnail = _jpeg(image.resize((480, 270), Image.Resampling.LANCZOS))
    except ImageValidationError:
        raise
    except (
        OSError,
        ValueError,
        SyntaxError,
        UserWarning,
        Image.DecompressionBombError,
        zlib.error,
    ):
        raise ImageValidationError("The image file is damaged or incomplete.") from None
    return PreparedArtwork(
        data,
        thumbnail,
        width,
        height,
        crop_width,
        crop_height,
        axis,
        hashlib.sha256(data).hexdigest(),
    )
