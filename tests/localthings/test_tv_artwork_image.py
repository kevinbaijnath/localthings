"""Strict artwork preparation tests with synthetic images only."""

from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import io
import random
import struct
import zlib
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image, ImageCms, ImageDraw, JpegImagePlugin


def encode_image(size=(3840, 2160), *, mode="RGB", format="JPEG", **kwargs) -> bytes:
    image = Image.new(mode, size)
    with io.BytesIO() as output:
        image.save(output, format=format, **kwargs)
        return output.getvalue()


def test_exact_4k_produces_verified_jpeg_and_thumbnail() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")

    prepared = module.prepare_image(encode_image())

    assert (prepared.source_width, prepared.source_height) == (3840, 2160)
    assert (prepared.crop_width, prepared.crop_height, prepared.crop_axis) == (3840, 2160, "none")
    assert prepared.digest == hashlib.sha256(prepared.data).hexdigest()
    with Image.open(io.BytesIO(prepared.data)) as final:
        final.load()
        assert final.format == "JPEG"
        assert isinstance(final, JpegImagePlugin.JpegImageFile)
        assert final.size == (3840, 2160)
        assert JpegImagePlugin.get_sampling(final) == 0
        assert final.quantization[0][:8] == [2, 1, 1, 2, 2, 4, 5, 6]
    with Image.open(io.BytesIO(prepared.thumbnail)) as thumbnail:
        thumbnail.load()
        assert thumbnail.format == "JPEG"
        assert thumbnail.width <= 480
        assert thumbnail.height <= 270


@pytest.mark.parametrize("size", [(3840, 2159), (3839, 2160), (3000, 4000)])
def test_crop_cannot_enlarge_an_undersized_source(size) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")

    with pytest.raises(module.ImageValidationError, match="3840"):
        module.prepare_image(encode_image(size))


def test_four_by_three_source_uses_largest_exact_landscape_crop() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")

    prepared = module.prepare_image(encode_image((4000, 3000)))

    assert (prepared.source_width, prepared.source_height) == (4000, 3000)
    assert (prepared.crop_width, prepared.crop_height, prepared.crop_axis) == (
        4000,
        2250,
        "vertical",
    )


@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_format_is_detected_from_bytes(format) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    prepared = module.prepare_image(encode_image(format=format))
    assert prepared.data.startswith(b"\xff\xd8")


def test_exif_orientation_is_applied_before_resolution_check() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    exif = Image.Exif()
    exif[274] = 6
    prepared = module.prepare_image(encode_image((2160, 3840), exif=exif))
    assert (prepared.source_width, prepared.source_height) == (3840, 2160)
    with Image.open(io.BytesIO(prepared.data)) as image:
        assert not image.getexif()


@pytest.fixture(scope="module")
def vertical_source() -> bytes:
    image = Image.new("RGB", (3840, 2400), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 3839, 119), fill="red")
    draw.rectangle((0, 2280, 3839, 2399), fill="blue")
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize(
    ("position", "top", "bottom"),
    [
        ("top", (254, 0, 0), (255, 255, 255)),
        ("center", (255, 255, 255), (255, 255, 255)),
        ("bottom", (255, 255, 255), (0, 0, 254)),
    ],
)
def test_vertical_crop_position_changes_visible_pixels(vertical_source, position, top, bottom):
    module = import_module("custom_components.localthings.tv_artwork_image")
    prepared = module.prepare_image(vertical_source, position)
    with Image.open(io.BytesIO(prepared.data)) as image:
        assert image.getpixel((1920, 20)) == top
        assert image.getpixel((1920, 2140)) == bottom


def test_horizontal_crop_position_changes_visible_pixels() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    image = Image.new("RGB", (4320, 2160), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 119, 2159), fill="red")
    draw.rectangle((4200, 0, 4319, 2159), fill="blue")
    output = io.BytesIO()
    image.save(output, format="PNG")
    for position, point, expected in [
        ("left", (20, 1080), (254, 0, 0)),
        ("right", (3820, 1080), (0, 0, 254)),
    ]:
        prepared = module.prepare_image(output.getvalue(), position)
        assert prepared.crop_axis == "horizontal"
        with Image.open(io.BytesIO(prepared.data)) as final:
            assert final.getpixel(point) == expected


@pytest.mark.parametrize(
    ("size", "position"),
    [
        ((4000, 3000), "left"),
        ((4320, 2160), "top"),
        ((3840, 2160), "bottom"),
        ((3840, 2160), "invalid"),
    ],
)
def test_crop_position_must_match_available_axis(size, position) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    with pytest.raises(module.ImageValidationError, match="crop position"):
        module.prepare_image(encode_image(size), position)


@pytest.mark.parametrize("mode", ["RGB", "L", "1"])
@pytest.mark.parametrize("format", ["JPEG", "PNG"])
def test_unprofiled_sdr_rgb_and_grayscale_are_accepted(mode, format) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    prepared = module.prepare_image(encode_image(mode=mode, format=format))
    with Image.open(io.BytesIO(prepared.data)) as image:
        assert image.mode == "RGB"
        assert "sRGB" in ImageCms.getProfileDescription(io.BytesIO(image.info["icc_profile"]))


def test_valid_profile_is_converted_and_source_metadata_is_removed() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    exif = Image.Exif()
    exif[271] = "private-camera-make"
    exif[270] = "private-image-description"
    prepared = module.prepare_image(encode_image(icc_profile=profile, exif=exif))
    with Image.open(io.BytesIO(prepared.data)) as image:
        assert not image.getexif()
        assert "sRGB" in ImageCms.getProfileDescription(io.BytesIO(image.info["icc_profile"]))
    assert b"private" not in prepared.data
    assert b"private" not in prepared.thumbnail


def test_pixels_are_converted_instead_of_only_retagging_the_color_profile() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    for index in range(struct.unpack_from(">I", profile, 128)[0]):
        tag, offset, _size = struct.unpack_from(">4sII", profile, 132 + index * 12)
        if tag == b"rTRC":
            # LittleCMS shares this curve between the three RGB channels.
            struct.pack_into(">5I", profile, offset + 12, 65536, 65536, 0, 65536, 0)
            break
    else:
        pytest.fail("The generated RGB profile has no transfer curve")
    image = Image.new("RGB", (3840, 2160), (128, 128, 128))
    output = io.BytesIO()
    image.save(output, format="JPEG", icc_profile=bytes(profile))

    prepared = module.prepare_image(output.getvalue())

    with Image.open(io.BytesIO(prepared.data)) as final:
        assert final.getpixel((1920, 1080)) == (188, 188, 188)


@pytest.mark.parametrize("profile", [b"private-broken-profile", b""])
def test_invalid_embedded_profile_is_rejected(profile) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    original = encode_image(format="PNG")
    source = (
        original[:33]
        + png_chunk(b"iCCP", b"profile\x00\x00" + zlib.compress(profile))
        + original[33:]
    )
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(source)


def test_valid_profile_for_wrong_color_space_is_rejected() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(encode_image(icc_profile=profile))


def test_unprofiled_cmyk_is_rejected() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    with pytest.raises(module.ImageValidationError, match="sRGB"):
        module.prepare_image(encode_image(mode="CMYK"))


@pytest.mark.parametrize("mode", ["RGBA", "LA"])
def test_all_opaque_alpha_is_accepted(mode) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    image = Image.new(mode, (3840, 2160), (100, 100, 100, 255) if mode == "RGBA" else (100, 255))
    output = io.BytesIO()
    image.save(output, format="PNG")
    assert module.prepare_image(output.getvalue()).source_width == 3840


@pytest.mark.parametrize("mode", ["RGBA", "LA", "P"])
def test_nonopaque_alpha_and_palette_transparency_are_rejected(mode) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = (
        encode_image(mode=mode, format="PNG", transparency=0)
        if mode == "P"
        else encode_image(mode=mode, format="PNG")
    )
    with pytest.raises(module.ImageValidationError, match="transparency"):
        module.prepare_image(source)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def truecolor_png(width: int, height: int, *, depth: int = 8, pixels: bool = True) -> bytes:
    row = b"\x00" + bytes(width * 3 * (depth // 8))
    data = zlib.compress(row * height) if pixels else b""
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, 2, 0, 0, 0))
        + png_chunk(b"IDAT", data)
        + png_chunk(b"IEND", b"")
    )


def png_with_idat(parts: list[bytes], *, width=3840, height=2160, interlace=0) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, interlace))
        + b"".join(png_chunk(b"IDAT", part) for part in parts)
        + png_chunk(b"IEND", b"")
    )


@pytest.fixture(scope="module")
def complete_png_idat() -> bytes:
    return zlib.compress(bytes(11521 * 2160))


@pytest.mark.parametrize("split", [False, True])
def test_complete_png_stream_across_single_or_multiple_idat_chunks(complete_png_idat, split):
    module = import_module("custom_components.localthings.tv_artwork_image")
    parts = [complete_png_idat]
    if split:
        parts = [complete_png_idat[:1], complete_png_idat[1:-2], complete_png_idat[-2:]]
    assert module.prepare_image(png_with_idat(parts)).source_width == 3840


@pytest.mark.parametrize("removed", [1, 2, 3, 4])
def test_png_missing_zlib_trailer_rejected_despite_valid_chunk_crcs(complete_png_idat, removed):
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = png_with_idat([complete_png_idat[:-removed]])
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(source)


@pytest.mark.parametrize("rows", [2159, 2161])
def test_png_missing_or_extra_raster_rows_rejected_despite_complete_zlib(rows) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = png_with_idat([zlib.compress(bytes(11521 * rows))])
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(source)


@pytest.mark.parametrize("offset", [2, -1], ids=["deflate", "adler32"])
def test_png_corrupt_compressed_stream_has_sanitized_error(complete_png_idat, offset) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    damaged = bytearray(complete_png_idat)
    damaged[offset] ^= 0x80
    with pytest.raises(module.ImageValidationError) as error:
        module.prepare_image(png_with_idat([bytes(damaged)]))
    assert "Error -" not in str(error.value)


def test_adam7_png_with_odd_dimensions_passes_exact_raster_check() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    pass_sizes = [
        (501, 282),
        (500, 282),
        (1001, 281),
        (1000, 563),
        (2001, 563),
        (2000, 1126),
        (4001, 1125),
    ]
    raster = b"".join(bytes((1 + width * 3) * height) for width, height in pass_sizes)
    source = png_with_idat([zlib.compress(raster)], width=4001, height=2251, interlace=1)
    with Image.open(io.BytesIO(source)) as image:
        assert image.info["interlace"] == 1
        assert image.getpixel((4000, 2250)) == (0, 0, 0)
    prepared = module.prepare_image(source)
    assert (prepared.source_width, prepared.source_height) == (4001, 2251)
    assert (prepared.crop_width, prepared.crop_height) == (4000, 2250)


def test_png_pixel_limit_precedes_stream_inflation(monkeypatch) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = truecolor_png(10000, 6001, pixels=False)
    inflater = Mock(side_effect=AssertionError("Pixel limit must precede inflation"))
    monkeypatch.setattr(zlib, "decompressobj", inflater)
    with pytest.raises(module.ImageValidationError, match="60 million"):
        module.prepare_image(source)
    inflater.assert_not_called()


@pytest.mark.parametrize("tail", [b"extra", zlib.compress(b"extra")], ids=["bytes", "stream"])
def test_png_data_after_zlib_end_rejected(complete_png_idat, tail) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(png_with_idat([complete_png_idat + tail]))


def test_png_excess_expansion_stops_before_full_raster_allocation(monkeypatch) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = png_with_idat([zlib.compress(bytes(4 * 1024 * 1024))], width=8, height=8)
    real_factory = zlib.decompressobj
    output_sizes = []

    class TrackedInflater:
        def __init__(self):
            self.inner = real_factory()

        def decompress(self, data, max_length=0):
            assert 0 < max_length <= 65536
            output = self.inner.decompress(data, max_length)
            output_sizes.append(len(output))
            return output

        def __getattr__(self, name):
            return getattr(self.inner, name)

    monkeypatch.setattr(zlib, "decompressobj", TrackedInflater)
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(source)
    # Eight RGB rows need 200 bytes including their filters; only one extra byte is needed.
    assert sum(output_sizes) == 201


def test_truecolor_sixteen_bit_png_is_rejected_before_pillow_discards_precision() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = truecolor_png(3840, 2160, depth=16)
    with Image.open(io.BytesIO(source)) as decoded:
        assert decoded.mode == "RGB"
    with pytest.raises(module.ImageValidationError, match="8-bit"):
        module.prepare_image(source)


@pytest.mark.parametrize("chunk,data", [(b"cICP", bytes([9, 16, 0, 1])), (b"mDCv", bytes(24))])
def test_png_hdr_color_description_is_rejected(chunk, data) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = encode_image(format="PNG")
    source = source[:33] + png_chunk(chunk, data) + source[33:]
    with pytest.raises(module.ImageValidationError, match="SDR"):
        module.prepare_image(source)


@pytest.mark.parametrize("marker", [b"hdrgm:Version=1", b"urn:com:apple:photo:2020:aux:hdrgainmap"])
def test_jpeg_gain_map_metadata_is_rejected_even_when_base_is_rgb(marker) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = encode_image()
    metadata = b"http://ns.adobe.com/xap/1.0/\x00" + marker
    source = source[:2] + b"\xff\xe1" + struct.pack(">H", len(metadata) + 2) + metadata + source[2:]
    with Image.open(io.BytesIO(source)) as image:
        assert image.mode == "RGB"
    with pytest.raises(module.ImageValidationError, match="SDR"):
        module.prepare_image(source)


def test_animated_png_is_rejected() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = encode_image(
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (3840, 2160), "red")],
        duration=100,
    )
    with pytest.raises(module.ImageValidationError, match="single"):
        module.prepare_image(source)


@pytest.mark.parametrize("format", ["GIF", "BMP", "TIFF", "WEBP"])
def test_other_formats_are_rejected_even_with_sufficient_pixels(format) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    with pytest.raises(module.ImageValidationError, match="JPEG or PNG"):
        module.prepare_image(encode_image(format=format))


@pytest.mark.parametrize(
    "source",
    [b"private-image-secret", b"", encode_image()[:-50]],
    ids=["invalid", "empty", "truncated-jpeg"],
)
def test_invalid_and_truncated_files_raise_only_sanitized_validation_errors(source) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    with pytest.raises(module.ImageValidationError) as error:
        module.prepare_image(source)
    assert "private" not in str(error.value)


@pytest.fixture(scope="module", params=[False, True], ids=["baseline", "progressive"])
def noisy_jpeg(request) -> bytes:
    image = Image.frombytes("L", (3840, 2160), random.Random(0).randbytes(3840 * 2160))
    output = io.BytesIO()
    image.convert("RGB").save(output, format="JPEG", quality=95, progressive=request.param)
    return output.getvalue()


def test_jpeg_with_missing_entropy_and_retained_end_marker_is_rejected(noisy_jpeg) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    damaged = noisy_jpeg[: len(noisy_jpeg) // 2] + b"\xff\xd9"

    with pytest.raises(module.ImageValidationError):
        module.prepare_image(damaged)


def test_jpeg_with_short_missing_entropy_and_retained_end_marker_is_rejected(noisy_jpeg) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    damaged = noisy_jpeg[:-102] + b"\xff\xd9"

    with pytest.raises(module.ImageValidationError):
        module.prepare_image(damaged)


@pytest.mark.parametrize("removed", [2, 10, 1000])
def test_other_incomplete_jpeg_scans_are_rejected(noisy_jpeg, removed) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(noisy_jpeg[: -removed - 2] + b"\xff\xd9")


@pytest.mark.parametrize("progressive", [False, True])
def test_complete_baseline_and_progressive_jpeg_pass_strict_decode(progressive) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    assert module.prepare_image(encode_image(progressive=progressive)).source_width == 3840


@pytest.fixture
def native_decoder(monkeypatch):
    library = SimpleNamespace(
        tjInitDecompress=Mock(return_value=1),
        tjDecompressHeader3=Mock(return_value=0),
        tjDecompress2=Mock(return_value=0),
        tjDestroy=Mock(return_value=0),
    )

    def header(_handle, _source, _length, width, height, sampling, colorspace):
        for pointer, value in zip(
            (width, height, sampling, colorspace), (3840, 2160, 0, 1), strict=True
        ):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = value
        return 0

    library.tjDecompressHeader3.side_effect = header
    monkeypatch.setattr(ctypes.util, "find_library", Mock(return_value="synthetic-decoder"))
    monkeypatch.setattr(ctypes, "CDLL", Mock(return_value=library))
    return library


@pytest.mark.parametrize("mode, pixel_format", [("RGB", 0), ("L", 6), ("CMYK", 11)])
def test_jpeg_validation_uses_one_strict_native_colorspace_decode(
    native_decoder, mode, pixel_format
):
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = encode_image(mode=mode)
    if mode == "CMYK":
        with pytest.raises(module.ImageValidationError, match="sRGB"):
            module.prepare_image(source)
    else:
        assert module.prepare_image(source).source_width == 3840
    native_decoder.tjDecompress2.assert_called_once()
    args = native_decoder.tjDecompress2.call_args.args
    assert args[2] == len(source)
    assert args[4:8] == (3840, 0, 2160, pixel_format)
    assert args[8] == 8192 | 32768
    native_decoder.tjDestroy.assert_called_once_with(1)


def test_png_does_not_require_the_jpeg_decoder(monkeypatch) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    decoder = Mock(side_effect=AssertionError("PNG must not use JPEG decoding"))
    monkeypatch.setattr(ctypes.util, "find_library", decoder)
    prepared = module.prepare_image(encode_image(format="PNG"))
    assert prepared.source_width == 3840
    decoder.assert_not_called()


@pytest.mark.parametrize("stage", ["tjDecompressHeader3", "tjDecompress2"])
def test_jpeg_decoder_errors_are_sanitized_and_not_retried(native_decoder, stage) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    decoder = getattr(native_decoder, stage)
    decoder.side_effect = None
    decoder.return_value = -1
    with pytest.raises(module.ImageValidationError) as error:
        module.prepare_image(encode_image())
    assert "private" not in str(error.value)
    assert decoder.call_count == 1
    native_decoder.tjDestroy.assert_called_once_with(1)


def test_jpeg_header_disagreement_rejected_before_native_pixel_allocation(
    monkeypatch, native_decoder
) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    native_decoder.tjDecompressHeader3.side_effect = None
    allocation = Mock(side_effect=AssertionError("Mismatched bounds must not allocate pixels"))
    monkeypatch.setattr(ctypes, "create_string_buffer", allocation)
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(encode_image())
    allocation.assert_not_called()
    native_decoder.tjDecompress2.assert_not_called()
    native_decoder.tjDestroy.assert_called_once_with(1)


def test_jpeg_header_pixel_limit_precedes_external_decode(monkeypatch) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = bytearray(encode_image((8, 8)))
    sof = source.index(b"\xff\xc0")
    struct.pack_into(">HH", source, sof + 5, 6001, 10000)
    decoder = Mock(side_effect=AssertionError("The header limit must run first"))
    monkeypatch.setattr(ctypes.util, "find_library", decoder)
    with pytest.raises(module.ImageValidationError, match="60 million"):
        module.prepare_image(bytes(source))
    decoder.assert_not_called()


def test_jpeg_fails_closed_when_native_library_is_missing(monkeypatch) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    monkeypatch.setattr(ctypes.util, "find_library", lambda _name: None)
    with pytest.raises(module.ImageValidationError, match="unavailable"):
        module.prepare_image(encode_image())


@pytest.mark.parametrize("failure", ["library", "symbol", "initialize", "allocation"])
def test_native_decoder_boundary_failures_are_sanitized(monkeypatch, native_decoder, failure):
    module = import_module("custom_components.localthings.tv_artwork_image")
    if failure == "library":
        monkeypatch.setattr(ctypes, "CDLL", Mock(side_effect=OSError("private-library-path")))
    elif failure == "symbol":
        del native_decoder.tjDecompress2
    elif failure == "initialize":
        native_decoder.tjInitDecompress.return_value = None
    else:
        monkeypatch.setattr(
            ctypes, "create_string_buffer", Mock(side_effect=MemoryError("private-image-data"))
        )
    with pytest.raises(module.ImageValidationError) as error:
        module.prepare_image(encode_image())
    assert "private" not in str(error.value)
    if failure == "allocation":
        native_decoder.tjDestroy.assert_called_once_with(1)
    else:
        native_decoder.tjDestroy.assert_not_called()


def test_damaged_png_checksum_is_rejected() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = bytearray(encode_image(format="PNG"))
    source[-1] ^= 1
    with pytest.raises(module.ImageValidationError):
        module.prepare_image(bytes(source))


def test_source_byte_limit_precedes_decode() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    with pytest.raises(module.ImageValidationError, match="30 MiB"):
        module.prepare_image(bytes(30 * 1024 * 1024 + 1))


def test_pixel_limit_precedes_full_decode() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    with pytest.raises(module.ImageValidationError, match="60 million"):
        module.prepare_image(truecolor_png(10000, 6001, pixels=False))


def test_prepared_byte_limit_rejects_large_output(monkeypatch) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    monkeypatch.setattr(module, "_MAX_OUTPUT_BYTES", 1024)
    with pytest.raises(module.ImageValidationError, match="20 MiB"):
        module.prepare_image(encode_image())


def test_multipicture_jpeg_is_rejected() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = encode_image(
        (8, 8), format="MPO", save_all=True, append_images=[Image.new("RGB", (8, 8), "red")]
    )
    with pytest.raises(module.ImageValidationError, match="single"):
        module.prepare_image(source)


def test_unhandled_png_transfer_curve_requires_standard_srgb_export() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = encode_image(format="PNG")
    source = source[:33] + png_chunk(b"gAMA", struct.pack(">I", 100000)) + source[33:]
    with pytest.raises(module.ImageValidationError, match="sRGB"):
        module.prepare_image(source)


def test_png_srgb_transfer_curve_is_accepted() -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = encode_image(format="PNG")
    source = source[:33] + png_chunk(b"gAMA", struct.pack(">I", 45455)) + source[33:]
    assert module.prepare_image(source).source_width == 3840


@pytest.mark.parametrize("white_x, accepted", [(31270, True), (40000, False)])
def test_png_chromaticity_must_be_srgb_without_a_profile(white_x, accepted) -> None:
    module = import_module("custom_components.localthings.tv_artwork_image")
    source = encode_image(format="PNG")
    description = struct.pack(">8I", white_x, 32900, 64000, 33000, 30000, 60000, 15000, 6000)
    source = source[:33] + png_chunk(b"cHRM", description) + source[33:]
    if accepted:
        assert module.prepare_image(source).source_width == 3840
    else:
        with pytest.raises(module.ImageValidationError, match="sRGB"):
            module.prepare_image(source)
