"""Values shared by the ownership ledger and native deletion flow."""

import io
import warnings
from dataclasses import dataclass

from homeassistant.exceptions import HomeAssistantError
from PIL import Image, ImageDraw, ImageFont, ImageOps


@dataclass(frozen=True)
class ArtworkDeletePreview:
    """A checked photo generation; the flow supplies its session and expiry."""

    content_id: str
    title: str
    record_token: str
    thumbnail: bytes
    mime_type: str


@dataclass(frozen=True)
class ArtworkDeleteResult:
    """A fresh catalog outcome, independent of the deletion acknowledgement."""

    content_id: str
    outcome: str


class ArtworkDeleteNotSentError(HomeAssistantError):
    """A safe refusal before any destructive request was sent."""


def delete_contact_sheet(previews: tuple[ArtworkDeletePreview, ...]) -> bytes:
    """Render the checked bytes and their full ordered IDs in an executor."""
    font = ImageFont.load_default(size=16)
    labels: list[list[str]] = []
    for index, preview in enumerate(previews, 1):
        lines = [""]
        for character in f"{index}. {preview.content_id}":
            if font.getlength(lines[-1] + character) > 440:
                lines.append("")
            lines[-1] += character
        labels.append(lines)
    columns = min(2, len(previews))
    cell_height = 288 + 22 * max(len(label) for label in labels)
    sheet = Image.new(
        "RGB", (480 * columns, cell_height * ((len(previews) + columns - 1) // columns)), "#181d25"
    )
    draw = ImageDraw.Draw(sheet)
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        for index, (preview, label) in enumerate(zip(previews, labels, strict=True)):
            with Image.open(io.BytesIO(preview.thumbnail), formats=("JPEG", "PNG")) as header:
                if header.width * header.height > 4_000_000 or getattr(header, "n_frames", 1) != 1:
                    raise ValueError("The checked thumbnail is too large or animated")
                header.verify()
            with Image.open(io.BytesIO(preview.thumbnail), formats=("JPEG", "PNG")) as image:
                image.load()
                thumbnail = ImageOps.contain(image.convert("RGB"), (440, 248))
            x, y = (index % columns) * 480, (index // columns) * cell_height
            sheet.paste(thumbnail, (x + (480 - thumbnail.width) // 2, y + 20))
            for line, text in enumerate(label):
                draw.text((x + 20, y + 278 + 22 * line), text, font=font, fill="white")
    with io.BytesIO() as output:
        sheet.save(output, format="JPEG", quality=90)
        return output.getvalue()
