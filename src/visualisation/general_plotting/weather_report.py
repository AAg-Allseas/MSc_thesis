from pathlib import Path
from pypdf import PdfReader, PdfWriter


def crop(
    input_file: Path,
    output_file: Path,
    page: int,
    left: float,
    right: float,
    top: float,
    bottom: float,
) -> None:
    writer = PdfWriter()
    reader = PdfReader(input_file)

    page = reader.pages[page - 1]

    page.cropbox.left = left
    page.cropbox.right = right

    page.cropbox.top = top
    page.cropbox.bottom = bottom

    writer.add_page(page)

    writer.write(output_file.with_suffix(".pdf"))


def main() -> None:
    input_file = Path(
        r"C:\Users\AAg\OneDrive - Allseas Engineering BV\Documents\Thesis\aud_weatherreport_2026-03-03-08-01.pdf"
    )

    # Original dimensions A4 in pt
    LEFT = 0
    RIGHT = 595
    TOP = 842
    BOTTOM = 0

    # Spectral forecast
    output_file = Path("report/images/literature_study/2D_spectrum")
    scale_factor = (557) / 595.28
    offset = (451, 56)
    top_left = (510, 440)

    bottom_right = (725, 630)

    width = (bottom_right[0] - top_left[0]) * scale_factor
    height = (bottom_right[1] - top_left[1]) * scale_factor
    orig_tl = [(tl - of) / scale_factor for tl, of in zip(top_left, offset)]

    left = orig_tl[0]
    right = width + left

    top = TOP - orig_tl[1]
    bottom = top - height

    crop(input_file, output_file, 4, left, right, top, bottom)

    # Hs measurements
    output_file = Path("report/images/literature_study/Hs_measurements")

    top = TOP - 100
    bottom = 460
    left = 60
    right = RIGHT - 60

    crop(input_file, output_file, 6, left, right, top, bottom)
