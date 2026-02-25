#!/usr/bin/env python3
import os
import click
import img2pdf

from multitasking.CONSTANTS import ROIS as ROI_ORDER

@click.command()
@click.option(
    "--path-to-folder",
    "-p",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Folder containing image files."
)
@click.option(
    "--extension",
    "-e",
    default="png",
    show_default=True,
    help="File extension to search for (e.g., png, jpg)."
)
@click.option(
    "--pattern",
    "-s",
    default="",
    show_default=True,
    help="Substring that must appear in the filename (case-insensitive)."
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Directory where the output PDF will be saved."
)
@click.option(
    "--output-filename",
    "-f",
    default="output.pdf",
    show_default=True,
    help="Name of the resulting PDF file."
)
@click.option(
    "--sort-by-roi",
    "-r",
    default=False,
    show_default=True,
    help="Sort the images by ROI."
)
def collate_images(path_to_folder, extension, pattern, output_dir, output_filename, sort_by_roi):
    """Scrape all images of a given extension (and optional name pattern) and save as one PDF."""

    # Normalize
    ext = extension.lower().lstrip(".")
    patt = pattern.lower()
    
    if output_filename[-4:] != ".pdf":
        output_filename += ".pdf"

    # Collect files with extension + optional substring
    files = sorted([
        os.path.join(path_to_folder, f)
        for f in os.listdir(path_to_folder)
        if f.lower().endswith(f".{ext}") and (patt in f.lower())
    ])

    if sort_by_roi:
        try:
            files = sorted(files, key=lambda x: ROI_ORDER.index(x.split("/")[-1].split("_")[-1].split(".")[0]))
        except ValueError:
            files = sorted(files, key=lambda x: ROI_ORDER.index(x.split("/")[-1].split("_")[0]))

    if not files:
        click.echo(
            f"No .{ext} files matching pattern '{pattern}' found in {path_to_folder}"
        )
        return

    output_path = os.path.join(output_dir, output_filename)


    # Convert to PDF
    with open(output_path, "wb") as f:
        f.write(img2pdf.convert(files))

    click.echo(f"PDF created: {output_path}")


if __name__ == "__main__":
    collate_images()
