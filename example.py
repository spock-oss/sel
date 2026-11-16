"""Send a text, image, and mixed request to a server started with serve.py."""

import argparse
import base64
import io
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw


def example_image():
    image = Image.new("RGB", (256, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 80, 111, 175), fill="#c53132")
    draw.ellipse((144, 80, 223, 175), fill="#2167bb")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def select(url, **fields):
    request = Request(
        url.rstrip("/") + "/select",
        data=json.dumps(fields).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=180) as response:
            return json.load(response)
    except HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode()}") from exc


def show(name, result, text=""):
    units = result["units"]
    selected = sum(unit["selected"] for unit in units)
    print(
        f"\n{name}: {result['input_tokens']} input tokens, {selected}/{len(units)} selected"
    )
    for kind in ("text", "image"):
        matches = [unit for unit in units if unit["kind"] == kind]
        if not matches:
            continue
        print(f"  Top {kind} scores")
        for unit in sorted(matches, key=lambda item: item["score"], reverse=True)[:3]:
            location = (
                repr(text[unit["start"] : unit["end"]])
                if kind == "text"
                else f"image box {[round(value) for value in unit['xyxy']]}"
            )
            print(f"    {unit['score']:.3f}  {location}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    text = "The notice period is 30 days. This agreement is governed by the laws of Oregon."
    show(
        "Text",
        select(args.url, question="Which state governs the agreement?", text=text),
        text,
    )

    image = example_image()
    show(
        "Image",
        select(args.url, question="Select the red rectangle.", image_base64=image),
    )

    legend = "The red rectangle marks the loading area. The blue circle marks the waiting area."
    show(
        "Mixed",
        select(
            args.url,
            question="Select the loading area in the image and the sentence that identifies it.",
            text=legend,
            image_base64=image,
        ),
        legend,
    )


if __name__ == "__main__":
    main()
