import glob
import math
import os

from PIL import Image, ImageDraw


SCRIPT_VERSION = "v20260624a"
INPUT_GLOB = "results/chirp_rate_scale_test/evt*_scale0994938967.png"
OUTPUT_BASE = f"results/chirp_rate_scale_test/chirp_rate_scale_test_gallery_{SCRIPT_VERSION}"


def main():
    paths = sorted(glob.glob(INPUT_GLOB))
    if not paths:
        raise RuntimeError(f"No diagnostic PNGs found with pattern {INPUT_GLOB!r}")

    thumb_w = 850
    thumb_h = 640
    title_h = 36
    thumbs = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h + title_h), "white")
        canvas.paste(image, ((thumb_w - image.width) // 2, title_h))
        label = os.path.basename(path).replace("_scale0994938967.png", "")
        ImageDraw.Draw(canvas).text((12, 10), label, fill=(0, 0, 0))
        thumbs.append(canvas)

    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    gap = 20
    out_w = cols * thumb_w + (cols + 1) * gap
    out_h = rows * (thumb_h + title_h) + (rows + 1) * gap
    output = Image.new("RGB", (out_w, out_h), "white")
    for idx, thumb in enumerate(thumbs):
        x = gap + (idx % cols) * (thumb_w + gap)
        y = gap + (idx // cols) * (thumb_h + title_h + gap)
        output.paste(thumb, (x, y))

    os.makedirs(os.path.dirname(OUTPUT_BASE), exist_ok=True)
    output.save(f"{OUTPUT_BASE}.png", dpi=(180, 180))
    output.save(f"{OUTPUT_BASE}.pdf", resolution=180)
    print(f"{OUTPUT_BASE}.png")
    print(f"{OUTPUT_BASE}.pdf")


if __name__ == "__main__":
    main()
