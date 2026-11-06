"""Extract labeled diagnostic frames from a recorded simulator rollout."""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cap = cv2.VideoCapture(str(args.video))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    assert frames
    selected = sorted(set([0, min(15, len(frames)-1), min(30, len(frames)-1),
                           min(45, len(frames)-1), min(75, len(frames)-1), len(frames)-1]))
    sheet = Image.new("RGB", (3*256, 2*284), "white")
    draw = ImageDraw.Draw(sheet)
    for i, index in enumerate(selected):
        x, y = (i % 3)*256, (i // 3)*284
        sheet.paste(Image.fromarray(frames[index]).resize((256, 256)), (x, y))
        draw.text((x+8, y+262), f"Video frame {index} / {len(frames)-1}", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)


if __name__ == "__main__":
    main()
