"""
Simple interactive labeling tool to draw YOLO boxes for chip stacks.

Usage:
  1) source .venv311/bin/activate
  2) python label_tool.py          # iterates data/*.jpg
     - Mouse: click-drag to draw a rectangle
     - Keys:  s=save&next, c=clear, n=next (skip), q=quit
  Labels saved to data/labels/<stem>.txt, YOLO format: `0 x y w h`
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import cv2


DATA_DIR = Path("data")
LABELS_DIR = DATA_DIR / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[float, float, float, float]:
    xmin, ymin = min(x1, x2), min(y1, y2)
    xmax, ymax = max(x1, x2), max(y1, y2)
    bw = max(1, xmax - xmin)
    bh = max(1, ymax - ymin)
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


class LabelSession:
    def __init__(self, image_path: Path) -> None:
        self.image_path = image_path
        self.img = cv2.imread(str(image_path))
        if self.img is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        self.h, self.w = self.img.shape[:2]
        self.view = self.img.copy()
        self.boxes: List[Tuple[int, int, int, int]] = []
        self.drawing = False
        self.start_pt: Tuple[int, int] | None = None

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and self.start_pt is not None:
            self.view = self.img.copy()
            for (x1, y1, x2, y2) in self.boxes:
                cv2.rectangle(self.view, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(self.view, self.start_pt, (x, y), (0, 255, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing and self.start_pt is not None:
            x1, y1 = self.start_pt
            self.boxes.append((x1, y1, x, y))
            self.drawing = False
            self.start_pt = None
            self.view = self.img.copy()
            for (bx1, by1, bx2, by2) in self.boxes:
                cv2.rectangle(self.view, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    def save_labels(self) -> Path:
        stem = self.image_path.stem
        out_path = LABELS_DIR / f"{stem}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for (x1, y1, x2, y2) in self.boxes:
                x, y, w, h = to_yolo(x1, y1, x2, y2, self.w, self.h)
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        return out_path


def list_images() -> List[Path]:
    imgs = [Path(p) for p in glob.glob(str(DATA_DIR / "*.jpg"))]
    return sorted(imgs)


def main() -> int:
    images = list_images()
    for img_path in images:
        session = LabelSession(img_path)
        win = f"label: {img_path.name}"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, session.on_mouse)
        while True:
            cv2.imshow(win, session.view)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(win)
                return 0
            if key == ord('n'):
                cv2.destroyWindow(win)
                break
            if key == ord('c'):
                session.boxes.clear()
                session.view = session.img.copy()
            if key == ord('s'):
                out = session.save_labels()
                print(f"Saved: {out}")
                cv2.destroyWindow(win)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


