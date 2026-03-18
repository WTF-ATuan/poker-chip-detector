from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2


COLOR_KEYS = {
    ord("1"): "red",
    ord("2"): "pink",
    ord("3"): "green",
    ord("4"): "black",
}

BOX_COLORS_BGR = {
    "red": (30, 80, 230),
    "pink": (180, 130, 240),
    "green": (80, 190, 90),
    "black": (40, 40, 40),
}


@dataclass
class LabeledBox:
    color: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class Annotator:
    def __init__(self, image_path: Path, gt_path: Path) -> None:
        self.image_path = image_path
        self.gt_path = gt_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise RuntimeError(f"Failed to open image: {image_path}")

        self.h, self.w = self.image.shape[:2]
        self.current_color = "red"
        self.boxes: List[LabeledBox] = self.load_existing_boxes()

        self.drawing = False
        self.start_pt: Tuple[int, int] | None = None
        self.temp_end: Tuple[int, int] | None = None

    def load_existing_boxes(self) -> List[LabeledBox]:
        if not self.gt_path.exists():
            return []
        try:
            data = json.loads(self.gt_path.read_text(encoding="utf-8"))
            out: List[LabeledBox] = []
            for ann in data.get("annotations", []):
                color = str(ann.get("color", "")).lower()
                bbox = ann.get("bbox", [])
                if color not in BOX_COLORS_BGR or len(bbox) != 4:
                    continue
                out.append(
                    LabeledBox(
                        color=color,
                        bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                    )
                )
            return out
        except Exception:
            return []

    def save(self) -> None:
        payload = {
            "image": str(self.image_path).replace("\\", "/"),
            "image_width": int(self.w),
            "image_height": int(self.h),
            "colors": ["red", "pink", "green", "black"],
            "annotations": [
                {"color": box.color, "bbox": [box.bbox[0], box.bbox[1], box.bbox[2], box.bbox[3]]}
                for box in self.boxes
            ],
        }
        self.gt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved GT: {self.gt_path} ({len(self.boxes)} boxes)")

    def draw(self) -> None:
        canvas = self.image.copy()
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box.bbox
            color = BOX_COLORS_BGR[box.color]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                canvas,
                f"{i+1}:{box.color}",
                (max(0, x1), max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

        if self.drawing and self.start_pt and self.temp_end:
            color = BOX_COLORS_BGR[self.current_color]
            cv2.rectangle(canvas, self.start_pt, self.temp_end, color, 2)

        legend = (
            f"[1]red [2]pink [3]green [4]black  current={self.current_color}  "
            "[u]undo [c]clear [s]save [q]quit"
        )
        cv2.rectangle(canvas, (0, 0), (self.w, 26), (0, 0, 0), -1)
        cv2.putText(canvas, legend, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.imshow("Regression Label Tool", canvas)

    def on_mouse(self, event, x, y, flags, param) -> None:
        x = max(0, min(self.w - 1, int(x)))
        y = max(0, min(self.h - 1, int(y)))
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.temp_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing and self.start_pt:
            self.drawing = False
            x1, y1 = self.start_pt
            x2, y2 = x, y
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                self.boxes.append(LabeledBox(color=self.current_color, bbox=(x1, y1, x2, y2)))
            self.start_pt = None
            self.temp_end = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label one regression image with color boxes.")
    parser.add_argument("--image", default="data/regression/image_1023.png", help="Image path.")
    parser.add_argument("--gt", default="data/regression/image_1023_gt.json", help="GT json output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    gt_path = Path(args.gt).resolve()
    gt_path.parent.mkdir(parents=True, exist_ok=True)

    tool = Annotator(image_path=image_path, gt_path=gt_path)
    cv2.namedWindow("Regression Label Tool", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Regression Label Tool", tool.on_mouse)

    while True:
        tool.draw()
        key = cv2.waitKey(20) & 0xFF
        if key in COLOR_KEYS:
            tool.current_color = COLOR_KEYS[key]
        elif key == ord("u"):
            if tool.boxes:
                tool.boxes.pop()
        elif key == ord("c"):
            tool.boxes.clear()
        elif key == ord("s"):
            tool.save()
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
