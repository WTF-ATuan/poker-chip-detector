"""
簡易互動式標註工具，可繪製 YOLO 格式的方框（支援 undo 與多色框）。

用法:
  1) source .venv311/bin/activate
  2) python label_tool.py          # 會遍歷 data/*.jpg
     - 滑鼠：點擊拖曳畫方框
     - 按鍵：s=儲存並下一張, c=清除全部, n=跳過, q=離開, z=undo(上一步)
  標註存於 data/labels/<stem>.txt，YOLO 格式：`0 x y w h`
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import List, Tuple

import cv2

DATA_DIR = Path("data")
LABELS_DIR = DATA_DIR / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# 預設一組顏色循環（BGR）
BOX_COLORS = [
    (0, 255, 0),      # 綠
    (0, 128, 255),    # 橘
    (255, 0, 0),      # 藍
    (255, 0, 255),    # 紫
    (0, 255, 255),    # 黃
    (255, 255, 0),    # 青
    (128, 0, 255),    # 粉紫
    (0, 0, 255),      # 紅
    (255, 128, 0),    # 深橘
    (128, 255, 0),    # 淺綠
]

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
        self.need_redraw = True  # 標記是否需要重繪
        self.last_temp_box: Tuple[int, int, int, int] | None = None  # 記錄上次暫時框

    def redraw(self, temp_box: Tuple[int, int, int, int] | None = None):
        # 不管是不是拖曳，每次都直接在 self.view 上畫，不複製底圖，允許畫面混亂
        if temp_box is None:
            # 畫所有永久框
            for idx, (x1, y1, x2, y2) in enumerate(self.boxes):
                color = BOX_COLORS[idx % len(BOX_COLORS)]
                cv2.rectangle(self.view, (x1, y1), (x2, y2), color, 2)
            self.need_redraw = False
        else:
            # 拖曳時直接在 view 上畫暫時框，不還原、不複製
            color = (0, 255, 255)
            cv2.rectangle(self.view, (temp_box[0], temp_box[1]), (temp_box[2], temp_box[3]), color, 2)
            self.last_temp_box = temp_box
    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_pt = (x, y)
            self.last_temp_box = None  # 開始畫新框，重置暫時框狀態
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing and self.start_pt is not None:
            self.redraw((self.start_pt[0], self.start_pt[1], x, y))
        elif event == cv2.EVENT_LBUTTONUP and self.drawing and self.start_pt is not None:
            x1, y1 = self.start_pt
            self.boxes.append((x1, y1, x, y))
            self.drawing = False
            self.start_pt = None
            self.need_redraw = True
            self.last_temp_box = None  # 結束畫框，重置暫時框狀態

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
            # 只有在 boxes 有變動（如undo/新增/清除）時才重繪底圖
            if session.need_redraw:
                session.redraw()
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
                session.need_redraw = True
                session.last_temp_box = None
            if key == ord('z'):
                # undo: 移除最後一個方框
                if session.boxes:
                    session.boxes.pop()
                    session.need_redraw = True
                    session.last_temp_box = None
            if key == ord('s'):
                out = session.save_labels()
                print(f"Saved: {out}")
                cv2.destroyWindow(win)
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
