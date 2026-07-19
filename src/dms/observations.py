from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from dms.io_utils import now_iso, write_json


def _bbox_to_list(bbox: Any) -> list[int] | None:
    if bbox is None:
        return None
    return [int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)]


def serialize_ui_element(element: Any, index: int) -> dict[str, Any]:
    return {
        "index": index,
        "text": element.text,
        "content_description": element.content_description,
        "class_name": element.class_name,
        "resource_name": element.resource_name,
        "package_name": element.package_name,
        "bounds": _bbox_to_list(element.bbox_pixels),
        "is_clickable": element.is_clickable,
        "is_editable": element.is_editable,
        "is_enabled": element.is_enabled,
        "is_scrollable": element.is_scrollable,
        "is_visible": element.is_visible,
    }


def compact_ui_elements(elements: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    compact = []
    for element in elements:
        if not element["is_visible"]:
            continue
        if not (
            element["text"]
            or element["content_description"]
            or element["is_clickable"]
            or element["is_editable"]
            or element["is_scrollable"]
        ):
            continue
        compact.append(
            {
                "index": element["index"],
                "text": element["text"],
                "content_description": element["content_description"],
                "class_name": element["class_name"],
                "bounds": element["bounds"],
                "clickable": element["is_clickable"],
                "editable": element["is_editable"],
                "scrollable": element["is_scrollable"],
            }
        )
        if len(compact) >= limit:
            break
    return compact


@dataclass(frozen=True)
class ObservationRecord:
    task_id: str
    step_id: int
    timestamp: str
    screenshot_path: str
    ui_elements_path: str
    metadata_path: str
    foreground_activity: str
    logical_screen_size: tuple[int, int]
    ui_element_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ObservationStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def capture(
        self,
        state: Any,
        env: Any,
        task_id: str,
        step_id: int,
        foreground_activity: str,
    ) -> ObservationRecord:
        step_dir = self.run_dir / "observations" / task_id / f"step_{step_id:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = step_dir / "screenshot.png"
        ui_elements_path = step_dir / "ui_elements.json"
        metadata_path = step_dir / "observation.json"

        Image.fromarray(state.pixels).save(screenshot_path)
        ui_elements = [
            serialize_ui_element(element, index)
            for index, element in enumerate(state.ui_elements)
        ]
        write_json(ui_elements_path, ui_elements)
        record = ObservationRecord(
            task_id=task_id,
            step_id=step_id,
            timestamp=now_iso(),
            screenshot_path=str(screenshot_path.resolve()),
            ui_elements_path=str(ui_elements_path.resolve()),
            metadata_path=str(metadata_path.resolve()),
            foreground_activity=foreground_activity,
            logical_screen_size=tuple(env.logical_screen_size),
            ui_element_count=len(ui_elements),
        )
        metadata = record.to_dict()
        metadata["compact_ui_elements"] = compact_ui_elements(ui_elements)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return record
