from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .actions import ParsedAction


@dataclass(frozen=True)
class Observation:
    screenshot: str
    xml_path: str
    index: int


class DeviceBackend(Protocol):
    paper_eligible: bool

    def reset(self, task: dict[str, Any], task_dir: Path) -> Observation: ...

    def act(self, action: ParsedAction) -> None: ...

    def observe(self, task_dir: Path, index: int) -> Observation: ...

    def close_task(self, task: dict[str, Any]) -> None: ...


class ReplayDevice:
    paper_eligible = False

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.current: dict[str, Any] = {}

    def reset(self, task: dict[str, Any], task_dir: Path) -> Observation:
        del task_dir
        self.current = task
        return self.observe(Path("."), 0)

    def act(self, action: ParsedAction) -> None:
        if not action.valid:
            raise ValueError(f"Replay device received invalid action: {action.error}")

    def observe(self, task_dir: Path, index: int) -> Observation:
        del task_dir
        current = self.current.get("input") if isinstance(self.current.get("input"), dict) else {}
        observations = current.get("replay_observations")
        observations = observations if isinstance(observations, list) else []
        if observations:
            row = observations[min(index, len(observations) - 1)]
            return Observation(str(row.get("screenshot") or ""), str(row.get("xml") or ""), index)
        return Observation(
            str(current.get("initial_screenshot") or ""),
            str(current.get("initial_xml") or ""),
            index,
        )

    def close_task(self, task: dict[str, Any]) -> None:
        del task


class AdbDevice:
    paper_eligible = True

    def __init__(self, config: dict[str, Any]):
        self.config = config
        adb_path = str(config.get("adb_path") or "adb")
        resolved = shutil.which(adb_path) or (adb_path if Path(adb_path).is_file() else "")
        if not resolved:
            raise FileNotFoundError(f"ADB executable was not found: {adb_path}")
        self.adb_path = resolved
        self.serial = str(config.get("serial") or "")
        self.observe_delay = float(config.get("observe_delay_seconds") or 1.0)
        self.action_delay = float(config.get("action_delay_seconds") or 1.0)
        self.wait_delay = float(config.get("wait_delay_seconds") or 2.0)
        self.long_click_duration = int(config.get("long_click_duration_ms") or 500)
        self.scroll_distance = int(config.get("scroll_distance_px") or 500)
        self.scroll_duration = int(config.get("scroll_duration_ms") or 300)
        self.hierarchy_backend = str(config.get("hierarchy_backend") or "adb_uiautomator_dump")
        self._uiautomator2_device: Any = None
        self._run(["get-state"])

    def reset(self, task: dict[str, Any], task_dir: Path) -> Observation:
        app = str((task.get("input") or {}).get("app") or "")
        hook = self._hook(app)
        self._commands(hook.get("before_task") or [])
        time.sleep(self.observe_delay)
        return self.observe(task_dir, 0)

    def act(self, action: ParsedAction) -> None:
        if not action.valid:
            raise ValueError(f"ADB device received invalid action: {action.error}")
        if action.action_type == "click":
            self._run(["shell", "input", "tap", str(action.x), str(action.y)])
        elif action.action_type == "long_click":
            self._run(
                [
                    "shell",
                    "input",
                    "swipe",
                    str(action.x),
                    str(action.y),
                    str(action.x),
                    str(action.y),
                    str(self.long_click_duration),
                ]
            )
        elif action.action_type == "scroll":
            x, y = int(action.x or 540), int(action.y or 1200)
            delta = self.scroll_distance
            endpoints = {
                "down": (x, y, x, y - delta),
                "up": (x, y, x, y + delta),
                "right": (x, y, x - delta, y),
                "left": (x, y, x + delta, y),
            }
            x1, y1, x2, y2 = endpoints[action.direction]
            self._run(
                ["shell", "input", "swipe", str(x1), str(y1), str(x2), str(y2), str(self.scroll_duration)]
            )
        elif action.action_type == "type":
            self._type_text(action.text)
        elif action.action_type == "press_back":
            self._run(["shell", "input", "keyevent", "4"])
        elif action.action_type == "press_home":
            self._run(["shell", "input", "keyevent", "3"])
        elif action.action_type == "press_recent":
            self._run(["shell", "input", "keyevent", "187"])
        elif action.action_type == "wait":
            time.sleep(self.wait_delay)
        elif action.action_type != "finished":
            raise ValueError(f"Unsupported ADB action: {action.action_type}")
        if action.action_type != "wait":
            time.sleep(self.action_delay)

    def observe(self, task_dir: Path, index: int) -> Observation:
        task_dir.mkdir(parents=True, exist_ok=True)
        screenshot = task_dir / f"observation_{index:04d}.png"
        xml_path = task_dir / f"observation_{index:04d}.xml"
        screenshot_data = self._run(["exec-out", "screencap", "-p"], binary=True)
        if not screenshot_data:
            raise RuntimeError("ADB returned an empty screenshot")
        screenshot.write_bytes(screenshot_data)
        if self.hierarchy_backend == "uiautomator2":
            xml = self._u2().dump_hierarchy()
            if not xml:
                raise RuntimeError("uiautomator2 returned an empty hierarchy")
            xml_path.write_text(xml, encoding="utf-8")
        elif self.hierarchy_backend == "adb_uiautomator_dump":
            self._run(["shell", "uiautomator", "dump", "/sdcard/window_dump.xml"])
            xml = self._run(["exec-out", "cat", "/sdcard/window_dump.xml"], binary=True)
            if not xml:
                raise RuntimeError("ADB returned an empty UI hierarchy")
            xml_path.write_bytes(xml)
        else:
            raise ValueError(f"Unsupported hierarchy_backend: {self.hierarchy_backend}")
        return Observation(str(screenshot), str(xml_path), index)

    def close_task(self, task: dict[str, Any]) -> None:
        app = str((task.get("input") or {}).get("app") or "")
        self._commands(self._hook(app).get("after_task") or [])

    def _hook(self, app: str) -> dict[str, Any]:
        hooks = self.config.get("app_hooks")
        hooks = hooks if isinstance(hooks, dict) else {}
        hook = hooks.get(app)
        if not isinstance(hook, dict):
            if self.config.get("require_app_hook", True):
                raise ValueError(f"No deterministic ADB reset hook is configured for app: {app}")
            return {}
        return hook

    def _commands(self, commands: list[Any]) -> None:
        for command in commands:
            if not isinstance(command, list) or not command or any(not isinstance(item, str) for item in command):
                raise ValueError(f"ADB hook commands must be non-empty JSON argv arrays: {command!r}")
            self._run(command)

    def _type_text(self, text: str) -> None:
        mode = str(self.config.get("text_input_mode") or "adb_keyboard")
        if mode == "uiautomator2":
            self._u2().send_keys(text)
            return
        if mode == "adb_keyboard":
            self._run(["shell", "am", "broadcast", "-a", "ADB_INPUT_TEXT", "--es", "msg", text])
            return
        if mode == "input_text":
            if not text.isascii():
                raise ValueError("ADB input text mode cannot reliably type non-ASCII text; use adb_keyboard")
            self._run(["shell", "input", "text", text.replace(" ", "%s")])
            return
        raise ValueError(f"Unsupported text_input_mode: {mode}")

    def _u2(self) -> Any:
        if self._uiautomator2_device is not None:
            return self._uiautomator2_device
        try:
            import uiautomator2 as u2
        except ImportError as error:
            raise RuntimeError("uiautomator2 is required by the selected device configuration") from error
        self._uiautomator2_device = u2.connect(self.serial) if self.serial else u2.connect()
        return self._uiautomator2_device

    def _run(self, arguments: list[str], *, binary: bool = False) -> Any:
        command = [self.adb_path]
        if self.serial:
            command.extend(["-s", self.serial])
        command.extend(arguments)
        completed = subprocess.run(command, check=True, capture_output=True, text=not binary)
        return completed.stdout


def create_device_backend(config: dict[str, Any]) -> DeviceBackend:
    backend = str(config.get("backend") or "")
    if backend == "replay":
        return ReplayDevice(config)
    if backend == "adb":
        return AdbDevice(config)
    raise ValueError(f"Unsupported device backend: {backend}")
