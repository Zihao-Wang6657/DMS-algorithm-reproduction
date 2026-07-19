from __future__ import annotations

import dataclasses
import json
import logging
import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


LOGGER = logging.getLogger(__name__)
ACCESSIBILITY_FORWARDER_SERVICE = (
    "com.google.androidenv.accessibilityforwarder/"
    "com.google.androidenv.accessibilityforwarder.AccessibilityForwarder"
)
ACCESSIBILITY_FORWARDER_PACKAGE = "com.google.androidenv.accessibilityforwarder"
ACCESSIBILITY_SET_GRPC_ACTION = "accessibility_forwarder.intent.action.SET_GRPC"


def _strict_infrastructure_protocol() -> bool:
    return os.environ.get("DMS_STRICT_INFRA_PROTOCOL", "").strip() == "1"


def configure_windows_adb_stability(*, a11y_method: str | None = None) -> None:
    """Prevent a client timeout from killing the shared Windows adb daemon."""
    if os.name != "nt":
        return

    from android_env.components.adb_controller import AdbController

    if not getattr(AdbController, "_dms_non_destructive_restart", False):

        def _ensure_server(self: Any, timeout: float | None = None) -> None:
            safe_timeout = max(10.0, float(timeout or 0.0))
            command = self.command_prefix(include_device_name=False) + ["start-server"]
            last_error: Exception | None = None
            for attempt in range(3):
                try:
                    subprocess.check_output(
                        command,
                        stderr=subprocess.STDOUT,
                        timeout=safe_timeout,
                        env=self._os_env_vars,
                    )
                    time.sleep(1.0)
                    return
                except (
                    subprocess.CalledProcessError,
                    subprocess.TimeoutExpired,
                ) as exc:
                    last_error = exc
                    if attempt < 2:
                        time.sleep(1.0)
            assert last_error is not None
            raise last_error

        AdbController._restart_server = _ensure_server
        AdbController._dms_non_destructive_restart = True

    from android_env.components.adb_call_parser import AdbCallParser

    if not getattr(AdbCallParser, "_dms_async_a11y_broadcast", False):
        original_send_broadcast = AdbCallParser._send_broadcast

        def _send_a11y_broadcast_async(
            self: Any,
            request: Any,
            timeout: float | None = None,
        ) -> Any:
            broadcast = request.send_broadcast
            action_parts = shlex.split(broadcast.action)
            if not action_parts or not action_parts[0].startswith(
                "accessibility_forwarder.intent.action."
            ):
                return original_send_broadcast(self, request, timeout)
            component_args = (
                ["-n", broadcast.component] if broadcast.component else []
            )
            response, _ = self._execute_command(
                [
                    "shell",
                    "am",
                    "broadcast",
                    "--async",
                    "-a",
                    action_parts[0],
                    *action_parts[1:],
                    *component_args,
                ],
                timeout=min(float(timeout or 10.0), 10.0),
            )
            return response

        AdbCallParser._send_broadcast = _send_a11y_broadcast_async
        AdbCallParser._dms_async_a11y_broadcast = True

    from android_world.env.android_world_controller import AndroidWorldController
    from android_world.env.android_world_controller import A11yMethod

    if (
        str(a11y_method or "").strip().lower() == "uiautomator"
        and not getattr(AndroidWorldController, "_dms_uiautomator_mode", False)
    ):
        original_init = AndroidWorldController.__init__

        def _init_with_uiautomator(
            self: Any,
            env: Any,
            a11y_method: Any = A11yMethod.A11Y_FORWARDER_APP,
            install_a11y_forwarding_app: bool = True,
        ) -> None:
            del a11y_method, install_a11y_forwarding_app
            original_init(
                self,
                env,
                a11y_method=A11yMethod.UIAUTOMATOR,
                install_a11y_forwarding_app=False,
            )

        AndroidWorldController.__init__ = _init_with_uiautomator
        AndroidWorldController._dms_uiautomator_mode = True

    if not getattr(AndroidWorldController, "_dms_refresh_cleanup", False):
        original_refresh = AndroidWorldController.refresh_env

        def _refresh_without_logcat_leak(self: Any) -> None:
            old_env = self.env
            try:
                old_env._coordinator._task_manager.stop()
            except Exception as exc:  # pragma: no cover - environment dependent.
                LOGGER.warning("failed to stop stale Android log stream: %s", exc)
            original_refresh(self)

        AndroidWorldController.refresh_env = _refresh_without_logcat_leak
        AndroidWorldController._dms_refresh_cleanup = True


def _load_adb_utils() -> Any:
    from android_world.env import adb_utils

    return adb_utils


def _controller_adb_target(controller: Any) -> tuple[str, str]:
    """Return the configured adb binary and emulator serial."""
    config = controller.env._coordinator._simulator._config
    adb_path = str(Path(config.adb_controller.adb_path).expanduser())
    console_port = int(config.emulator_launcher.emulator_console_port)
    return adb_path, f"emulator-{console_port}"


def _run_adb_direct(
    controller: Any,
    *args: str,
    timeout_sec: float = 15,
) -> str:
    """Run a best-effort adb client without restarting the global daemon."""
    adb_path, serial = _controller_adb_target(controller)
    completed = subprocess.run(
        [adb_path, "-P", "5037", "-s", serial, *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return completed.stdout


def _direct_uiautomator_dump(controller: Any, timeout_sec: float = 20) -> str:
    _run_adb_direct(
        controller,
        "shell",
        "uiautomator",
        "dump",
        "/sdcard/window_dump.xml",
        timeout_sec=timeout_sec,
    )
    return _run_adb_direct(
        controller,
        "shell",
        "cat",
        "/sdcard/window_dump.xml",
        timeout_sec=timeout_sec,
    )


def _bbox_to_list(bbox: Any) -> list[int] | None:
    if bbox is None:
        return None
    return [
        int(bbox.x_min),
        int(bbox.y_min),
        int(bbox.x_max),
        int(bbox.y_max),
    ]


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


def _wake_android_device(controller: Any) -> None:
    """Best-effort wake/stay-on guard before collecting observations."""
    commands = (
        ("shell", "input", "keyevent", "KEYCODE_WAKEUP"),
        ("shell", "wm", "dismiss-keyguard"),
        ("shell", "svc", "power", "stayon", "true"),
        ("shell", "settings", "put", "system", "screen_off_timeout", "2147483647"),
    )
    for command in commands:
        try:
            _run_adb_direct(controller, *command, timeout_sec=10)
        except Exception as exc:  # pragma: no cover - environment dependent.
            LOGGER.warning("failed Android wake guard command %r: %s", command, exc)


def _disable_airplane_mode(controller: Any) -> None:
    """Best-effort restore of network-dependent emulator state before tasks."""
    commands = (
        ("shell", "settings", "put", "global", "airplane_mode_on", "0"),
        (
            "shell",
            "am",
            "broadcast",
            "--async",
            "-a",
            "android.intent.action.AIRPLANE_MODE",
            "--ez",
            "state",
            "false",
        ),
    )
    for command in commands:
        try:
            _run_adb_direct(controller, *command, timeout_sec=10)
        except Exception as exc:  # pragma: no cover - environment dependent.
            LOGGER.warning("failed to disable airplane mode with %r: %s", command, exc)


def _controller_grpc_port(controller: Any) -> int:
    try:
        return int(
            controller.env._coordinator._simulator._config.emulator_launcher.grpc_port
        )
    except Exception:
        return 8554


def _ensure_accessibility_forwarder(
    controller: Any,
    *,
    force_restart: bool = False,
) -> None:
    """Best-effort recovery for the AndroidWorld accessibility forwarder."""
    grpc_port = _controller_grpc_port(controller)
    commands: list[tuple[tuple[str, ...], int]] = []
    if force_restart:
        commands.append(
            (("shell", "am", "force-stop", ACCESSIBILITY_FORWARDER_PACKAGE), 10)
        )
    commands.extend(
        [
            (
                (
                    "shell",
                    "settings",
                    "put",
                    "secure",
                    "enabled_accessibility_services",
                    ACCESSIBILITY_FORWARDER_SERVICE,
                ),
                10,
            ),
            (
                ("shell", "settings", "put", "secure", "accessibility_enabled", "1"),
                10,
            ),
            (
                (
                    "shell",
                    "am",
                    "broadcast",
                    "--async",
                    "-a",
                    ACCESSIBILITY_SET_GRPC_ACTION,
                    "--ei",
                    "port",
                    str(grpc_port),
                ),
                15,
            ),
        ]
    )
    for command, timeout_sec in commands:
        try:
            _run_adb_direct(controller, *command, timeout_sec=timeout_sec)
        except Exception as exc:  # pragma: no cover - environment dependent.
            LOGGER.warning(
                "failed Android accessibility recovery command %r: %s",
                command,
                exc,
            )


def _uiautomator_fallback_elements(env: Any) -> list[Any]:
    """Fallback to uiautomator when the a11y gRPC tree is empty."""
    controller = getattr(env, "controller", None)
    if controller is None:
        return []
    elements = []
    for attempt in range(2):
        try:
            from android_world.env import representation_utils

            xml = _direct_uiautomator_dump(controller, timeout_sec=20)
            elements = representation_utils.xml_dump_to_ui_elements(xml)
            break
        except Exception as exc:  # pragma: no cover - environment dependent.
            LOGGER.warning("failed to fallback to uiautomator UI dump: %s", exc)
            if attempt == 0:
                _ensure_accessibility_forwarder(controller, force_restart=True)
                _refresh_android_env(env)
                time.sleep(1.0)
    if not elements:
        return []
    visible = [
        element
        for element in elements
        if element.is_visible
        and (
            element.text
            or element.content_description
            or element.is_clickable
            or element.is_editable
        )
    ]
    if visible:
        LOGGER.warning(
            "a11y tree returned no UI elements; using %d uiautomator elements",
            len(visible),
        )
    return visible


def _refresh_android_env(env: Any) -> None:
    controller = getattr(env, "controller", None)
    refresh = getattr(controller, "refresh_env", None)
    if callable(refresh):
        refresh()


def _reset_with_a11y_retries(
    env: Any,
    *,
    go_home: bool = True,
    attempts: int = 3,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        controller = getattr(env, "controller", None)
        if controller is not None:
            _wake_android_device(controller)
            _disable_airplane_mode(controller)
            _ensure_accessibility_forwarder(controller, force_restart=attempt > 0)
        try:
            return env.reset(go_home=go_home)
        except Exception as exc:
            last_error = exc
            if "Could not get a11y tree" not in str(exc) or attempt + 1 >= attempts:
                raise
            LOGGER.warning(
                "reset failed to get a11y tree; refreshing Android env "
                "before retry %d/%d",
                attempt + 1,
                attempts - 1,
            )
            _refresh_android_env(env)
            if controller is not None:
                _ensure_accessibility_forwarder(controller, force_restart=True)
            time.sleep(1.0)
    assert last_error is not None
    raise last_error


def _should_use_uiautomator_fallback(
    elements: list[Any],
    foreground_package: str,
) -> bool:
    if not elements:
        return True
    visible = [element for element in elements if element.is_visible]
    if not visible:
        return True
    packages = {element.package_name for element in visible if element.package_name}
    if packages and packages <= {"com.android.systemui"}:
        return True
    if foreground_package and foreground_package != "com.android.systemui":
        return not any(
            element.package_name == foreground_package for element in visible
        )
    return False


@dataclasses.dataclass(frozen=True)
class ObservationRecord:
    task_id: str
    step_id: int
    timestamp: str
    screenshot_path: str
    ui_elements_path: str
    metadata_path: str
    foreground_activity: str
    package_name: str
    logical_screen_size: tuple[int, int]
    ui_element_count: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class AndroidWorldObservationStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)

    def capture(
        self,
        state: Any,
        env: Any,
        task_id: str,
        step_id: int,
    ) -> ObservationRecord:
        step_dir = self.run_dir / task_id / f"step_{step_id:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = step_dir / "screenshot.png"
        ui_elements_path = step_dir / "ui_elements.json"
        metadata_path = step_dir / "observation.json"

        Image.fromarray(state.pixels).save(screenshot_path)
        foreground_activity = env.foreground_activity_name
        package_name = foreground_activity.split("/", maxsplit=1)[0]
        source_elements = list(state.ui_elements)
        if _should_use_uiautomator_fallback(source_elements, package_name):
            source_elements = _uiautomator_fallback_elements(env)
            if not source_elements:
                controller = getattr(env, "controller", None)
                if controller is not None:
                    _ensure_accessibility_forwarder(controller, force_restart=True)
                    _refresh_android_env(env)
                    _wake_android_device(controller)
                try:
                    state = env.get_state(wait_to_stabilize=True)
                    Image.fromarray(state.pixels).save(screenshot_path)
                    foreground_activity = env.foreground_activity_name
                    package_name = foreground_activity.split("/", maxsplit=1)[0]
                    source_elements = list(state.ui_elements)
                    if _should_use_uiautomator_fallback(
                        source_elements,
                        package_name,
                    ):
                        source_elements = _uiautomator_fallback_elements(env)
                except Exception as exc:  # pragma: no cover - environment dependent.
                    LOGGER.warning("failed to recover empty UI observation: %s", exc)
                    if _strict_infrastructure_protocol():
                        raise
        if not source_elements and _strict_infrastructure_protocol():
            raise RuntimeError(
                "Strict infrastructure protocol could not obtain a non-empty UI observation."
            )
        ui_elements = [
            serialize_ui_element(element, index)
            for index, element in enumerate(source_elements)
        ]
        ui_elements_path.write_text(
            json.dumps(ui_elements, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        record = ObservationRecord(
            task_id=task_id,
            step_id=step_id,
            timestamp=datetime.now().astimezone().isoformat(),
            screenshot_path=str(screenshot_path.resolve()),
            ui_elements_path=str(ui_elements_path.resolve()),
            metadata_path=str(metadata_path.resolve()),
            foreground_activity=foreground_activity,
            package_name=package_name,
            logical_screen_size=tuple(env.logical_screen_size),
            ui_element_count=len(ui_elements),
        )
        metadata_path.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record


def get_state_with_a11y_retries(
    env: Any,
    *,
    wait_to_stabilize: bool,
    attempts: int = 3,
) -> Any:
    last_error: Exception | None = None
    controller = getattr(env, "controller", None)
    for attempt in range(attempts):
        try:
            return env.get_state(wait_to_stabilize=wait_to_stabilize)
        except Exception as exc:
            last_error = exc
            if "Could not get a11y tree" not in str(exc) or attempt + 1 >= attempts:
                raise
            LOGGER.warning(
                "state capture failed to get a11y tree; refreshing Android env "
                "before retry %d/%d",
                attempt + 1,
                attempts - 1,
            )
            _refresh_android_env(env)
            if controller is not None:
                _wake_android_device(controller)
                _ensure_accessibility_forwarder(
                    controller,
                    force_restart=attempt > 0,
                )
            time.sleep(1.0)
    assert last_error is not None
    raise last_error


def reset_task_environment(env: Any, *, go_home: bool = True) -> Any:
    """Reset a task without inheriting stale Android automation state."""
    controller = getattr(env, "controller", None)
    state = _reset_with_a11y_retries(env, go_home=go_home)
    if controller is None:
        return state
    _wake_android_device(controller)
    _disable_airplane_mode(controller)
    try:
        adb_utils = _load_adb_utils()
    except Exception as exc:  # pragma: no cover - environment dependent.
        LOGGER.warning("failed to load Android adb utilities: %s", exc)
        return state

    try:
        if hasattr(env, "hide_automation_ui"):
            env.hide_automation_ui()
        else:
            adb_utils.issue_generic_request(
                "shell settings put system pointer_location 0",
                controller,
                timeout_sec=2,
            )
    except Exception as exc:  # pragma: no cover - environment dependent.
        LOGGER.warning("failed to hide Android automation UI: %s", exc)

    try:
        adb_utils.press_back_button(controller, timeout_sec=2)
        if go_home:
            adb_utils.press_home_button(controller, timeout_sec=2)
    except Exception as exc:  # pragma: no cover - environment dependent.
        LOGGER.warning("failed to sanitize Android task start state: %s", exc)

    for attempt in range(3):
        try:
            return get_state_with_a11y_retries(
                env,
                wait_to_stabilize=True,
                attempts=3 - attempt,
            )
        except Exception as exc:  # pragma: no cover - environment dependent.
            if "Could not get a11y tree" not in str(exc) or attempt == 2:
                LOGGER.warning("failed to capture sanitized Android state: %s", exc)
                if _strict_infrastructure_protocol():
                    raise
                return state
            LOGGER.warning(
                "sanitized state capture failed to get a11y tree; "
                "refreshing Android env before retry %d/2",
                attempt + 1,
            )
            _refresh_android_env(env)
            _wake_android_device(controller)
            _disable_airplane_mode(controller)
            _ensure_accessibility_forwarder(controller, force_restart=attempt > 0)
            time.sleep(1.0)
    return state
