from __future__ import annotations

import dataclasses
import json
import logging
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


LOGGER = logging.getLogger(__name__)
ACCESSIBILITY_FLAGS_RECEIVER = (
    "com.google.androidenv.accessibilityforwarder/"
    "com.google.androidenv.accessibilityforwarder.FlagsBroadcastReceiver"
)
ACCESSIBILITY_SET_GRPC_ACTION = "accessibility_forwarder.intent.action.SET_GRPC"
ACCESSIBILITY_DISABLE_TREE_ACTION = (
    "accessibility_forwarder.intent.action.DISABLE_ACCESSIBILITY_TREE_LOGS"
)
PERMISSION_CONTROLLER_PACKAGES = frozenset(
    {
        "com.android.permissioncontroller",
        "com.google.android.permissioncontroller",
    }
)
A11Y_TRANSITION_ATTEMPTS = 6
A11Y_TRANSITION_RETRY_SECONDS = 0.75


class A11yInfrastructureError(RuntimeError):
    """The forwarder did not provide a trustworthy accessibility observation."""


def _strict_a11y_protocol() -> bool:
    return os.environ.get("DMS_STRICT_A11Y_PROTOCOL", "").strip() == "1"


def _load_adb_utils() -> Any:
    from android_world.env import adb_utils

    return adb_utils


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
    try:
        adb_utils = _load_adb_utils()
    except Exception as exc:  # pragma: no cover - environment dependent.
        LOGGER.warning("failed to load Android adb utilities: %s", exc)
        return

    commands = (
        "shell input keyevent KEYCODE_WAKEUP",
        "shell wm dismiss-keyguard",
        "shell svc power stayon true",
        "shell settings put system screen_off_timeout 2147483647",
    )
    for command in commands:
        try:
            adb_utils.issue_generic_request(command, controller, timeout_sec=3)
        except Exception as exc:  # pragma: no cover - environment dependent.
            LOGGER.warning("failed Android wake guard command %r: %s", command, exc)


def _disable_airplane_mode(controller: Any) -> None:
    """Best-effort restore of network-dependent emulator state before tasks."""
    try:
        adb_utils = _load_adb_utils()
    except Exception as exc:  # pragma: no cover - environment dependent.
        LOGGER.warning("failed to load Android adb utilities: %s", exc)
        return

    commands = (
        "shell settings put global airplane_mode_on 0",
        "shell am broadcast -a android.intent.action.AIRPLANE_MODE --ez state false",
    )
    for command in commands:
        try:
            adb_utils.issue_generic_request(command, controller, timeout_sec=3)
        except Exception as exc:  # pragma: no cover - environment dependent.
            LOGGER.warning("failed to disable airplane mode with %r: %s", command, exc)


def _controller_a11y_server_port(controller: Any) -> int | None:
    """Return AndroidEnv's dynamic a11y port, never the emulator control port."""
    wrapped_env = getattr(controller, "env", None)
    visited: set[int] = set()
    while wrapped_env is not None and id(wrapped_env) not in visited:
        visited.add(id(wrapped_env))
        get_port = getattr(wrapped_env, "get_port", None)
        if callable(get_port):
            try:
                port = int(get_port())
            except (TypeError, ValueError):
                port = 0
            if port > 0:
                return port
        wrapped_env = getattr(wrapped_env, "_env", None)
    return None


def _host_port_is_listening(port: int, *, timeout_seconds: float = 1.0) -> bool:
    try:
        with socket.create_connection(
            ("127.0.0.1", port),
            timeout=timeout_seconds,
        ):
            return True
    except OSError:
        return False


def _disable_accessibility_forwarding(controller: Any) -> None:
    """Stop device-side sends before AndroidEnv closes its host gRPC server."""
    adb_utils = _load_adb_utils()
    commands = (
        (
            f"shell am broadcast -a {ACCESSIBILITY_DISABLE_TREE_ACTION} "
            f"-n {ACCESSIBILITY_FLAGS_RECEIVER}"
        ),
        (
            f"shell am broadcast -a {ACCESSIBILITY_SET_GRPC_ACTION} "
            f'--es host "10.0.2.2" --ei port 0 -n {ACCESSIBILITY_FLAGS_RECEIVER}'
        ),
    )
    for command in commands:
        adb_utils.issue_generic_request(command, controller, timeout_sec=5)


def _reset_with_a11y_retries(
    env: Any,
    *,
    go_home: bool = True,
) -> Any:
    controller = getattr(env, "controller", None)
    if controller is not None:
        _wake_android_device(controller)
        _disable_airplane_mode(controller)
    try:
        return env.reset(go_home=go_home)
    except Exception as exc:
        if "Could not get a11y tree" in str(exc):
            raise A11yInfrastructureError(
                "Accessibility tree was unavailable during task reset; "
                "the environment was not refreshed or replaced."
            ) from exc
        raise


def _is_untrustworthy_a11y_observation(
    elements: list[Any],
    foreground_package: str,
) -> bool:
    if not elements:
        return True
    visible = [element for element in elements if element.is_visible]
    if not visible:
        return True
    packages = {
        str(element.package_name)
        for element in visible
        if element.package_name
    }
    if not packages or foreground_package in packages:
        return False
    # Runtime permission dialogs legitimately belong to PermissionController
    # while the target app remains the foreground activity.
    if packages & PERMISSION_CONTROLLER_PACKAGES:
        return False
    # During app startup the forwarder can briefly return the previous app's
    # tree, or only the SystemUI status/navigation bars, after the foreground
    # activity has already changed.  Neither is useful evidence for the actor.
    return True


def _wait_for_trustworthy_a11y_state(
    env: Any,
    initial_state: Any,
    *,
    attempts: int = A11Y_TRANSITION_ATTEMPTS,
    retry_seconds: float = A11Y_TRANSITION_RETRY_SECONDS,
) -> tuple[Any, str]:
    """Wait briefly for a forwarder tree that matches the current app.

    This handles application launch splashes without refreshing AndroidEnv,
    restarting the APK, or falling back to UIAutomator.  A persistent empty or
    stale tree remains a fatal infrastructure error under the strict protocol.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")

    state = initial_state
    last_activity = ""
    last_packages: set[str] = set()
    for attempt in range(attempts):
        last_activity = env.foreground_activity_name
        foreground_package = last_activity.split("/", maxsplit=1)[0]
        elements = list(state.ui_elements)
        last_packages = {
            str(element.package_name)
            for element in elements
            if element.is_visible and element.package_name
        }
        if not _is_untrustworthy_a11y_observation(
            elements,
            foreground_package,
        ):
            return state, last_activity
        if attempt + 1 >= attempts:
            break
        LOGGER.warning(
            "a11y observation is empty or stale during an app transition; "
            "retrying the same AndroidEnv instance (%d/%d)",
            attempt + 1,
            attempts - 1,
        )
        time.sleep(retry_seconds)
        state = get_state_with_a11y_retries(
            env,
            wait_to_stabilize=False,
            attempts=3,
        )

    raise A11yInfrastructureError(
        "Accessibility observation remained empty or stale after "
        f"{attempts} checks on the same AndroidEnv; "
        f"foreground_activity={last_activity!r}, "
        f"visible_packages={sorted(last_packages)!r}. "
        "UIAutomator fallback is disabled."
    )


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

        if _strict_a11y_protocol():
            state, foreground_activity = _wait_for_trustworthy_a11y_state(
                env,
                state,
            )
        else:
            foreground_activity = env.foreground_activity_name
        Image.fromarray(state.pixels).save(screenshot_path)
        package_name = foreground_activity.split("/", maxsplit=1)[0]
        source_elements = list(state.ui_elements)
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
    for attempt in range(attempts):
        try:
            return env.get_state(wait_to_stabilize=wait_to_stabilize)
        except Exception as exc:
            last_error = exc
            if "Could not get a11y tree" not in str(exc) or attempt + 1 >= attempts:
                if "Could not get a11y tree" in str(exc):
                    raise A11yInfrastructureError(
                        "Accessibility tree remained unavailable; the same "
                        "AndroidEnv instance was retained and the run must stop."
                    ) from exc
                raise
            LOGGER.warning(
                "state capture failed to get a11y tree; retrying the same "
                "AndroidEnv instance (%d/%d)",
                attempt + 1,
                attempts - 1,
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

    return get_state_with_a11y_retries(
        env,
        wait_to_stabilize=True,
        attempts=3,
    )


def verify_live_accessibility(env: Any) -> dict[str, Any]:
    """Verify the exact dynamic endpoint and observation used by the run."""
    controller = getattr(env, "controller", None)
    if controller is None:
        raise A11yInfrastructureError(
            "AndroidWorld environment has no controller."
        )
    port = _controller_a11y_server_port(controller)
    if port is None:
        raise A11yInfrastructureError(
            "AndroidEnv has no accessibility gRPC server port."
        )
    if port == 8554:
        raise A11yInfrastructureError(
            "Accessibility endpoint incorrectly equals emulator control port 8554."
        )
    if not _host_port_is_listening(port):
        raise A11yInfrastructureError(
            f"Accessibility gRPC server 127.0.0.1:{port} is not listening."
        )
    # The wrapper has only just broadcast the endpoint.  Let the service emit
    # its first tree before the strict preflight starts consuming observations.
    time.sleep(2.0)
    state = get_state_with_a11y_retries(
        env,
        wait_to_stabilize=True,
        attempts=3,
    )
    state, foreground_activity = _wait_for_trustworthy_a11y_state(
        env,
        state,
    )
    package_name = foreground_activity.split("/", maxsplit=1)[0]
    elements = list(state.ui_elements)
    return {
        "a11y_ready": True,
        "grpc_port": port,
        "grpc_listener_ready": True,
        "ui_element_count": len(elements),
        "foreground_activity": foreground_activity,
    }


def close_androidworld_env(env: Any) -> None:
    """Disable APK forwarding before closing its host-side gRPC owner."""
    controller = getattr(env, "controller", None)
    if controller is not None:
        try:
            _disable_accessibility_forwarding(controller)
        except Exception as exc:  # pragma: no cover - environment dependent.
            LOGGER.warning("failed to disable accessibility forwarding: %s", exc)
    env.close()
