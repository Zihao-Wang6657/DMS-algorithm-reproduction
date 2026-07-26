from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE / "src"))
sys.path.insert(0, str(WORKSPACE / "third_party" / "android_world"))

from env import androidworld_env  # noqa: E402


class _FakeAdbUtils:
    def __init__(self) -> None:
        self.commands: list[str] = []

    def issue_generic_request(
        self,
        command: str,
        controller: object,
        *,
        timeout_sec: int,
    ) -> None:
        del controller, timeout_sec
        self.commands.append(command)


def test_dynamic_a11y_port_is_not_emulator_control_port() -> None:
    wrapper = SimpleNamespace(
        get_port=lambda: 43127,
        _coordinator=SimpleNamespace(
            _simulator=SimpleNamespace(
                _config=SimpleNamespace(
                    emulator_launcher=SimpleNamespace(grpc_port=8554)
                )
            )
        ),
    )
    controller = SimpleNamespace(env=wrapper)

    assert androidworld_env._controller_a11y_server_port(controller) == 43127


def test_close_only_disables_forwarding_then_closes(monkeypatch) -> None:
    fake_adb = _FakeAdbUtils()
    events: list[str] = []
    env = SimpleNamespace(
        controller=SimpleNamespace(),
        close=lambda: events.append("close"),
    )
    monkeypatch.setattr(androidworld_env, "_load_adb_utils", lambda: fake_adb)

    androidworld_env.close_androidworld_env(env)

    assert events == ["close"]
    assert len(fake_adb.commands) == 2
    assert "DISABLE_ACCESSIBILITY_TREE_LOGS" in fake_adb.commands[0]
    assert "--ei port 0" in fake_adb.commands[1]
    assert all("force-stop" not in command for command in fake_adb.commands)
    assert all("--ei port 8554" not in command for command in fake_adb.commands)
    assert all("uiautomator" not in command for command in fake_adb.commands)


def test_state_retry_keeps_same_environment(monkeypatch) -> None:
    calls = 0

    class _Env:
        def get_state(self, *, wait_to_stabilize: bool) -> str:
            nonlocal calls
            del wait_to_stabilize
            calls += 1
            if calls < 3:
                raise RuntimeError("Could not get a11y tree")
            return "ok"

    monkeypatch.setattr(androidworld_env.time, "sleep", lambda _: None)

    assert androidworld_env.get_state_with_a11y_retries(
        _Env(),
        wait_to_stabilize=False,
        attempts=3,
    ) == "ok"
    assert calls == 3


def test_state_retry_fails_without_refresh_or_fallback(monkeypatch) -> None:
    class _Env:
        def get_state(self, *, wait_to_stabilize: bool) -> None:
            del wait_to_stabilize
            raise RuntimeError("Could not get a11y tree")

    monkeypatch.setattr(androidworld_env.time, "sleep", lambda _: None)

    with pytest.raises(androidworld_env.A11yInfrastructureError):
        androidworld_env.get_state_with_a11y_retries(
            _Env(),
            wait_to_stabilize=False,
            attempts=2,
        )


def test_system_overlay_is_not_mistaken_for_empty_a11y_tree() -> None:
    element = SimpleNamespace(
        is_visible=True,
        package_name="com.google.android.permissioncontroller",
    )

    assert not androidworld_env._is_untrustworthy_a11y_observation(
        [element],
        "com.dimowner.audiorecorder",
    )


def test_previous_app_tree_is_rejected_during_foreground_transition() -> None:
    elements = [
        SimpleNamespace(
            is_visible=True,
            package_name="com.google.android.apps.nexuslauncher",
        ),
        SimpleNamespace(
            is_visible=True,
            package_name="com.android.systemui",
        ),
    ]

    assert androidworld_env._is_untrustworthy_a11y_observation(
        elements,
        "com.android.camera2",
    )


def test_systemui_only_tree_is_rejected_during_app_splash() -> None:
    element = SimpleNamespace(
        is_visible=True,
        package_name="com.android.systemui",
    )

    assert androidworld_env._is_untrustworthy_a11y_observation(
        [element],
        "com.android.camera2",
    )


def test_empty_a11y_tree_remains_an_infrastructure_failure() -> None:
    assert androidworld_env._is_untrustworthy_a11y_observation(
        [],
        "com.dimowner.audiorecorder",
    )


def test_transition_retry_waits_for_foreground_app_tree(monkeypatch) -> None:
    empty_state = SimpleNamespace(ui_elements=[])
    stale_state = SimpleNamespace(
        ui_elements=[
            SimpleNamespace(
                is_visible=True,
                package_name="com.google.android.apps.nexuslauncher",
            )
        ]
    )
    systemui_state = SimpleNamespace(
        ui_elements=[
            SimpleNamespace(
                is_visible=True,
                package_name="com.android.systemui",
            )
        ]
    )
    camera_state = SimpleNamespace(
        ui_elements=[
            SimpleNamespace(
                is_visible=True,
                package_name="com.android.camera2",
            )
        ]
    )

    class _Env:
        foreground_activity_name = (
            "com.android.camera2/com.android.camera.CameraLauncher"
        )

        def __init__(self) -> None:
            self.states = iter(
                [stale_state, systemui_state, camera_state]
            )
            self.calls = 0

        def get_state(self, *, wait_to_stabilize: bool) -> object:
            assert not wait_to_stabilize
            self.calls += 1
            return next(self.states)

    sleeps: list[float] = []
    monkeypatch.setattr(
        androidworld_env.time,
        "sleep",
        lambda seconds: sleeps.append(seconds),
    )
    env = _Env()

    state, activity = androidworld_env._wait_for_trustworthy_a11y_state(
        env,
        empty_state,
        attempts=4,
        retry_seconds=0.75,
    )

    assert state is camera_state
    assert activity == env.foreground_activity_name
    assert env.calls == 3
    assert sleeps == [0.75, 0.75, 0.75]


def test_transition_retry_fails_closed_without_refresh_or_fallback(
    monkeypatch,
) -> None:
    empty_state = SimpleNamespace(ui_elements=[])

    class _Env:
        foreground_activity_name = (
            "com.android.camera2/com.android.camera.CameraLauncher"
        )

        def __init__(self) -> None:
            self.calls = 0

        def get_state(self, *, wait_to_stabilize: bool) -> object:
            assert not wait_to_stabilize
            self.calls += 1
            return empty_state

    monkeypatch.setattr(androidworld_env.time, "sleep", lambda _: None)
    env = _Env()

    with pytest.raises(
        androidworld_env.A11yInfrastructureError,
        match="remained empty or stale",
    ):
        androidworld_env._wait_for_trustworthy_a11y_state(
            env,
            empty_state,
            attempts=3,
            retry_seconds=0.0,
        )

    assert env.calls == 2
