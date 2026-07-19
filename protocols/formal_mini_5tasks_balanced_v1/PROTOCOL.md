# Formal Balanced Mini Experiment Protocol v1

- Protocol ID: `formal_mini_5tasks_balanced_v1`
- Frozen at: `2026-07-19T15:42:29.932043+08:00`
- Scale: 5 tasks × 5 rounds × 3 methods = 75 scored records
- Method order: Baseline A → Baseline B → DMS
- Budget success rule: scoring attempt steps must be strictly below the official budget
- Infrastructure retry: at most once with the identical task, seed, Prompt and parameters
- Resource accounting: cumulative across the infrastructure attempt and final attempt
- Pilot data: explicitly excluded
- Device state: restored from a validated golden AVD before every round
- Experiment state: host-side results, audit logs, memory and RNG state persist across rounds

## Frozen task order

| # | Task | Seed | App | Complexity | Official budget |
| ---: | --- | ---: | --- | ---: | ---: |
| 1 | `SystemWifiTurnOn` | 1047 | settings | 1.0 | 10 |
| 2 | `CameraTakePhoto` | 1032 | camera | 1.0 | 10 |
| 3 | `AudioRecorderRecordAudio` | 1030 | audio recorder | 1.2 | 12 |
| 4 | `SimpleSmsSend` | 1046 | simple sms messenger | 1.2 | 12 |
| 5 | `BrowserDraw` | 1033 | chrome | 2.0 | 20 |

## Task selection disclosure

The fixed list contains two easy, two medium, and one hard task.
Selection was informed by abandoned diagnostic runs, so conclusions are
limited to this mini benchmark and the selection bias is explicit.

## Integrity

Every formal runner process verifies the SHA256 mapping in
`protocol_manifest.json` before execution and before each task.
