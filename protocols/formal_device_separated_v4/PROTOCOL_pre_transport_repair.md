# Formal Main Experiment Protocol v1

- Protocol ID: `formal_main_device_separated_v4`
- Frozen at: `2026-07-19T12:28:00.937764+08:00`
- Scale: 20 tasks × 5 rounds × 3 methods = 300 scored records
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
| 1 | `AudioRecorderRecordAudio` | 1030 | audio recorder | 1.2 | 12 |
| 2 | `RecipeAddSingleRecipe` | 1031 | broccoli app | 2.4 | 24 |
| 3 | `CameraTakePhoto` | 1032 | camera | 1.0 | 10 |
| 4 | `BrowserDraw` | 1033 | chrome | 2.0 | 20 |
| 5 | `ClockStopWatchRunning` | 1034 | clock | 1.0 | 10 |
| 6 | `ContactsAddContact` | 1035 | contacts | 1.2 | 12 |
| 7 | `FilesDeleteFile` | 1036 | files | 2.2 | 22 |
| 8 | `NotesTodoItemCount` | 1037 | joplin | 1.0 | 10 |
| 9 | `MarkorCreateFolder` | 1038 | markor | 1.0 | 10 |
| 10 | `SportsTrackerActivityDuration` | 1039 | open tracks sports tracker | 1.0 | 10 |
| 11 | `OsmAndFavorite` | 1040 | osmand | 1.3 | 13 |
| 12 | `ExpenseAddSingle` | 1041 | pro expense | 1.2 | 12 |
| 13 | `RetroCreatePlaylist` | 1042 | retro music | 2.4 | 24 |
| 14 | `SimpleCalendarDeleteOneEvent` | 1043 | simple calendar pro | 1.2 | 12 |
| 15 | `SimpleDrawProCreateDrawing` | 1044 | simple draw pro | 1.8 | 18 |
| 16 | `SaveCopyOfReceiptTaskEval` | 1045 | simple gallery pro | 1.6 | 16 |
| 17 | `SimpleSmsSend` | 1046 | simple sms messenger | 1.2 | 12 |
| 18 | `SystemWifiTurnOn` | 1047 | settings | 1.0 | 10 |
| 19 | `TasksDueOnDate` | 1048 | tasks | 1.0 | 10 |
| 20 | `VlcCreatePlaylist` | 1049 | vlc | 2.8 | 28 |

## Contacts compatibility selection

`ContactsAddContact` was selected before freeze because its official
database-based evaluator is compatible with the UIAutomator runtime.
It is not claimed to be equivalent to `ContactsNewContactDraft`.

## Integrity

Every formal runner process verifies the SHA256 mapping in
`protocol_manifest.json` before execution and before each task.
