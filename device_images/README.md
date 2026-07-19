# Device images

The Android emulator golden image and AndroidWorld application snapshots are
local runtime assets and are intentionally not stored in Git. Together they are
about 5.3 GiB and include emulator and application state that should not be
published.

`AndroidWorld_device_state_v3_manifest.json` records the expected asset paths,
sizes, hashes, and complete tree hash used by the formal experiment. Generate or
restore the local assets before running the device-separated experiment.
