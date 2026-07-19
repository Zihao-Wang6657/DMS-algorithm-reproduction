# Paper Alignment Notes

This file distinguishes paper-specified behavior from local AndroidWorld
execution details for the DMS reproduction.

## Sources

- Reproduction assignment:
  `docs/references/project_requirements.pdf`
- Reference paper:
  `docs/references/darwinian_memory_mi_2026.pdf`
- AndroidWorld paper:
  `docs/references/androidworld_rawles_2025.pdf`
- AndroidWorld local source:
  `third_party/android_world`

## Baselines Required by the Assignment

- Baseline A: pure zero-shot VLM with no memory mechanism.
- Baseline B: traditional static memory, appending historical interaction
  trajectories in chronological order without pruning.
- DMS: the PA-Lite backbone augmented with the dedicated Darwinian memory
  backend.

## Paper-Specified PA-Lite Structure

The reference paper defines a canonical Planner-Actor baseline:

- Planner decomposes high-level task `T` into executable sub-tasks:
  `P(T, o_t, q) = {p_1, ..., p_k}`, where `k <= 5`.
- Each sub-task should be structured as `p_i = <Precondition, Goal>`.
- Actor executes atomic actions for each sub-task.
- Control returns to the Planner when the Actor finishes the sequence or fails
  a sub-task.
- The process repeats until the global task succeeds or a global step limit is
  exceeded.

## Paper-Specified Action Space

Appendix A lists:

- `swipe(startx, starty, endx, endy, durationms)`
- `input text(text, clear)`
- `press key(press key)`
- `tap(index, x, y, durationms)`
- `start app(package)`
- `remember(information)`
- `complete(success, reason)`

For Baseline A/B, model-visible executable actions are restricted to:

- `swipe`
- `input_text`
- `press_key`
- `tap`
- `start_app`
- `complete`

The paper's `remember(information)` action is a memory-writing tool action and
is disabled until the DMS milestone. The local AndroidWorld environment exposes
compatible JSON actions such as `click`, `long_press`, `input_text`,
`press_keyboard`, `swipe`, `open_app`, and `status`; the reproduction code maps
paper-style model outputs to AndroidWorld `JSONAction` objects rather than
changing AndroidWorld itself. AndroidWorld `status` is used only as the
execution-layer representation of paper `complete(success, reason)`, and is not
exposed as a model action.

AndroidWorld also exposes actions that are not in the DMS paper's action table,
including `wait`, `scroll`, `navigate_back`, `navigate_home`, `answer`, and
`double_tap`. These are not part of the paper action space and should not be
used to define the baseline policy.

## Prompt Alignment and Local Adapters

Appendix H provides the Planner and CodeAct prompts. Baseline A/B now use those
prompt semantics:

- Planner role: "Android Task Planner" creating 1-5 functional steps.
- Planner inputs: goal, screenshot, JSON UI elements, visible activity, and
  complete task history.
- Planner step format: functional goals with recommended
  `Precondition: ... Goal: ...` structure.
- Planner memory persistence: all task history remains visible across planning
  cycles.
- CodeAct/Actor constraints: precondition failure handling, QA visual
  hardcoding, strict literal execution, anti-overreach, anti-loop rules,
  one-screen execution, index-first targeting, exact matching for user data,
  and no waiting.

Two local adapters remain necessary because this reproduction runner is not a
native Python tool executor:

- The paper's Planner tools, `set_tasks_with_agents(...)` and
  `complete_goal(...)`, are represented as tool-call semantics in the planner
  prompt and normalized from the model output.
- The paper's CodeAct output is a Python code block that calls tools; this
  runner parses that code block, whitelists the paper tools, and maps the
  resulting paper action to AndroidWorld `JSONAction`.

The adapters preserve the paper action names and accept paper aliases such as
`input text`, `press key`, `start app`, `startx`, and `durationms`, but they do
not add new capabilities.

## DMS Parameters From Appendix B

These values are used by the DMS configuration and are not applied to either
baseline:

- Novelty bonus: `Vnew = 1.0`
- Base protection period: `Tbase = 30.0`
- Longevity coefficient in the paper appendix text: `alpha = 15.0`
- Decay steepness: `beta = 0.5`
- Penalty coefficient: `gamma = 1.0`
- Dynamic threshold sensitivity: `lambda = 0.3`
- Verification depth: `K = 3`

These parameters must not be used to make Baseline A/B look like DMS.

## Runtime Values Not Specified by the DMS Paper

The paper names `MaxP` and `MaxA` but does not provide numeric values in the
visible text. For finite execution during baselines, this project uses
AndroidWorld's own runtime convention:

- Per-task step budget: `int(task.complexity * 10)`.

This is an AndroidWorld runtime limit, not a DMS algorithm parameter.

## Implemented DMS Scope

The active backend in `src/dms/darwinian_memory.py` implements Survival
Value scoring, dual-factor retrieval, Bayesian risk suppression, epsilon
mutation with evolutionary replacement, and elbow-based dynamic pruning.
Baseline A and Baseline B remain isolated from these mechanisms.
