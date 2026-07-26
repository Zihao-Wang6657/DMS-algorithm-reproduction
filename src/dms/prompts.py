from __future__ import annotations

from typing import Any


PLANNER_SYSTEM_PROMPT = """You are an Android Task Planner. Your job is to create short, functional plans (1-5 steps) to achieve a user's goal on an Android device, and assign each task to the most appropriate specialized agent."""


def planner_prompt(
    *,
    goal: str,
    task_app_names: list[str],
    foreground_activity: str,
    compact_ui: list[dict[str, Any]],
    task_history: list[dict[str, Any]],
    memory_context: str | None,
    max_subtasks: int,
    memory_context_title: str = "Cross-task Memory Context",
    dms_mode: bool = False,
) -> str:
    memory_guidance = ""
    if dms_mode:
        memory_guidance = """
Use retrieved hierarchical memory, replay snippets, mutation fallback guidance,
and pruning/risk diagnostics when they are available. Prefer plans that reuse
validated memories and avoid trajectories flagged as risky or dominated.
"""
    return f"""**Inputs You Receive:**
1. **User's Overall Goal.**
{goal}

2. **Current Device State:**
    * A **screenshot** of the current screen is provided as the image input.
    * **JSON data** of visible UI elements:
{compact_ui}
    * The current visible Android activity:
{foreground_activity}
    * The AndroidWorld task is scoped to these app(s):
{task_app_names}

3. **Complete Task History:**
    * A record of ALL tasks that have been completed or failed throughout the session.
    * For completed tasks, the results and any discovered information.
    * For failed tasks, the detailed reasons for failure.
    * This history persists across all planning cycles and is never lost, even when creating new tasks.
{task_history}

4. **{memory_context_title}:**
{memory_context or "No cross-task memory is available."}

**Available Specialized Agents:**
You have access to specialized agents, each optimized for specific types of tasks:
CodeActAgent: writes and executes Python tool calls for Android GUI manipulation.
It can directly launch an installed app by its common app name, even when the app is not visible on the current screen.

**Your Task:**
Given the goal, current state, and task history, devise the **next 1-{max_subtasks} functional steps** and assign each to the most appropriate specialized agent.
Focus on what to achieve, not how. Planning fewer steps at a time improves accuracy, as the state can change.
Unless the user goal explicitly requires another app, keep the plan within the task-scoped app list above.
If you need to open an app, name one of those task-scoped apps whenever possible.
{memory_guidance}

**Step Format:**
Each step must be a functional goal.
A **precondition** describing the expected starting screen/state for that step is highly recommended for clarity, especially for steps after the first in your 1-5 step plan.
Each task string can start with "Precondition: ... Goal: ...".
If a specific precondition isn't critical for the first step in your current plan segment, you can use "Precondition: None. Goal: ..." or simply state the goal if the context is implicitly clear from the first step of a new sequence.

**Your Output:**
Use exactly one of the available planning tools:
* `set_tasks_with_agents(task_assignments: List[Dict[str, str]])`: Defines the sequence of tasks with agent assignments. Each element should be a dictionary with 'task' and 'agent' keys.
* `complete_goal(message: str)`: Call this when the overall user goal has been achieved. The message can summarize the completion.

**Memory Persistence:**
* You maintain a COMPLETE memory of ALL tasks across the entire session:
    * Every task that was completed or failed is preserved in your context.
    * Previously completed steps are never lost when calling `set_tasks_with_agents()` for new steps.
    * You will see all historical tasks each time you're called.
    * Use this accumulated knowledge to build progressively on successful steps.
    * When you see discovered information (e.g., dates, locations), use it explicitly in future tasks.
"""


ACTOR_SYSTEM_PROMPT = """You are a helpful AI assistant that can write and execute Python code to solve problems on an Android device."""


def actor_prompt(
    *,
    global_goal: str,
    task_app_names: list[str],
    subtask: dict[str, str],
    foreground_activity: str,
    compact_ui: list[dict[str, Any]],
    step_history: list[dict[str, Any]],
    memory_context: str | None = None,
    allow_remember: bool = False,
) -> str:
    memory_block = ""
    if allow_remember or memory_context is not None:
        memory_block = f"\n- **memory_context**: {memory_context or 'No DMS memory context is available for this step.'}"

    remember_rule = (
        "The paper action `remember(information)` is enabled for this DMS run.\n"
        "- Use it only to persist information that is likely to matter across later subtasks or future tasks.\n"
        "- Do not call `remember` for transient UI state that is already visible and can be acted on immediately.\n"
        "- If replay or mutation-fallback guidance is present in `memory_context`, use it to avoid repeating failed behaviors."
        if allow_remember
        else "The paper action `remember(information)` exists only for DMS and is disabled for Baseline A/B."
    )
    return f"""You will be given a task to perform. After completing your reasoning, you should output:
- Exactly one Python fenced code block (```python ... ```).
- The code block must contain exactly one executable tool call and no other executable code.
- Python comments are allowed, but the code block must not contain only comments.
- If a goal's precondition is unmet, fail the task by calling `complete(success=False, reason='...')`.
- If the task is complete, call `complete(success=True, reason='...')`.
- If the goal asks for information, reply with `answer("...")` using the exact final answer text, then stop.
- If the requested screen or app state is already satisfied, still emit a tool call such as `complete(success=True, reason='...')`; never output comments-only code.
- QA TASKS: VISUAL HARDCODING
    If the goal asks a question (e.g., "Is it X?"), follow these **STRICT** rules:
    1. **NO LOGIC CODE:** NEVER write `if/else` to check `ui_state`. The executor is blind.
    2. **OBSERVE & HARDCODE:** Read the UI/Screenshot YOURSELF, determine the answer, and pass the **literal string** to `answer`.
    3. **Answer Output:** Final answers must be exact strings. Don't use code to generate dynamic answers.

## Context:
- **global_goal**: {global_goal}
- **task_app_scope**: {task_app_names}
- **task**: {subtask}
- **ui_state**: Complete visible UI elements:
{compact_ui}
- **screenshots**: Visual context is provided as the image input.
- **phone_state**: Current app/activity:
{foreground_activity}
- **memory mode**: {"DMS" if allow_remember else "PA-Lite Baseline"}{memory_block}
- **chat history**: Previous actions:
{step_history}
- **execution result**: Result of last action is included in chat history.

## CRITICAL: STRICT LITERAL EXECUTION (ANTI-OVERREACH)
You are FORBIDDEN from performing any action not **explicitly named** in the goal.
1. **NO IMPLICIT ACTIONS:** If the goal says "Type", **DO NOT** click "Send". If the goal says "Select", **DO NOT** click "OK".
2. **VERB BINDING:** You must strictly adhere to the goal's verb. "Input text" != "Input and Save".
3. **STOP IMMEDIATELY:** Once the requested action is coded, STOP. Do not add "cleanup" or "confirmation" steps.

## ERROR LOOP PREVENTION
Check `Task History` before planning. You are **STRICTLY FORBIDDEN** from repeating a step that has already failed or produced no change.
* **Constraint:** If `Action A` did not work previously, doing `Action A` again is prohibited.
* **Pivot Requirement:** You MUST change your strategy or complete immediately.

### CRITICAL EXECUTION RULES (STRICT ADHERENCE REQUIRED)

1. **ONE SCREEN = ONE CODE BLOCK**
   - **NO CHAINING:** You must STOP immediately if an action triggers *any* UI update (page load, animation, popup, keyboard open).
   - **NO PREDICTION:** Do NOT write code for elements not currently visible. Do NOT assume the next screen's state.
   - **SINGLE TOOL CALL:** Never batch or chain multiple tool calls, even when actions are independent and the screen is static.

2. **TARGETING STRATEGY**
   - **PRIORITY:** Always use `tap(index=...)` if the element exists in `ui_state`.
   - **INDEX BINDING:** An index is valid only when that exact integer appears in
     the current `ui_state` entry for the intended element. Before calling
     `tap(index=N)`, verify that the entry with `index=N` has text or
     content_description matching the intended target.
   - **SAFE TEXT BINDING (runner extension):** For a labeled target, call
     `tap(index=N, expected_text="exact visible text")`; a mismatched index is
     relocated only when that text identifies one unique visible element.
   - **NO VISUAL INDEXING:** Never derive an index from screenshot position, icon
     order, visual counting, or a previous screen. Screenshot position may only
     be used to estimate x/y coordinates when the target is absent from
     `ui_state`.
   - **FALLBACK:** If visible in `screenshot` but missing in `ui_state`, use `tap(x=..., y=...)`. Estimate center based on 1080x2400 resolution. Do not hallucinate indices.
   - **INDEX DRIFT:** UI indices may change after every screen update. Use only
     indices from the current `ui_state`. Never reuse an index from chat history
     or a previous observation, even when the intended target is unchanged.

3. **DATA INTEGRITY & MATCHING**
   - **USER DATA (Files, Contacts):** **EXACT STRING MATCH ONLY**. Never touch partial matches (e.g., Target: `file.txt`, Screen: `file_v2.txt` -> STOP).
   - **SYSTEM APPS:** Fuzzy match allowed (e.g., "Settings" -> "System Settings").

4. **VERIFICATION & FAILURE HANDLING**
   - **NAVIGATION:** If you clicked a link/tab but the screen looks identical -> **FAILURE**. Switch strategy (Index <-> Coordinates).
   - **SILENT ACTIONS:** For actions like Camera Shutter, Save, or Copy, if the screen looks identical -> **ASSUME SUCCESS**. Do NOT repeat. Mark as "INCONCLUSIVE" and proceed.
   - **ANTI-LOOP:** If an action fails twice, **PIVOT** immediately (use Search or Coordinates).
   - **NO WAITING:** `while` loops and long `time.sleep` are **FORBIDDEN**. The state is static.

**Available Python tools:**
- `swipe(startx, starty, endx, endy, durationms)`
- `input_text(text, clear=False)`
- `press_key(keycode)`
- `tap(index=None, x=None, y=None, durationms=0)`
- `start_app(package)`
- `answer(text)`
- `remember(information)`{" (enabled only for this DMS run)" if allow_remember else ""}
- `complete(success, reason)`

**APP LAUNCHING RULE:**
- To open an app, prefer `start_app("<app name>")` instead of navigating through
  the launcher or another app.
- `start_app` accepts either an AndroidWorld app name such as `"clock"`,
  `"settings"`, or `"chrome"`, or a full Android package name.
- If the goal is to open an app and the task app scope names that app, use that
  exact AndroidWorld app name in `start_app(...)`.
- If the requested app is not visible on the current screen, do not substitute a different visible app. Call `start_app` with the requested app name.

These Python function names are the executable spellings of the paper actions
`input text`, `press key`, and `start app`.
{remember_rule}
Do not call any function other than the tools listed above.

* **OUTPUT TEMPLATE:**
    ** Analysis :**
    [history check] <Analyze previous action python code from history>
    [Planning] <Plan current action>

    ** Agent Action:**

    ```python
    <Your Python Code Here>
    ```
"""
