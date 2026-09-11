# WebUI GPU Selection Implementation Plan

## Purpose

Add a GPU selector to the WebUI so that, on machines with multiple NVIDIA GPUs, the user can choose one GPU and run all WebUI-launched GPU work on that single device.

The research/analysis Python code must remain completely unchanged.

## Current branch / checkpoint

- Branch: `feat/webui-gpu-selection`
- Start milestone commit: `c55c24d5746df154ff4b14c8d64bbca3f338c97d`
- The milestone commit contains no file changes and marks the beginning of this work.

## PRIMARY

Implement GPU selection only in the WebUI layer.

The WebUI should expose available NVIDIA GPUs in the settings screen, save the selected GPU, and apply that selection to all GPU-capable child processes launched by the WebUI by setting `CUDA_VISIBLE_DEVICES` in the child process environment.

This is intentionally not a multi-GPU scheduler. Only one selected GPU is exposed to each WebUI-launched process.

## Acceptance criteria

1. The settings page shows a GPU selector on NVIDIA systems where GPUs can be enumerated.
2. The selector includes an `Auto` option that preserves the existing behavior.
3. Available GPUs are displayed with enough information to identify them, ideally index, model name, and memory.
4. A selected GPU is persisted in the existing WebUI settings mechanism.
5. New jobs capture the GPU selection that was active when the job was created.
6. The selected GPU is applied through `CUDA_VISIBLE_DEVICES` to all relevant child processes launched by the WebUI.
7. The same selection applies consistently to:
   - normal analysis
   - rembg analysis
   - model check
   - focus preview
8. With a selected physical GPU, the existing research code sees that device as its only CUDA-visible GPU, normally as logical `cuda:0`.
9. `Auto` does not override `CUDA_VISIBLE_DEVICES`; existing external configuration remains intact.
10. If NVIDIA GPU enumeration fails, `nvidia-smi` is unavailable, no NVIDIA GPU exists, or the platform is unsupported, the WebUI must continue to work with `Auto` only and must not fail startup.
11. No new dependency is added.
12. No file outside `webui/` is modified.

## Preserve

The following behavior must remain unchanged unless required solely to expose/apply GPU selection:

- existing model selection
- YOLO/CNN thresholds
- rembg settings
- job queue behavior
- existing concurrency limits
- output layout and download behavior
- model-check behavior
- focus-preview behavior
- CLI behavior outside the WebUI
- all research/analysis algorithms

Existing installations without a new GPU setting must continue to load successfully and default to `Auto`.

## Forbidden changes

Do not modify any file outside `webui/`.

In particular, do not modify:

- `muscut.py`
- `muscut_with_rembg.py`
- `focus_threshold_checker.py`
- `batcher*.py`
- `muscut_tools/`
- `muscut_functions/`
- `muscut_models/`
- `pyproject.toml`

Also forbidden:

- multi-GPU load balancing
- automatic free-VRAM-based scheduling
- changes to TensorFlow device-selection logic
- changes to PyTorch / Ultralytics device-selection logic
- changing WebUI concurrency limits as part of this task
- dependency additions
- unrelated refactoring, cleanup, renaming, or architecture changes
- speculative fixes to adjacent issues

## Allowed change boundary

Changes are restricted to `webui/`.

Expected files are primarily:

- `webui/app.py`
- `webui/templates/settings.html`

Optional only if required for presentation:

- `webui/static/style.css`

Other files under `webui/` may be changed only if directly necessary to satisfy the acceptance criteria above.

## Implementation design

### 1. Enumerate NVIDIA GPUs in the WebUI layer

Use `nvidia-smi` from the WebUI backend rather than importing TensorFlow or PyTorch solely for device discovery.

The WebUI should attempt to obtain, at minimum:

- physical GPU index
- GPU name
- GPU UUID
- total memory when readily available

GPU UUID should be preferred as the persisted runtime identifier because it is more stable than device ordering.

Enumeration failure must be non-fatal.

### 2. Add a persisted GPU setting

Extend the existing WebUI settings with a GPU-selection field, e.g. `selected_gpu`.

Default value:

`auto`

The validation logic must accept only:

- `auto`
- a GPU identifier that is present in the currently enumerated allowed GPU list

Do not trust arbitrary form input as an environment-variable value.

### 3. Add the selector to the settings page

Example UI:

```text
GPU to use
[ Auto                                      v ]
  Auto
  GPU 0 - NVIDIA RTX 4090 - 24564 MiB
  GPU 1 - NVIDIA RTX 4090 - 24564 MiB
```

The wording may be adapted to the existing Japanese UI style.

If no GPU list is available, keep `Auto` available and avoid presenting the failure as a fatal error.

### 4. Snapshot GPU selection per job

When a job is created, store the selected GPU with that job.

Do not read the global settings again only at execution time, because a queued job must not silently switch GPUs if the user changes settings before it starts.

This rule applies to normal jobs and should be mirrored for model-check and focus-preview jobs as appropriate.

### 5. Apply GPU selection only at child-process launch

For each relevant `subprocess.Popen(...)`, start from a copy of the current process environment.

If the job uses `Auto`:

- do not set or remove `CUDA_VISIBLE_DEVICES`
- preserve the inherited environment exactly

If the job uses a specific GPU:

- set `CUDA_VISIBLE_DEVICES` to the validated selected GPU identifier, preferably its UUID

Do not pass `cuda:1`, `-d`, or any new GPU argument into the research code.

The research code should remain unaware of physical GPU numbering.

### 6. Apply consistently to all WebUI-launched GPU work

Ensure the selected environment is used by:

- standard analysis child process
- rembg analysis child process
- model-check child process
- focus-preview child process

Any additional WebUI-launched process should only receive the selected GPU environment if it belongs to the same analysis workflow and directly uses CUDA-capable libraries.

## Important behavior of CUDA_VISIBLE_DEVICES

If physical GPU 1 is selected and the child environment contains only that device in `CUDA_VISIBLE_DEVICES`, the existing application typically sees one CUDA device and refers to it as logical `cuda:0`.

That is desired. Do not rewrite the research code to use the physical index.

## Auto behavior

`Auto` means the WebUI does not interfere with CUDA visibility.

Examples:

- single-GPU machine: existing behavior remains unchanged
- externally launched with `CUDA_VISIBLE_DEVICES=2`: preserve that value
- no NVIDIA GPU: existing CPU/fallback behavior remains unchanged

## Failure behavior

GPU enumeration is an optional WebUI capability, not a startup requirement.

On enumeration failure:

- keep the WebUI operational
- expose only `Auto` if necessary
- do not alter existing device behavior
- do not add fallback imports of TensorFlow/PyTorch just to enumerate GPUs

## Verification

After implementation, verify all of the following:

1. Branch is still `feat/webui-gpu-selection`.
2. `git diff` / compare against the milestone shows changes only under `webui/`.
3. Existing settings load without a `selected_gpu` key.
4. `Auto` saves and reloads correctly.
5. A discovered GPU can be selected and saves/reloads correctly.
6. Invalid GPU identifiers from form input are rejected or safely normalized to the allowed behavior.
7. A new normal job snapshots its selected GPU.
8. A new model-check job snapshots its selected GPU.
9. A new focus-preview job snapshots its selected GPU.
10. Specific-GPU child processes receive the expected `CUDA_VISIBLE_DEVICES` value.
11. `Auto` child processes preserve an inherited `CUDA_VISIBLE_DEVICES` value rather than deleting or replacing it.
12. `nvidia-smi` failure does not prevent the WebUI from starting or opening settings.
13. No external dependency was introduced.
14. No file outside `webui/` changed.

If GPU hardware is not available in the implementation environment, environment-construction and enumeration-failure paths must still be tested mechanically, and actual multi-GPU hardware verification must be reported as not performed rather than guessed.

## Stop / recovery rule

If satisfying the task appears to require modifying any file outside `webui/`, stop before making that change and report the dependency.

If an existing WebUI behavior conflicts with this design in a way that materially widens the scope, do not refactor around it. Preserve the current branch state, report the conflict, and request review before expanding scope.

If a test fails, diagnose the failure inside the allowed boundary before making further changes. Do not alter research code to make a WebUI test pass.

## Completion report requirements

Before this branch is considered ready for review, report:

- files changed
- exact behavior added
- how GPU enumeration works
- how the selected GPU is stored
- where `CUDA_VISIBLE_DEVICES` is applied
- confirmation that `Auto` preserves inherited environment behavior
- tests/checks performed and their results
- whether real multi-GPU hardware was tested
- confirmation that no file outside `webui/` changed
- final branch HEAD SHA

Do not merge to `main` until review has passed.
