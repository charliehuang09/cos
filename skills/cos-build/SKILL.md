---
name: cos-build
description: Configure and build this COS CMake repository. Use when Codex is asked to prepare this repo after checkout, initialize or update submodules, configure the Ninja build directory, list the repo contents during setup, run the CMake build, or diagnose basic setup/build failures for this project.
---

# COS Build

## Overview

Use this skill to perform the standard setup and build sequence for this repository. The expected flow is submodule initialization, CMake configuration with Ninja, a quick top-level listing when useful, then the CMake build.

## Standard Workflow

Run commands from the repository root.

1. Initialize and update submodules:

```bash
git submodule init
git submodule update
```

If the repo is in a linked worktree and Git cannot lock submodule config because the git metadata is outside the writable sandbox, rerun the same submodule command with the required approval instead of changing the workflow.

2. Configure CMake with Ninja:

```bash
cmake -B build -G Ninja
```

Treat a successful configure as the point where CMake has generated build files under `build/`.

3. If the user asks to inspect the repo or says to list files, run:

```bash
ls
```

4. Build through CMake:

```bash
cmake --build build
```

Report compiler, linker, dependency, or configuration errors with the failing target and the first actionable diagnostic. If the build reports `ninja: no work to do.`, treat that as a successful up-to-date build.

## Expected Signals

During CMake configuration, project-local dependencies may be reported from `third_party/`, including `cpp-mjpeg-streamer`, `json`, and `libusb-1.0` via pkg-config. Do not treat those messages as errors unless CMake exits nonzero.

Keep the workflow narrow. Do not introduce alternate generators, extra build directories, dependency managers, or cleanup commands unless the user asks or an observed failure requires it.
