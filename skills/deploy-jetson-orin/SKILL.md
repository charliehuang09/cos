---
name: deploy-jetson-orin
description: Deploy this COS repository to the remote development Jetson Orin and run deployed binaries for testing. Use when Codex is asked to deploy code to dev-orin, test code on the Jetson Orin, SSH into the remote Orin, inspect deployed binaries under /root/bin, or diagnose basic deployment/test issues on the dev Orin.
---

# Deploy Jetson Orin

## Overview

Use this skill to deploy the current repository build to the remote development Jetson Orin and run binaries on the device. Keep the workflow narrow: deploy through the CMake target, connect as root to `dev-orin`, and test binaries from `/root/bin`.

## Standard Workflow

Run deployment commands from the repository root.

1. Deploy to the remote development Orin:

```bash
cmake --build build --target dev-orin
```

If `build/` is missing or CMake reports that the build directory is not configured, configure the repo using the repository's normal CMake build workflow before retrying the `dev-orin` target.

2. Open an interactive shell on the device for remote inspection or testing:

```bash
ssh root@dev-orin
```

Use the interactive shell as the default after deployment so commands run in the same remote session and exploratory testing stays visible. Use direct SSH commands only for simple one-off checks when an interactive session is unnecessary.

## Remote Layout

The remote login lands in `/root`. Known directories include:

- `/root/bin` - deployed binaries to test
- `/root/constants` - deployed constants/configuration
- `/root/lib` - deployed libraries

List `/root/bin` from the interactive shell if the user asks to test code but does not name a binary:

```bash
ls /root/bin
```

Run named binaries from `/root/bin` unless the user gives a different remote path:

```bash
/root/bin/<binary-name>
```

## Reporting

Report the deployment command result first, then the remote command result. Include the exact binary path tested and any failing stderr lines that indicate missing libraries, missing files, permissions, crashes, or assertion failures.

Do not run destructive remote commands, restart services, reboot the device, or modify files outside the requested deployment/test flow unless the user explicitly asks.
