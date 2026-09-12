---
name: Python test runner
description: Replit Python 3.13 packaging behavior and the project's reliable test entry point.
---

The Replit Python 3.13 base environment is externally managed and does not expose pip for ad-hoc test dependency installs. Keep verification runnable with the standard library or dependencies already provisioned by the package manager.

**Why:** Installing pytest with pip was blocked by the immutable externally-managed environment, while the application dependencies were already available through Replit's language package flow.

**How to apply:** Use `python tests/run_tests.py` for this project. If a future test needs a new library, first try the Replit package-management flow; otherwise prefer a standard-library implementation.