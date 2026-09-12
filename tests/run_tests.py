"""Small dependency-free test runner for the Replit Python environment."""
import importlib
import inspect
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    test_modules = ["tests.test_services", "tests.test_api"]
    failures = []
    total = 0
    for module_name in test_modules:
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                function()
                print(f"PASS {module_name}.{name}")
            except Exception as exc:
                failures.append((module_name, name, exc))
                print(f"FAIL {module_name}.{name}: {exc}")
    print(f"\n{total - len(failures)}/{total} tests passed")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())