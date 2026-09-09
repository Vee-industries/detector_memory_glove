import subprocess
import sys
from pathlib import Path


TESTS = [
    "tests/test_artifact.py",
    "tests/test_multi_window.py",
    "tests/test_multi_window_long.py",
    "tests/test_independent_tiers.py",
    "tests/test_state_builders.py",
]


def main():
    print("=== RUN ALL DETECTOR TESTS ===\n")
    results = []

    for test in TESTS:
        p = Path(test)
        if not p.exists():
            print(f"MISSING {test}")
            results.append((test, "MISSING"))
            continue

        print(f"Running {test} ...")
        proc = subprocess.run(
            [sys.executable, str(p)],
            capture_output=True,
            text=True,
            timeout=180,
        )

        if proc.returncode == 0:
            print(f"PASS  {test}\n")
            results.append((test, "PASS"))
        else:
            print(f"FAIL  {test}")
            print("--- stdout ---")
            print(proc.stdout[-2000:])
            print("--- stderr ---")
            print(proc.stderr[-2000:])
            print()
            results.append((test, "FAIL"))

    print("\n=== SUMMARY ===")
    all_ok = True

    for name, status in results:
        print(f"{status:8s} {name}")
        if status != "PASS":
            all_ok = False

    if all_ok:
        print("\nAll tests passed.")
    else:
        print("\nSome tests failed. See details above.")
        sys.exit(1)


if __name__ == "__main__":
    main()