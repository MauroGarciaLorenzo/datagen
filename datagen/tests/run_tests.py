import subprocess
import re

def normalize_warning(line):
    line = re.sub(r"^\s*[-\d:, ]+-\s+", "", line)
    line = re.sub(r"Iteration \d+: ", "Iteration X: ", line)
    return line.strip()

def run_unittests_and_parse():
    print("Running tests...")
    result = subprocess.run(
        ["python", "-m", "unittest", "discover"],
        capture_output=True,
        text=True
    )

    output = result.stdout + result.stderr

    # Save output (optional)
    with open("tests.txt", "w") as f:
        f.write(output)

    # Collect normalized warnings
    warnings = set()
    for line in output.splitlines():
        if re.search(r"warning", line, flags=re.IGNORECASE):
            norm = normalize_warning(line)
            warnings.add(norm)

    if warnings:
        print("\n⚠️  WARNINGS DETECTED:")
        for w in sorted(warnings):
            print(f"  - {w}")

    # Check if tests failed
    if "FAILED" not in output:
        print("\n✅ All tests passed!")
        return

    # Extract failure blocks
    failure_blocks = re.findall(
        r"=+\n(FAIL|ERROR): (.*?) \((.*?)\)\n-+\n(.*?)(?=\n=+|\Z)",
        output,
        flags=re.DOTALL
    )

    print("\n========== FAILED TESTS ==========")
    for kind, test_name, location, details in failure_blocks:
        print(f"\n🔴 {kind}: {test_name} [{location}]\n{details.strip()}\n")

if __name__ == "__main__":
    run_unittests_and_parse()
