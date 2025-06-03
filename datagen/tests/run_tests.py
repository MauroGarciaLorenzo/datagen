import subprocess
import re

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
