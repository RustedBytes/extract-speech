from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

EXPECTED_INTERPRETERS = {
    "cp39": "3.9",
    "cp310": "3.10",
    "cp311": "3.11",
    "cp312": "3.12",
    "cp313": "3.13",
    "cp314": "3.14",
    "cp315": "3.15",
}


def interpreter_tag(wheel: Path) -> str:
    match = re.search(r"-(cp3(?:9|10|11|12|13|14|15))-", wheel.name)
    if match is None:
        raise ValueError(f"wheel has an unsupported interpreter tag: {wheel.name}")
    return match.group(1)


def verify_wheel_contents(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        files = set(archive.namelist())

    required_suffixes = (
        "extract_speech/__init__.pyi",
        "extract_speech/py.typed",
        ".dist-info/licenses/LICENSE",
    )
    for suffix in required_suffixes:
        if not any(name.endswith(suffix) for name in files):
            raise AssertionError(f"{wheel.name} does not contain {suffix}")


def smoke_test(wheel: Path, python_version: str) -> None:
    subprocess.run(
        [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--python",
            python_version,
            "--with",
            str(wheel.resolve()),
            "python",
            "-c",
            (
                "import extract_speech; "
                "assert extract_speech.SAMPLE_RATE == 16_000; "
                "assert extract_speech.__version__"
            ),
        ],
        check=True,
    )


def main() -> None:
    wheel_directory = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    wheels_by_tag: dict[str, Path] = {}
    for wheel in sorted(wheel_directory.glob("*.whl")):
        tag = interpreter_tag(wheel)
        if tag in wheels_by_tag:
            raise AssertionError(f"multiple wheels found for {tag}")
        wheels_by_tag[tag] = wheel

    if wheels_by_tag.keys() != EXPECTED_INTERPRETERS.keys():
        missing = sorted(EXPECTED_INTERPRETERS.keys() - wheels_by_tag.keys())
        unexpected = sorted(wheels_by_tag.keys() - EXPECTED_INTERPRETERS.keys())
        raise AssertionError(f"unexpected wheel set; missing={missing}, unexpected={unexpected}")

    for tag, python_version in EXPECTED_INTERPRETERS.items():
        wheel = wheels_by_tag[tag]
        verify_wheel_contents(wheel)
        smoke_test(wheel, python_version)
        print(f"Tested {wheel.name} with Python {python_version}")


if __name__ == "__main__":
    main()
