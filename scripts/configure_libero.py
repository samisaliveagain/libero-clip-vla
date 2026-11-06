"""Write clone-relative LIBERO paths; optionally apply the verified Windows fixes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def expected_paths(root: Path) -> dict[str, str]:
    benchmark = root / "third_party" / "LIBERO" / "libero" / "libero"
    return {
        "assets": (benchmark / "assets").as_posix(),
        "bddl_files": (benchmark / "bddl_files").as_posix(),
        "benchmark_root": benchmark.as_posix(),
        "datasets": (root / "data" / "datasets").as_posix(),
        "init_states": (benchmark / "init_files").as_posix(),
    }


def windows_fixes() -> None:
    if os.name != "nt":
        raise SystemExit("--windows-fixes is only for native Windows.")
    suite_spec = importlib.util.find_spec("robosuite")
    mujoco_spec = importlib.util.find_spec("mujoco")
    if suite_spec is None or mujoco_spec is None:
        raise SystemExit("Install requirements.txt first.")
    suite = Path(suite_spec.origin).parent
    mujoco = Path(mujoco_spec.origin).parent
    private = suite / "macros_private.py"
    if not private.exists():
        shutil.copyfile(suite / "macros.py", private)
    shutil.copyfile(mujoco / "mujoco.dll", suite / "utils" / "mujoco.dll")
    binding = suite / "utils" / "binding_utils.py"
    text = binding.read_text(encoding="utf-8")
    original = '        os.environ["MUJOCO_GL"] = "cgl"\n    else:\n        os.environ["MUJOCO_GL"] = "egl"'
    patched = '        os.environ["MUJOCO_GL"] = "cgl"\n    elif _SYSTEM == "Windows":\n        os.environ["MUJOCO_GL"] = "wgl"\n    else:\n        os.environ["MUJOCO_GL"] = "egl"'
    # Accept the locally verified Windows-only patch as well as this portable one.
    existing = '        os.environ["MUJOCO_GL"] = "cgl"\n    else:\n        os.environ["MUJOCO_GL"] = "wgl"'
    if original in text:
        binding.write_text(text.replace(original, patched, 1), encoding="utf-8")
    elif patched not in text and existing not in text:
        raise SystemExit("Unexpected robosuite renderer source; inspect it before patching.")
    print("Applied native-Windows DLL, private-macros, and WGL compatibility fixes.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate config without writing.")
    parser.add_argument("--windows-fixes", action="store_true")
    args = parser.parse_args()
    if args.check and args.windows_fixes:
        parser.error("Use --check separately from --windows-fixes.")
    paths = expected_paths(ROOT)
    for key in ("assets", "bddl_files", "init_states"):
        if not Path(paths[key]).is_dir():
            raise SystemExit("Initialize the pinned LIBERO submodule first: git submodule update --init")
    config = ROOT / ".libero" / "config.yaml"
    if args.check:
        import yaml
        actual = yaml.safe_load(config.read_text(encoding="utf-8"))
        if actual != paths:
            raise SystemExit("LIBERO paths differ; run python scripts/configure_libero.py.")
        print("LIBERO config matches this checkout.")
        return
    config.parent.mkdir(exist_ok=True)
    # JSON strings are valid quoted YAML scalars, including paths with spaces.
    config.write_text("".join(f"{key}: {json.dumps(value)}\n" for key, value in paths.items()), encoding="utf-8")
    if args.windows_fixes:
        windows_fixes()
    print("Configured", config)


if __name__ == "__main__":
    main()
