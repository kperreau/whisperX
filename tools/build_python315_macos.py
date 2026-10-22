#!/usr/bin/env python3
"""Build an isolated Python 3.15.1 CPU environment on Apple silicon.

The regular project environment is untouched. Missing native wheels are built
from pinned upstream releases. For distribution wheels, use build_python315_native.py.
"""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.15.1"
SOURCES = {
    "ctranslate2": ("OpenNMT/CTranslate2", "v4.8.3", "8d91aa08225004755360daa43deb62b35b1ed65d"),
    "onnxruntime": ("microsoft/onnxruntime", "v1.31.0", "402979ac1bef89dcd55add41eaa7277c407599eb"),
    "torchaudio": ("pytorch/audio", "v2.11.0", "34c52a67e8941bbd8e6adaca0eb0b9eabec11d78"),
    "pybind11": ("pybind/pybind11", "v3.1.0", "97bf890db679505a14dfe547a5e77bb2bd05dc90"),
}
BINARIES = {
    "torch-2.14.1": "729992e6b581de722989012547b7767ef8ac019bd060e34805f06a52ff9a120e",
    "torchvision-0.29.1": "81631181cff6389b76195c770dca6292321bffc6fd7954bb72dd690c39c7e375",
    "torchcodec-0.17.0": "d37b11569191c9215240591057138ab66dcd7f9ee79049414b94d3a88521a6bc",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "tmp/python315")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("This recipe has only been validated on macOS arm64 with CPU inference.")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    for tool in ("uv", "cmake", "git", "cargo", "clang"):
        if not shutil.which(tool):
            parser.error(f"Required build tool is missing: {tool}")

    work = args.work_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    wheels = work / "wheels"
    wheels.mkdir(exist_ok=True)
    env = os.environ.copy()
    uv = ["uv", "--no-config", "--cache-dir", str(work / "uv-cache")]

    def run(command, name, cwd=work, extra=None):
        print(f"{name} (log: {work / (name + '.log')})", flush=True)
        with (work / (name + ".log")).open("w") as output:
            subprocess.run([str(v) for v in command], cwd=cwd,
                           env=env | (extra or {}), stdout=output,
                           stderr=subprocess.STDOUT, check=True)

    def wheel(directory, pattern):
        matches = list(directory.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one wheel for {pattern}, found {matches}")
        return matches[0]

    for name, (repository, tag, commit) in SOURCES.items():
        target = work / (name + "-src")
        if not target.exists():
            run(["git", "clone", "--depth", "1", "--branch", tag,
                 f"https://github.com/{repository}.git", target], name + "-clone")
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=target, text=True).strip()
        if commit and actual != commit:
            raise RuntimeError(f"Unexpected source revision for {name}: {actual}")
        subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=target, check=True)

    run(uv + ["python", "install", PYTHON_VERSION, "--install-dir", work / "python", "--no-bin"],
        "python-install")
    base_python = work / f"python/cpython-{PYTHON_VERSION}-macos-aarch64-none/bin/python3.15"
    python = work / "venv/bin/python"
    if not python.exists():
        run(uv + ["venv", "--python", base_python, work / "venv"], "venv")
    actual = subprocess.check_output([python, "-c", "import platform; print(platform.python_version())"], text=True).strip()
    if actual != PYTHON_VERSION:
        raise RuntimeError(f"Build environment uses {actual}, expected {PYTHON_VERSION}")
    env.update(PATH=str(work / "venv/bin") + os.pathsep + env["PATH"],
               VIRTUAL_ENV=str(work / "venv"), CARGO_HOME=str(work / "cargo-home"),
               CARGO_BUILD_JOBS=str(args.jobs))
    binary_urls = [f"https://download.pytorch.org/whl/cpu/{name}-cp315-cp315-macosx_14_0_arm64.whl#sha256={digest}"
                   for name, digest in BINARIES.items()]
    run(uv + ["pip", "install", "--python", python, *binary_urls,
              "pybind11==3.1.0", "setuptools==84.2.1", "wheel", "build", "ninja",
              "numpy==2.5.4", "pyyaml", "maturin>=1.15,<2", "pytest>=9.1.1,<10",
              "delocate", "flatbuffers", "packaging", "protobuf>=4.25.8", "psutil", "jinja2",
              "pip>=26.2.2", "certifi>=2026.10.19"], "build-environment")

    ct2 = work / "ctranslate2-src"
    run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1",
         "third_party/spdlog", "third_party/ruy", "third_party/cpu_features"], "ct2-submodules", ct2)
    prefix = work / "ct2-prefix"
    run(["cmake", "-S", ct2, "-B", work / "ct2-build", "-DCMAKE_BUILD_TYPE=Release",
         f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
         "-DWITH_MKL=OFF", "-DWITH_ACCELERATE=ON", "-DWITH_RUY=ON", "-DWITH_DNNL=OFF",
         "-DOPENMP_RUNTIME=NONE", "-DBUILD_CLI=OFF", "-DBUILD_TESTS=OFF"], "ct2-configure")
    run(["cmake", "--build", work / "ct2-build", "--parallel", args.jobs], "ct2-build")
    run(["cmake", "--install", work / "ct2-build"], "ct2-install")
    run([python, "setup.py", "bdist_wheel", "--dist-dir", wheels], "ct2-wheel", ct2 / "python",
        {"CTRANSLATE2_ROOT": str(prefix), "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs),
         "MACOSX_DEPLOYMENT_TARGET": "14.0"})
    repaired = work / "repaired-wheels"
    run([work / "venv/bin/delocate-wheel", "-w", repaired,
         wheel(wheels, "ctranslate2-4.8.3-*.whl")], "ct2-delocate",
        extra={"DYLD_LIBRARY_PATH": str(prefix / "lib")})

    run([python, "setup.py", "bdist_wheel", "--dist-dir", wheels], "torchaudio-build",
        work / "torchaudio-src", {"BUILD_VERSION": "2.11.0", "USE_CUDA": "0",
                                  "USE_ROCM": "0", "MAX_JOBS": str(args.jobs)})
    ort = work / "onnxruntime-src"
    run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1",
         "cmake/external/onnx"], "ort-submodules", ort)
    run([python, "tools/ci_build/build.py", "--config", "Release", "--update", "--build",
         "--build_wheel", "--skip_tests", "--skip_submodule_sync", "--compile_no_warning_as_error",
         "--apple_deploy_target", "14.0",
         "--parallel", args.jobs, "--cmake_generator", "Ninja", "--build_dir", work / "ort-build-isolated",
         "--cmake_extra_defines", "CMAKE_POLICY_VERSION_MINIMUM=3.5",
         "CMAKE_IGNORE_PREFIX_PATH=/opt/homebrew", "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER",
         f"FETCHCONTENT_SOURCE_DIR_PYBIND11_PROJECT={work / 'pybind11-src'}",
         "onnxruntime_BUILD_UNIT_TESTS=OFF"], "ort-build", ort)

    project = Path(tempfile.mkdtemp(prefix="project-", dir=work))
    for name in ("whisperx", "crates/whisperx_ext", "tests"):
        shutil.copytree(ROOT / name, project / name, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("target", "__pycache__", ".pytest_cache"))
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / name, project / name)
    run([work / "venv/bin/maturin", "build", "--release", "--interpreter", python, "--out", wheels],
        "rust-build", project, {"CARGO_TARGET_DIR": str(work / "rust-target")})
    run(uv + ["pip", "install", "--python", python,
              wheel(repaired, "ctranslate2-4.8.3-*.whl"), wheel(wheels, "torchaudio-2.11.0-*.whl"),
              wheel(work / "ort-build-isolated/Release/dist", "onnxruntime-*.whl"),
              wheel(wheels, "whisperx-*.whl")], "install")
    run(uv + ["pip", "check", "--python", python], "dependency-check")
    run([python, "-m", "pytest", "tests/", "-q"], "tests", project,
        {"MPLCONFIGDIR": str(work / "matplotlib")})
    print(f"Validated Python {PYTHON_VERSION} environment: {python}")


if __name__ == "__main__":
    main()
