#!/usr/bin/env python3
"""Build missing CPython 3.15 wheels from unmodified, pinned upstream sources.

Linux builds run in manylinux_2_28; CTranslate2 additionally needs OpenBLAS
and the CUDA 12.8/cuDNN 9 development packages. macOS targets Apple silicon 14+.
Each invocation builds one package so CI can isolate dependencies and artifacts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from build_python315_macos import SOURCES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", choices=("ctranslate2", "onnxruntime", "torchaudio"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    mac = sys.platform == "darwin"
    if sys.version_info[:2] != (3, 15) or args.jobs < 1:
        parser.error("Use CPython 3.15 and a positive job count")
    if (mac and platform.machine() != "arm64") or (not mac and
            (sys.platform != "linux" or platform.machine() != "x86_64")):
        parser.error("Supported targets: macOS arm64 and Linux x86_64")
    work = args.work_dir.resolve()
    output = args.output_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    raw = work / "raw-wheels"
    raw.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(MAX_JOBS=str(args.jobs), CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs),
               MACOSX_DEPLOYMENT_TARGET="14.0")
    if mac:
        env["ARCHFLAGS"] = "-arch arm64"
        # setup-python uses a universal2 interpreter; the linked PyTorch and
        # CTranslate2 libraries are arm64, so the wheel must not claim x86_64.
        env["_PYTHON_HOST_PLATFORM"] = "macosx-14.0-arm64"

    def run(command, cwd=work, extra=None):
        print("+", " ".join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), cwd=cwd, env=env | (extra or {}), check=True)

    def source(name):
        repository, tag, commit = SOURCES[name]
        path = work / (name + "-src")
        if not path.exists():
            run(["git", "clone", "--depth", "1", "--branch", tag,
                 f"https://github.com/{repository}.git", path])
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        if actual != commit:
            raise RuntimeError(f"Unexpected {name} revision: {actual}")
        run(["git", "diff", "--quiet", "HEAD"], path)
        return path

    def only_wheel(directory):
        wheels = list(directory.glob("*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"Expected one wheel in {directory}: {wheels}")
        return wheels[0]

    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip>=26.2.2",
         "certifi>=2026.10.19", "setuptools==84.2.1", "wheel", "cmake>=3.30,<4",
         "ninja", "numpy==2.5.4", "pybind11==3.1.0", "packaging", "pyyaml",
         "delocate" if mac else "auditwheel"])
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
    src = source(args.package)
    if args.package == "ctranslate2":
        submodules = ["third_party/spdlog", "third_party/ruy", "third_party/cpu_features"]
        if not mac:
            submodules += ["third_party/thrust", "third_party/cutlass"]
        run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1",
             *submodules], src)
        prefix = work / "prefix"
        # setup.py lives under python/, outside the upstream LICENSE file.
        # Include the upstream and bundled third-party notices in distributions.
        shutil.copy2(src / "LICENSE", src / "python/LICENSE")
        notices = []
        for directory in [src / name for name in submodules] + [Path("/usr/share/licenses/openblas"),
                         Path("/usr/share/licenses/libgfortran"), Path("/usr/local/cuda-12.8")]:
            for pattern in ("LICENSE*", "COPYING*", "NOTICE*", "EULA*"):
                for license_file in sorted(directory.glob(pattern)):
                    if license_file.is_file():
                        notices.append(f"\n--- {directory.name}/{license_file.name} ---\n" +
                                       license_file.read_text(errors="replace"))
        (src / "python/NOTICE").write_text("Third-party dependency licenses\n" + "\n".join(notices))
        options = ["-DWITH_MKL=OFF", "-DWITH_DNNL=OFF", "-DWITH_RUY=ON",
                   "-DBUILD_CLI=OFF", "-DBUILD_TESTS=OFF"]
        if mac:
            options += ["-DWITH_ACCELERATE=ON", "-DOPENMP_RUNTIME=NONE"]
        else:
            options += ["-DWITH_OPENBLAS=ON", "-DOPENBLAS_INCLUDE_DIR=/usr/include/openblas",
                        "-DOPENMP_RUNTIME=COMP",
                        "-DWITH_CUDA=ON", "-DWITH_CUDNN=OFF", "-DCUDA_DYNAMIC_LOADING=ON",
                        "-DCUDA_ARCH_LIST=Common", "-DCUDA_NVCC_FLAGS=-Xfatbin=-compress-all"]
        run(["cmake", "-S", src, "-B", work / "build", "-DCMAKE_BUILD_TYPE=Release",
             "-DCMAKE_POLICY_VERSION_MINIMUM=3.5", f"-DCMAKE_INSTALL_PREFIX={prefix}", *options])
        run(["cmake", "--build", work / "build", "--parallel", args.jobs])
        run(["cmake", "--install", work / "build"])
        run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", raw], src / "python",
            {"CTRANSLATE2_ROOT": str(prefix)})
        env["DYLD_LIBRARY_PATH" if mac else "LD_LIBRARY_PATH"] = str(prefix / "lib")
        built = only_wheel(raw)
    elif args.package == "torchaudio":
        # PyTorch's official index, rather than PyPI, carries the stable cp315 wheels.
        run([sys.executable, "-m", "pip", "install", "torch==2.14.1",
             "--index-url", "https://download.pytorch.org/whl/cpu"])
        run([sys.executable, "setup.py", "bdist_wheel", "--dist-dir", raw], src,
            {"BUILD_VERSION": "2.11.0", "USE_CUDA": "0", "USE_ROCM": "0"})
        built = only_wheel(raw)
        if mac:
            torch_lib = subprocess.check_output(
                [sys.executable, "-c", "import torch,pathlib; print(pathlib.Path(torch.__file__).parent/'lib')"],
                text=True).strip()
            env["DYLD_LIBRARY_PATH"] = torch_lib
    else:
        pybind = source("pybind11")
        run([sys.executable, "-m", "pip", "install", "flatbuffers", "protobuf>=4.25.8",
             "psutil", "jinja2"])
        run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1",
             "cmake/external/onnx"], src)
        options = ["--apple_deploy_target", "14.0"] if mac else ["--allow_running_as_root"]
        run([sys.executable, "tools/ci_build/build.py", "--config", "Release", "--update",
             "--build", "--build_wheel", "--skip_tests", "--skip_submodule_sync",
             "--compile_no_warning_as_error", "--parallel", args.jobs,
             "--cmake_generator", "Ninja", "--build_dir", work / "build", *options,
             "--cmake_extra_defines", "CMAKE_POLICY_VERSION_MINIMUM=3.5",
             "CMAKE_IGNORE_PREFIX_PATH=/opt/homebrew", "FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER",
             f"FETCHCONTENT_SOURCE_DIR_PYBIND11_PROJECT={pybind}",
             "onnxruntime_BUILD_UNIT_TESTS=OFF"], src)
        built = only_wheel(work / "build/Release/dist")

    if mac:
        exclusions = ["--exclude", "libtorch", "--exclude", "libc10"] if args.package == "torchaudio" else []
        run([sys.executable, "-m", "delocate.cmd.delocate_wheel", "--require-archs", "arm64",
             *exclusions, "-w", output, built])
    elif args.package == "torchaudio":
        # TorchAudio deliberately resolves libtorch libraries from the installed
        # torch package. Do not copy hundreds of MB of libtorch into this wheel.
        run(["auditwheel", "repair", "--plat", "manylinux_2_28_x86_64", "-w", output,
             "--exclude", "libc10.so", "--exclude", "libtorch.so",
             "--exclude", "libtorch_cpu.so", "--exclude", "libtorch_python.so", built])
    else:
        run(["auditwheel", "repair", "--plat", "manylinux_2_28_x86_64", "-w", output, built])

    wheel = only_wheel(output)
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", wheel])
    if args.package == "onnxruntime":
        run([sys.executable, "-c", "import onnxruntime as o; assert 'CPUExecutionProvider' in o.get_available_providers(); print(o.__version__)"])
    else:
        run([sys.executable, "-c", f"import {args.package} as p; print(p.__version__)"])
    manifest = {"package": args.package, "python": platform.python_version(),
                "platform": platform.platform(), "sources": SOURCES,
                "wheel": wheel.name, "sha256": hashlib.file_digest(wheel.open("rb"), "sha256").hexdigest(),
                "cuda": "12.8, dynamic cuBLAS, native convolution (runtime test required)" if not mac and args.package == "ctranslate2" else None}
    (output / f"{args.package}-build.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
