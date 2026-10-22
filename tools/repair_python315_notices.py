#!/usr/bin/env python3
"""Include pinned licenses for numerical libraries bundled by auditwheel.

manylinux images may remove RPM documentation. This packaging step adds the
OpenBLAS and GNU license texts without changing any compiled library.
"""

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import zipfile


LICENSES = {
    "OpenBLAS-LICENSE": (
        "https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/v0.3.15/LICENSE",
        "190b5a9c8d9723fe958ad33916bd7346d96fab3c5ea90832bb02d854f620fcff"),
    "GCC-COPYING3": (
        "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-8.5.0/COPYING3",
        "8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903"),
    "GCC-COPYING.RUNTIME": (
        "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-8.5.0/COPYING.RUNTIME",
        "9d6b43ce4d8de0c878bf16b54d8e7a10d9bd42b75178153e3af6a815bdc90f74"),
    "GCC-COPYING.LIB": (
        "https://raw.githubusercontent.com/gcc-mirror/gcc/releases/gcc-8.5.0/COPYING.LIB",
        "a9bdde5616ecdd1e980b44f360600ee8783b1f99b8cc83a2beb163a0a390e861"),
}


def repair(manifest):
    manifest = Path(manifest)
    data = json.loads(manifest.read_text())
    wheel = manifest.parent / data["wheel"]
    assert data["package"] == "ctranslate2" and "manylinux" in wheel.name
    original_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert original_hash == data["sha256"]
    with zipfile.ZipFile(wheel) as archive:
        entries = {v.filename: (v, archive.read(v)) for v in archive.infolist() if not v.is_dir()}
    assert any("libopenblas-" in name and "3.15.so" in name for name in entries), "Review the OpenBLAS license for this library version"
    metadata = next(name for name in entries if name.endswith(".dist-info/METADATA"))
    info_dir = metadata.rsplit("/", 1)[0]
    if all(f"{info_dir}/licenses/{name}" in entries and
           hashlib.sha256(entries[f"{info_dir}/licenses/{name}"][1]).hexdigest() == digest
           for name, (_, digest) in LICENSES.items()):
        return
    for name, (url, digest) in LICENSES.items():
        with urlopen(url, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != digest:
            raise RuntimeError(f"Unexpected license checksum: {url}")
        path = f"{info_dir}/licenses/{name}"
        entries[path] = (zipfile.ZipInfo(path, (2026, 10, 22, 0, 0, 0)), content)
    info, content = entries[metadata]
    newline = b"\r\n" if b"\r\n" in content else b"\n"
    headers, separator, body = content.partition(newline * 2)
    if not separator:  # Header-only metadata has no description separator.
        headers, separator, body = content.rstrip(b"\r\n"), newline * 2, b""
    for name in LICENSES:
        field = f"License-File: {name}".encode()
        if field not in headers.splitlines():
            headers += newline + field
    entries[metadata] = (info, headers + separator + body)
    record = f"{info_dir}/RECORD"
    rows = []
    for name, (_, content) in sorted(entries.items()):
        if name != record:
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
            rows.append((name, "sha256=" + digest, len(content)))
    rows.append((record, "", ""))
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    entries[record] = (entries[record][0], buffer.getvalue().encode())
    temporary = wheel.with_suffix(".whl.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, (info, content) in entries.items():
            archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED)
    temporary.replace(wheel)
    data["original_wheel_sha256"] = original_hash
    data["sha256"] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    data["additional_license_sources"] = LICENSES
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    print("Included verified OpenBLAS and GNU licenses:", wheel.name)


if __name__ == "__main__":
    repair(sys.argv[1])
