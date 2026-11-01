"""Collect only values read from actual ESP-IDF build artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(build, output):
    build = Path(build).resolve()
    description = json.loads((build / "project_description.json").read_text(encoding="utf-8"))
    source = Path(description["project_path"])
    dependencies = yaml.safe_load((source / "dependencies.lock").read_text(encoding="utf-8"))["dependencies"]
    image = build / Path(description["app_elf"]).with_suffix(".bin").name
    elf = build / Path(description["app_elf"]).name
    config = Path(description["config_file"])
    settings = config.read_text(encoding="utf-8")
    camera_mode = "CONFIG_ATV_CAMERA_MODE=y" in settings
    sources = [source / "main" / ("camera_main.cpp" if camera_mode else "app_main.cpp")]
    if camera_mode:
        sources.extend(source / "main" / name for name in ("camera_core.hpp", "camera_model_config.h"))
    if any(p.stat().st_mtime_ns > image.stat().st_mtime_ns for p in sources):
        raise ValueError("Build sources changed after the image; run build before collecting evidence")
    flash_args = json.loads((build / "flasher_args.json").read_text(encoding="utf-8"))
    payloads = [
        {"offset": offset, "bytes": (build / filename).stat().st_size, "sha256": sha(build / filename)}
        for offset, filename in flash_args["flash_files"].items()
    ]
    report = {
        "compiled": True,
        "mode": "camera" if "CONFIG_ATV_CAMERA_MODE=y" in settings else "fixture",
        "transport": "USB Serial/JTAG" if "CONFIG_ATV_TRANSPORT_USB_JTAG=y" in settings else "UART0 bridge",
        "target": description["target"],
        "idf": dependencies["idf"]["version"],
        "esp_dl": dependencies["espressif/esp-dl"]["version"],
        "project_version": description["project_version"],
        "firmware_bytes": image.stat().st_size,
        "firmware_bin_sha256": sha(image),
        "firmware_elf_sha256": sha(elf),
        "sdkconfig_sha256": sha(config),
        "model_bytes": (source / "main/tinyvision.espdl").stat().st_size,
        "model_sha256": sha(source / "main/tinyvision.espdl"),
        "configured_flash_size": flash_args["flash_settings"]["flash_size"],
        "flash_payloads": payloads,
        "compiled_flash_payload_bytes": sum(item["bytes"] for item in payloads),
        "hardware_status": "PENDING",
        "build_directory": str(build),
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    print(json.dumps(collect(a.build, a.output), indent=2))
