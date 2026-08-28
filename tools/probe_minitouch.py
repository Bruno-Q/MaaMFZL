#!/usr/bin/env python3
"""Probe ADB touch input with MaaFramework forced to MinitouchAndAdbKey."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Device:
    name: str
    adb_path: Path
    address: str
    screencap_methods: int
    config: dict[str, Any]


def _bootstrap_import_path() -> None:
    site_packages = sorted((ROOT / ".venv" / "lib").glob("python*/site-packages"))
    for path in site_packages:
        sys.path.insert(0, str(path))


def _find_by_maafw(specified_adb: str | None) -> list[Device]:
    from maa.toolkit import Toolkit

    devices = []
    for dev in Toolkit.find_adb_devices(specified_adb):
        devices.append(
            Device(
                name=dev.name,
                adb_path=dev.adb_path,
                address=dev.address,
                screencap_methods=int(dev.screencap_methods),
                config=dev.config,
            )
        )
    return devices


def _find_by_adb(adb_path: str | None) -> list[Device]:
    from maa.controller import MaaAdbScreencapMethodEnum

    resolved_adb = adb_path or shutil.which("adb")
    if not resolved_adb:
        return []

    try:
        output = subprocess.check_output(
            [resolved_adb, "devices"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []

    devices: list[Device] = []
    for line in output.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 2 or parts[1] != "device":
            continue
        serial = parts[0]
        devices.append(
            Device(
                name=serial,
                adb_path=Path(resolved_adb),
                address=serial,
                screencap_methods=int(MaaAdbScreencapMethodEnum.Default),
                config={},
            )
        )
    return devices


def _select_device(devices: list[Device], index: int, address: str | None) -> Device:
    if address:
        for dev in devices:
            if dev.address == address or dev.name == address:
                return dev
        raise SystemExit(f"未找到指定设备: {address}")

    if not devices:
        raise SystemExit("未发现 ADB 设备。请先确认 `adb devices` 能看到真机。")

    if index < 0 or index >= len(devices):
        raise SystemExit(f"设备序号超出范围: {index}，当前发现 {len(devices)} 台")
    return devices[index]


def _print_devices(devices: list[Device]) -> None:
    print("发现设备:")
    for i, dev in enumerate(devices):
        print(
            f"  [{i}] name={dev.name} address={dev.address} "
            f"adb={dev.adb_path} screencap_methods={dev.screencap_methods}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Force MaaFramework ADB input to MinitouchAndAdbKey and send a probe click."
    )
    parser.add_argument("--adb", help="指定 adb 路径，默认使用 MaaToolkit/系统 PATH")
    parser.add_argument("--address", help="指定设备 serial/address，默认取第一台")
    parser.add_argument("--index", type=int, default=0, help="设备序号，默认 0")
    parser.add_argument("--x", type=int, default=360, help="点击 x 坐标，默认 360")
    parser.add_argument("--y", type=int, default=360, help="点击 y 坐标，默认 360")
    parser.add_argument(
        "--short-side",
        type=int,
        default=720,
        help="截图短边缩放，默认 720，保持和 interface.json 一致",
    )
    args = parser.parse_args()

    _bootstrap_import_path()

    from maa.controller import AdbController, MaaAdbInputMethodEnum
    from maa.tasker import Tasker, LoggingLevelEnum

    Tasker.set_log_dir(ROOT / "debug")
    Tasker.set_stdout_level(LoggingLevelEnum.All)

    devices = _find_by_maafw(args.adb)
    if not devices:
        print("MaaToolkit 未发现设备，回退到 `adb devices`。")
        devices = _find_by_adb(args.adb)

    _print_devices(devices)
    dev = _select_device(devices, args.index, args.address)

    print(f"使用设备: {dev.name} ({dev.address})")
    print("强制输入方式: MinitouchAndAdbKey = 2")

    ctrl = AdbController(
        dev.adb_path,
        dev.address,
        dev.screencap_methods,
        int(MaaAdbInputMethodEnum.MinitouchAndAdbKey),
        dev.config,
    )
    if args.short_side > 0:
        print(f"设置截图短边: {args.short_side}")
        ctrl.set_screenshot_target_short_side(args.short_side)

    conn = ctrl.post_connection().wait()
    print(f"连接状态: {conn.status}")
    if not conn.succeeded:
        return 2

    image = ctrl.post_screencap().wait().get()
    if image is None:
        print("截图失败：连接成功但无法取图。")
        return 3
    print(f"截图成功: shape={getattr(image, 'shape', None)}")
    print(f"原始分辨率: {ctrl.get_resolution()}")

    click = ctrl.post_click(args.x, args.y).wait()
    print(f"点击 ({args.x}, {args.y}) 状态: {click.status}")
    if not click.succeeded:
        return 4

    print("探针点击已发送。请观察真机对应位置是否有反应。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
