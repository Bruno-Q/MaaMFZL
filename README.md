<!-- markdownlint-disable MD033 MD041 -->
<p align="center">
  <img alt="Maa logo" src="https://cdn.jsdelivr.net/gh/MaaAssistantArknights/design@main/logo/maa-logo_512x512.png" width="192" height="192" />
</p>

<div align="center">

# MaaMFZL

魔法之路小助手 —— 基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 的自动化工具。

</div>

## 功能

- **刷血缘**：自动选择关卡并进行战斗，可按装备稀有度和战斗次数筛选。
- **刷无尽**：按屏蔽词条自动刷新无尽词条，战斗失败后会尝试恢复并继续刷新。
- **刷幸运星**：自动进入活动并循环领取幸运星。
- **统一入口**：客户端任务列表还保留了“听说这里有怪兽”入口，实际任务以 `assets/interface.json` 中的配置为准。

## 使用前准备

获取代码时建议同时拉取公共资源子模块：

```bash
git clone --recurse-submodules https://github.com/Bruno-Q/MaaMFZL.git
cd MaaMFZL
```

1. 准备 Python 3.8 或更高版本，并确保已安装并配置好 ADB。
2. 使用支持 MaaFramework 的客户端（例如 MFAAvalonia 或 MaaPiCli），并准备一个已连接的 Android 设备或模拟器。
3. 下载与当前版本匹配的 [MaaFramework Release](https://github.com/MaaXYZ/MaaFramework/releases)，解压到项目根目录的 `deps` 文件夹。解压后至少应包含：

    ```text
    deps/
    ├── bin/
    └── share/MaaAgentBinary/
    ```

4. 初始化公共资源子模块并配置 OCR：

    ```bash
    git submodule update --init --recursive
    python3 -m venv .venv       # Windows：py -3 -m venv .venv
    source .venv/bin/activate       # Windows：.venv\Scripts\activate
    python -m pip install --upgrade pip
    python -m pip install -r tools/requirements.txt
    python -m pip install "MaaFw>=5.9.0" numpy opencv-python
    python tools/configure.py
    ```

    `tools/configure.py` 会将 OCR 模型复制到 `assets/resource/model/ocr/`。该目录已被 Git 忽略，不需要提交。

## 配置与运行

启动前，请打开 [`assets/interface.json`](./assets/interface.json)，将 `agent.child_exec` 和 `agent.child_args` 修改为本机的 Python 与 `agent/main.py` 路径。示例：

```jsonc
"agent": {
    "child_exec": "/项目路径/.venv/bin/python",
    "child_args": ["/项目路径/agent/main.py"]
}
```

Windows 请使用本机路径格式，并按 JSON 要求转义反斜杠。然后在 MFAAvalonia 或 MaaPiCli 中加载项目根目录（或打包后的 `install` 目录），选择对应任务运行。

### 任务选项

| 任务 | 选项 | 说明 |
| --- | --- | --- |
| 刷血缘 | 血缘关卡 | 例如 `地狱162` |
| 刷血缘 | 血缘装备掉落标签 | `0` 金、`1` 远古、`2` 紫、`3` 太古，可用逗号分隔；默认 `3` |
| 刷血缘 | 血缘战斗次数 | `0` 表示一直刷到钥匙不足 |
| 刷无尽 | 无尽屏蔽词条 | 用逗号分隔需要避开的词条 |
| 刷无尽 | 无尽战斗次数 | `0` 表示一直刷到体力不足 |
| 刷幸运星 | 无 | 进入活动后自动循环 |
| 听说这里有怪兽 | 主界面 | 入口任务，具体行为以当前资源配置为准 |

## 开发与检查

资源文件位于 `assets/resource/`，自定义识别和动作位于 `agent/`。修改资源后可以运行以下检查：

```bash
python check_resource.py assets/resource/
python tools/validate_schema.py \
  --schema-dir deps/tools \
  --resource-dirs assets/resource \
  --exclude-dirs assets/resource/announcement \
  --interface-files assets/interface.json
```

运行日志和识别调试图片写入 `debug/`。需要排查 YOLO 识别时，可在相关动作中启用调试截图；请注意 `debug/` 不应提交到仓库。

## 打包发布

GitHub Actions 会在推送 `v*` 标签时生成各平台安装包。也可以在本地执行：

```bash
python tools/install.py v1.0.0 macos aarch64
```

其中平台参数支持 `win`、`macos`、`linux`、`android`，架构参数支持 `x86_64` 和 `aarch64`（Android 使用对应的 MaaFramework 资源）。生成结果位于 `install/`。

## 问题反馈

提交 Issue 时请附上：

- MaaMFZL 版本、操作系统、模拟器及分辨率；
- `debug/` 中与问题时间对应的日志；
- 能复现问题的截图或录屏；
- 使用的任务及任务选项。

## 许可证

本项目使用 [MIT License](./LICENSE)。
