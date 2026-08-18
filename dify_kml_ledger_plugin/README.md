# KML 经纬度台账 Dify 工具插件

该插件将一个 `.kml` 文件转换为分图软件可使用的 `.xlsx` 经纬度台账。

## 当前状态

这是 V1 可打包插件，支持单回输出一本台账、`甲乙`双回输出两本台账，并支持异常文件名的手动覆盖参数。正式安装前应在目标 Dify 环境完成：

1. 安装 Dify Plugin CLI 和 Python 3.12。
2. 安装 `requirements.txt`。
3. 用脱敏 KML 执行远程调试。
4. 通过 `dify plugin package ./dify_kml_ledger_plugin` 打包。
5. 在 Dify 的插件页面安装生成的 `.difypkg`。
6. 在工作流中选择 `KML 转经纬度台账` 工具，并导出一次实际 Workflow YAML，取得真实 `plugin_unique_identifier`。

## Windows 安装 CLI

PowerShell 中提示“找不到 `dify` 命令”时，说明 Dify Plugin CLI 尚未安装或没有加入 PATH。请从 [Dify Plugin Daemon Releases](https://github.com/langgenius/dify-plugin-daemon/releases) 下载 `dify-plugin-windows-amd64.exe`（普通 Intel/AMD Windows）或 `dify-plugin-windows-arm64.exe`（Windows on ARM），重命名为 `dify.exe`，放入例如 `C:\Tools\dify` 的目录，并将该目录加入用户 PATH。

重新打开 PowerShell 后验证：

```powershell
dify version
```

在项目根目录打包：

```powershell
Set-Location 'E:\Sep（最终提交版）'
dify plugin package '.\dify_kml_ledger_plugin'
```

如果暂时不想改 PATH，也可以使用完整路径直接运行：

```powershell
& 'C:\Tools\dify\dify.exe' version
```

## 名称和输出

标准文件名例如 `220kV砚利甲乙线N1-N120.kml` 会输出：

```text
砚利甲线经纬度台账.xlsx
砚利乙线经纬度台账.xlsx
```

每本 XLSX 的工作表为 `经纬度台账`，前四列保持分图兼容，后面附带线路元数据：

```text
线路名称 | 杆塔编号 | 经度 | 纬度 | 线路全称 | 电压等级(kV) | 线路类型
```

工作流还返回 `manual_required`、`manual_reason`、`file_names`、`line_names` 等变量。桌面端收到 `manual_required` 后会弹窗收集电压等级、线路类型和线路名称，再复用已上传文件重新执行。

## 注意

- 缺失塔号首版按顺序编号并返回警告；重复塔号和非法坐标直接失败。
- 不支持 KMZ 和同一文件混合多条无关线路；甲乙双回文件需要符合标准命名，异常格式通过桌面端手动补录。
- 不要把真实 API Key、生产 KML 或真实台账提交到项目目录。
- 当前插件内含一份独立转换实现；后续应发布共享依赖包，避免桌面端与插件逻辑漂移。
