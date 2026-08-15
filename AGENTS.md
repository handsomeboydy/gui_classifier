# AGENTS.md

## 项目定位

本项目是面向输电线路无人机巡检照片的 Windows 桌面分图工具。它读取线路杆塔经纬度台账和照片 EXIF 信息，按线路、杆塔与照片类别将文件复制到结构化输出目录，并提供手动飞行、自主飞行及若干扩展工具。

项目以 Python 和 Tkinter 实现，既可从源码运行，也可使用 PyInstaller 打包为单文件 EXE。

## 当前阶段目标

当前工作区属于交付/维护阶段。后续修改应优先保证现有分图规则、Windows GUI、单 EXE 多模式启动和升级链路不被破坏。

根目录是唯一权威源码，后续开发、测试和发布均以根目录文件为准。`repo_tmp/gui_classifier/` 只作为历史 Git 仓库、README 和构建资料参考，不得反向覆盖根目录。当前发布版本和根目录 `APP_VERSION` 均为 `1.0.1`。

## 首要阅读顺序

1. `AGENTS.md`
2. `repo_tmp/gui_classifier/README.md`
3. 根目录 `gui_classifier.py`
4. `multi_mode_launcher.py` 与 `extension.py`
5. `classify.py`、`classify_channels.py`、`classify_autonomous.py`
6. `side_parser.py`
7. `button1.py`、`button2.py`、`button3.py`
8. `updater_github.py`
9. `repo_tmp/gui_classifier/build_exe.bat` 与 `repo_tmp/gui_classifier/御3T分图工具.spec`

README 反映总体产品设计，但最终行为必须以准备修改的那份源码为准；文档和根目录代码目前存在少量版本差异。

## 项目结构速览

- `gui_classifier.py`：根目录 GUI 主入口，组合手动/自主分图、扩展工具和 GitHub 更新检查。
- `gui_classifier_updated.py`：历史 GUI 变体，目前不是默认入口或权威源码。
- `gui_classifier_pencil.py`：另一份 GUI 变体，目前不是默认入口。
- `classify.py`：手动飞行分类；按最近杆塔、距离阈值和可选侧别规则输出精细化/红外照片。
- `classify_channels.py`：自主飞行第一阶段；根据 EXIF 时间窗提取通道照片并生成 `skip_ir.txt`。
- `classify_autonomous.py`：自主飞行第二阶段；读取 `skip_ir.txt`，分类剩余精细化和红外照片。
- `side_parser.py`：解析整线或分段的左/右/单侧别规则。
- `multi_mode_launcher.py`：源码与冻结 EXE 共用的 `--tool=...` 子模式分发器。
- `extension.py`：扩展工具菜单。
- `button1.py`：提取单张照片坐标。
- `button2.py`：计算两张照片坐标中点。
- `button3.py`：激光雷达照片分图 GUI。
- `updater_github.py`：查询 GitHub Release、下载 EXE、生成批处理并替换当前程序。
- `gui_classifier_config.json`：运行时保存台账库路径；属于机器本地配置，不应作为通用默认值维护。
- `favicon2.ico`：打包图标。
- `使用录屏.mp4`：大体积使用演示文件，不属于源码或测试夹具。
- `repo_tmp/gui_classifier/`：带 `.git`、README、构建脚本和构建产物的历史仓库副本，仅供参考；大部分 Python 文件与根目录同名文件相同，但主 GUI 已有差异。
- `repo_tmp/gui_classifier/build/`、`dist/`：PyInstaller 产物，不应手工编辑。
- `__pycache__/`：解释器缓存，不应手工编辑或纳入功能改动。

## 核心业务流程

### GUI 入口

1. `gui_classifier.py` 在创建 Tk 窗口前调用 `dispatch_mode_or_continue()`。
2. 普通启动进入 `ClassifierGUI`；`--tool=extension|button1|button2|button3` 则由当前 Python/EXE 进程执行对应模块。
3. 用户选择台账库、一个或多个源照片目录、输出目录、线路名称和距离阈值。
4. GUI 以后台线程执行分类，并把标准输出/错误重定向到日志控件。

不要把工具模式分发移动到 Tk 主窗口创建之后。根目录主入口目前有两次分发调用；如要去重，必须同时验证源码启动和冻结 EXE 的所有 `--tool` 模式。

### 手动飞行

1. 从 `{线路名称}经纬度台账.xlsx` 读取 `杆塔编号`、`经度`、`纬度`。
2. 从照片 EXIF 读取 GPS，使用 Haversine 距离匹配最近杆塔。
3. 仅在距离阈值内分类；无 GPS、超阈值或侧别不符的照片被跳过。
4. 文件名包含 `_T` 时归入 `红外照片`，否则归入 `精细化`。
5. 为精细化照片对应杆塔创建空的 `通道` 目录。

### 自主飞行

1. 先运行 `classify_channels.classify_channels()`。
2. 根据 `_V_` 照片的 EXIF 拍摄时间形成杆塔间时间窗，将符合条件且名称不含 `_T` 的候选照片复制到 `通道/<杆塔号>/`。
3. 生成 `<输出根目录>/<线路名称>/skip_ir.txt`，记录与通道可见光照片对应、应跳过的红外文件名。
4. 再运行 `classify_autonomous.classify_autonomous()`，按 GPS、阈值、侧别和 `_V_`/`_T_` 命名分类其余照片。

不要交换这两个阶段的顺序。自主分类全部成功完成后，默认入口必须删除 `skip_ir.txt`；它只允许在两个阶段之间临时存在。

### 输出约定

主分图输出结构为：

```text
<output_root>/<line_name>/
├── 精细化/<tower>/
├── 红外照片/<tower>/
├── 通道/<tower>/
└── skip_ir.txt              # 自主模式的中间文件，成功完成后删除
```

现有主分类逻辑使用 `shutil.copy`，不移动源照片，但相同目标路径可能被覆盖。不要在真实巡检资料上进行开发试跑；应使用脱敏副本和独立临时输出目录。

## 输入数据与命名约定

- 主 GUI 按 `{线路名称}经纬度台账.xlsx` 拼接台账文件名。
- 基本必需列为 `杆塔编号`、`经度`、`纬度`。
- `classify_channels.py` 还要求 `线路名称` 列，并用它过滤数据。
- 杆塔编号在排序和侧别判断中会转换为整数；非纯数字编号当前可能报错，不要无验证地宣称支持字母或复合塔号。
- 自主模式通过 `_V_` 识别精细化照片、通过 `_T_` 识别红外照片；手动模式对红外使用较宽松的 `_T` 判断。
- 自主通道提取还依赖 EXIF 拍摄时间；缺少 GPS 或时间的文件不能完整参与该流程。
- 手动模式只查找同目录下固定名称 `1双回塔台账文件.xlsx`；自主模式查找首个 `*双回塔台账文件.xlsx`。两者行为不同，调整时必须补回归验证。
- 双回塔台账使用 `杆塔1名称`、`杆塔2名称`、`杆塔1方位`、`杆塔2方位`，方位可为整线 `左/右` 或类似 `1-15:左,16-20:单,21-25:右` 的分段规则。

## 技术栈与依赖

- Python 3.8+（项目文档声明；实际发布环境版本待确认）
- Tkinter：桌面 GUI
- `pandas`、`openpyxl`：Excel 台账读取
- `exifread`：主分类 EXIF 解析
- `Pillow`：`button1.py`、`button2.py` 的图片 EXIF 解析
- PyInstaller：Windows 单文件 EXE 打包

仓库没有 `requirements.txt` 或 `pyproject.toml`。基础依赖可安装为：

```powershell
python -m pip install pandas exifread openpyxl pillow
```

不要仅为文档或静态检查自动安装依赖；需要安装或升级时先确认目标 Python 环境。

## 开发约定

- 所有 Python 与 Markdown 文本保持 UTF-8；注意 Windows PowerShell 旧版本读取 UTF-8 时可能显示乱码，不能据此改写文件编码。
- 只把根目录作为权威源码修改。`repo_tmp/gui_classifier/` 不自动同步；若用户明确要求更新历史仓库副本，先比较差异，再有选择地同步，绝不反向覆盖根目录。
- 不要编辑 `build/`、`dist/`、`.git/`、`__pycache__/` 或录屏文件来实现源码功能。
- 保持 Windows 中文、空格和括号路径兼容，所有外部路径在命令与批处理中应正确加引号。
- 保持分类函数可被 GUI 和命令行共同调用；不要把核心业务逻辑塞入 Tk 回调。
- 修改 EXIF、距离、塔序、侧别、命名判断或输出目录规则时，必须覆盖手动和自主两条链路，不能只验 GUI 是否能启动。
- 不要吞掉新增核心逻辑的异常。现有宽泛 `except` 可逐步收紧，但需保持 GUI 对无 GPS、坏图片和坏台账的友好提示。
- 更新 `APP_VERSION`、GitHub Release tag 和发布资产名时必须保持一致。
- 不要顺手提交运行时配置、真实台账、真实照片、临时输出或构建产物。

## 测试与验收方式

项目当前没有自动化测试目录。至少执行以下分层验证。

### 静态语法检查

在根目录执行，不导入第三方模块：

```powershell
python -c "import ast,pathlib; [ast.parse(p.read_text(encoding='utf-8')) for p in pathlib.Path('.').glob('*.py')]; print('syntax ok')"
```

### 纯逻辑检查

至少验证侧别解析：

```powershell
python -c "from side_parser import get_expected_side_for_tower as g; assert g('1-15:左,16-20:单,21-25:右', 3)=='左'; assert g('1-15:左,16-20:单,21-25:右', 18) is None; assert g('右', 8)=='右'; print('side parser ok')"
```

### GUI 与业务烟雾测试

- 用脱敏的小型 `.xlsx`、少量带 EXIF 的样例照片和独立临时输出目录运行 `python gui_classifier.py`。
- 分别验证手动模式、自主模式和多个源目录。
- 验证无 GPS、超阈值、缺列、非数字阈值、侧别不符与重复文件名的行为。
- 验证 `--tool=extension` 以及三个扩展工具在源码模式和打包 EXE 中均能启动。
- 自主模式需核对 `通道`、`精细化`、`红外照片` 的实际文件集合，并确认成功完成后不存在 `skip_ir.txt`。
- GUI 启动会触发 GitHub 静默更新检查；离线环境下应确认失败不会影响分类主流程。

### 打包验收

构建脚本位于 `repo_tmp/gui_classifier/build_exe.bat`。该脚本会尝试安装 PyInstaller，并删除仓库副本下已有的 `build/` 和 `dist/` 后重建；未明确要求打包时不要运行。

发布验收至少包括：EXE 启动、中文路径、图标、手动/自主分图、扩展工具子进程、检查更新、旧版本备份与新版本重启。

## 配置与敏感信息规则

- 台账包含线路和杆塔坐标，照片包含 GPS 与拍摄时间，均按敏感生产资料处理。
- `gui_classifier_config.json` 与可能生成的 `button3_config.json` 会记录本机台账路径；不要把真实路径固化为共享默认配置。
- 不要在代码、AGENTS、README、测试夹具或日志中写入真实 Token、Authorization、私有仓库凭据、人员信息、生产路径或未经脱敏的线路数据。
- 测试数据应脱敏、最小化，并与真实输出目录隔离。
- `updater_github.py` 会访问公开 GitHub API 并下载 Release 资产。当前更新调用未提供强制 SHA-256 值，修改或发布升级链路时必须审查资产来源、版本比较、完整性校验和回滚行为。
- 更新脚本会生成批处理，备份并覆盖当前 EXE；不要用开发目录中的重要文件冒充更新目标做测试。

## 常见风险与避坑经验

- 根目录与 `repo_tmp/gui_classifier/` 是两套近似代码；根目录是权威源码，历史仓库副本不得用于反向同步。
- README 的描述可能落后于当前源码；涉及删除、跳过或覆盖文件的行为必须直接核对代码并用样例验证。
- 手动与自主模式对红外文件名、双回塔台账文件名和侧别算法的实现并不完全一致。
- `glob('*.*')` 会遍历源目录中的各种文件；非图片或损坏文件可能触发 EXIF 解析异常。扩展支持格式时不要只改文件选择器。
- 分类按整数排序塔号；台账中的空值、重复塔号、非数字塔号或非数值坐标需要显式测试。
- `skip_ir.txt` 是两个自主阶段之间的接口，不是普通日志；不能在第二阶段读取前删除。
- GUI 在后台线程执行任务并重定向全局 `sys.stdout/sys.stderr`；修改并发或日志逻辑时注意多个任务、Tk 线程安全和异常后的恢复。
- 自动更新会覆盖正在运行的 EXE，并依赖 Release 资产名称匹配；版本号或资产名不一致会造成假更新或无法更新。
- 大型录屏、Git 对象和构建产物会显著增加扫描与打包成本，日常搜索应排除这些目录和文件。

## 文档维护规则

- 只有项目定位、权威入口、稳定架构、数据契约、测试命令、发布流程或反复验证的风险发生长期变化时，才更新 `AGENTS.md`。
- 单次修复过程、临时 TODO、报错堆栈、构建日志和一次性交接信息不要写入本文件。
- 功能说明变化同步更新 README；版本变化写入正式发布说明或 CHANGELOG（如后续建立），不要把 `AGENTS.md` 当作开发流水账。
- 权威源码树或 `skip_ir.txt` 生命周期发生变化时，必须同步更新本文件；版本变化应同时更新发布说明。

## 已确认的稳定结论与待确认项

已确认：

- 默认源码入口名为 `gui_classifier.py`，核心分类模块同时支持函数调用和命令行调用。
- 根目录是唯一权威源码，`repo_tmp/gui_classifier/` 只供历史和构建资料参考。
- 当前发布版本为 `1.0.1`，根目录 `APP_VERSION` 必须与对应 Release tag 保持一致。
- 主流程复制照片到输出目录，不主动移动源照片。
- 自主模式必须先提取通道，再分类精细化/红外。
- 自主分类成功完成后必须删除 `skip_ir.txt`。
- 项目面向 Windows，并维护 PyInstaller 单 EXE 与子工具模式。
- 当前没有自动化测试套件或标准依赖清单。

待确认：

- `gui_classifier_updated.py`、`gui_classifier_pencil.py` 是否仍需长期保留，或应在验证后归档。
