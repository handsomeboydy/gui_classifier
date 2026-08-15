# 开发任务卡：任务 A 源码、依赖与构建基线

## 一、背景

- 来源：`PRD.md` 13.1 任务 A；关键决策 D-01、D-08、D-09、D-10 已确认（`DECISIONS.md`，2026-08-15）。
- 根目录是唯一权威源码，但根目录 `.git` 是空壳；真实仓库位于 `repo_tmp/gui_classifier/.git`（远程 `handsomeboydy/gui_classifier`，分支 `main`）。
- 当前解释器为 Python 3.14.3，缺少 `exifread`；项目没有依赖清单。
- 构建脚本位于历史仓库目录，产物名、脚本提示和更新器资产匹配规则不一致。
- 主入口 `gui_classifier.py` 调用了两次 `dispatch_mode_or_continue()`。

## 二、目标

- 根目录成为可正常工作的 Git 仓库，保留历史提交。
- 建立运行/开发依赖清单，固定正式打包环境为 Python 3.12。
- 构建入口迁移到根目录，从权威源码产出“御3T分图工具.exe”。
- 主 GUI 与四个 `--tool` 模式在源码和（可行时）冻结 EXE 下均可启动。
- 更新器资产匹配规则与构建产物名保持一致。

## 三、涉及文件

- 根目录 `.git`：空壳，需要恢复。
- `repo_tmp/gui_classifier/.git`：历史仓库，只复制，不移动或删除。
- `gui_classifier.py`：主入口，去重两次分发调用。
- `multi_mode_launcher.py`：阅读验证，不修改（除非发现必要问题）。
- `updater_github.py`：仅修改 `ASSET_NAME_PATTERN`。
- `repo_tmp/gui_classifier/build_exe.bat`、`repo_tmp/gui_classifier/御3T分图工具.spec`：迁移到根目录并调整。
- 新增：`requirements.txt`、`requirements-dev.txt`、`.gitignore`、`PYTHON_VERSION.md`。
- 可能涉及：`extension.py`、`button1.py`、`button2.py`、`button3.py`（仅验证，不改）。

## 四、当前问题

- 根目录执行 `git status` 报 “not a git repository”。
- 历史仓库因沙箱用户与文件属主不一致报 “dubious ownership”。
- 构建脚本位于历史副本，可能打包历史 GUI；脚本提示输出 `dist\gui_classifier.exe`，实际产物名为“御3T分图工具.exe”。
- 更新器只匹配 `^mavic3T\.exe$`，与构建产物名不一致，会导致“检查更新”找不到资产。
- `gui_classifier.py` 中 `dispatch_mode_or_continue()` 被调用两次。
- 无依赖清单；当前环境缺 `exifread`；项目声明 Python 3.8+ 但未固定构建版本。

## 五、修改要求

1. Git 基线：
   - 将 `repo_tmp/gui_classifier/.git` 复制（`Copy-Item -Recurse`）到根目录，原目录保留作为备份。
   - 处理文件归属问题：`git config --global --add safe.directory 'E:/Sep（最终提交版）'`；如权限不足，说明原因并请求批准。
   - 审查 `git status` 与 `git log`，确认历史可见。
   - 新增 `.gitignore`，至少忽略：`build/`、`dist/`、`__pycache__/`、`*.pyc`、`gui_classifier_config.json`、`button3_config.json`、`使用录屏.mp4`、`repo_tmp/`。
   - 建立基线提交，记录根目录权威源码、PRD、DECISIONS 和任务卡；提交信息说明 “Task A baseline”。
   - 禁止使用 `git checkout --`、`git reset --hard` 等破坏性命令。
2. 依赖基线：
   - `requirements.txt`：`pandas`、`exifread`、`openpyxl`、`pillow`；安装成功后锁定实际版本。
   - `requirements-dev.txt`：`-r requirements.txt` 加 `pyinstaller`。
   - `PYTHON_VERSION.md`：明确正式打包环境为 Python 3.12（D-09）；若本机只能用 3.14 验证，必须记录差异。
   - 安装缺失依赖以完成验证；网络受限时说明原因并请求批准。
3. 构建入口：
   - 将 `build_exe.bat` 与 spec 迁移到根目录，内容改为针对根目录源码。
   - 修正输出提示与产物名一致：`dist\御3T分图工具.exe`。
   - 构建脚本不得修改 `repo_tmp` 目录；除非任务明确要求打包验证，不自动运行构建。
4. 主入口：
   - 将 `gui_classifier.py` 中两次 `dispatch_mode_or_continue()` 去重为一次，且保持在创建 Tk 主窗口之前。
   - 必须同时验证源码启动与（构建后）冻结 EXE 的所有 `--tool` 模式。
5. 更新器命名：
   - 将 `updater_github.py` 的 `ASSET_NAME_PATTERN` 改为匹配“御3T分图工具.exe”（D-08）。
   - 不修改 SHA-256 强制校验等其他更新逻辑（属任务 E）。

## 六、非目标范围

- 本轮不做：改变分类业务规则、输出目录层级、预检、结果清单、仅分析模式（任务 C/D）。
- 本轮不做：SHA-256 强制校验与更新回滚闭环（任务 E）。
- 本轮不修改：`repo_tmp` 内容，不反向覆盖根目录，不删除历史仓库副本。
- 本轮不提交：真实台账、照片、本地配置路径。
- 本轮不处理：`gui_classifier_updated.py`、`gui_classifier_pencil.py` 的归档决策。
- 本轮不得影响：现有手动/自主分类结果、四个扩展工具模式、GitHub 静默更新检查的失败降级。

## 七、验收标准

- [ ] 根目录 `git status` 与 `git log` 可正常执行，历史提交可见。
- [ ] `.gitignore` 生效：`build/`、`__pycache__/`、`gui_classifier_config.json`、`使用录屏.mp4`、`repo_tmp/` 被忽略，`gui_classifier.py` 未被忽略。
- [ ] `requirements.txt` 与 `requirements-dev.txt` 存在，并能从干净环境安装。
- [ ] `PYTHON_VERSION.md` 明确正式打包环境为 Python 3.12；若验证环境不同，已记录差异。
- [ ] 根目录存在构建入口，入口使用根目录 `gui_classifier.py`，输出名称为“御3T分图工具.exe”。
- [ ] `gui_classifier.py` 中 `dispatch_mode_or_continue()` 仅保留一次，且在 Tk 主窗口创建之前。
- [ ] 源码模式下四个工具模式均可启动对应窗口（进程保持运行、无报错退出）。
- [ ] 普通启动能打开主 GUI。
- [ ] `updater_github.ASSET_NAME_PATTERN` 能匹配“御3T分图工具.exe”，不能匹配“mavic3T.exe”。
- [ ] 若本任务完成一次构建，EXE 产物存在且能启动主 GUI 与工具模式。

## 八、测试用例

1. **Git 基线恢复**
   - 前置条件：`repo_tmp/gui_classifier/.git` 存在；根目录 `.git` 为空壳。
   - 操作：复制 `.git` 到根目录，执行 `git status`、`git log -5`。
   - 预期：命令成功，历史提交可见，工作树差异可查看。

2. **忽略规则**
   - 前置条件：`.gitignore` 已创建。
   - 操作：`git check-ignore` 依次检查 `build/`、`使用录屏.mp4`、`gui_classifier_config.json`、`repo_tmp/` 和 `gui_classifier.py`。
   - 预期：前四项被忽略，`gui_classifier.py` 未被忽略。

3. **静态与逻辑回归**
   - 前置条件：python 可用。
   - 操作：执行 AGENTS.md 中的语法检查和 `side_parser` 断言。
   - 预期：输出 `syntax ok`、`side parser ok`。

4. **工具模式分发（源码）**
   - 前置条件：依赖已安装。
   - 操作：分别运行 `python gui_classifier.py --tool=extension`、`--tool=button1`、`--tool=button2`、`--tool=button3`。
   - 预期：各模式窗口启动且进程保持，无报错退出。

5. **分发去重回归**
   - 前置条件：源码修改完成。
   - 操作：运行普通启动与四个工具模式。
   - 预期：行为与修改前一致，无重复窗口或重复分发。

6. **构建产物名**
   - 前置条件：根目录构建脚本可用。
   - 操作：检查脚本与 PyInstaller 命令的输出名称。
   - 预期：输出 `dist\御3T分图工具.exe`，脚本提示一致。

7. **更新器资产匹配**
   - 前置条件：`updater_github.py` 修改完成。
   - 操作：`python -c "import re, updater_github as u; p=re.compile(u.ASSET_NAME_PATTERN); print(bool(p.search('御3T分图工具.exe')), bool(p.search('mavic3T.exe')))"`。
   - 预期：输出 `True False`。

8. **主 GUI 启动（手动）**
   - 前置条件：桌面环境可用。
   - 操作：`python gui_classifier.py`。
   - 预期：主窗口出现；离线时更新检查失败不影响界面。

## 九、风险点

- Git 文件归属：沙箱用户与仓库属主不一致导致 `dubious ownership`；控制方式：`safe.directory` 或请求批准执行。
- `.git` 迁移：复制体积未知，可能耗时；控制方式：先备份、只复制不移动，本任务不清理历史副本。
- 忽略规则误伤：`repo_tmp/` 整体忽略符合“历史副本仅参考”约定，但需确认 README 和构建资料仍可手动查阅。
- Python 版本差异：本机 3.14 与目标 3.12；控制方式：记录差异，不把“能安装”当作兼容结论。
- GUI 启动受限：桌面会话可能受沙箱限制；控制方式：请求批准或由用户手动确认。
- 分发去重回归：冻结 EXE 环境与源码环境可能存在差异；控制方式：构建后补测所有 `--tool` 模式。

## 十、回滚方案

- Git：删除根目录 `.git` 后从 `repo_tmp/gui_classifier/.git` 重新复制（历史副本始终保留）。
- 文件：本任务文件在提交前保留修改副本；如需回退使用 `git restore` 仅针对本任务文件，禁止 `git reset --hard`。
- 构建产物：`build/`、`dist/` 已忽略，可安全删除后重建。
- 更新器与主入口：单文件小改动，可 `git revert` 或手工恢复。

## 十一、Codex 执行要求

- 先阅读 `AGENTS.md`、`gui_classifier.py`、`multi_mode_launcher.py`、`updater_github.py`、`repo_tmp/gui_classifier/build_exe.bat` 与 spec。
- 先说明涉及文件、当前实现、修改方案和风险点，再修改。
- 只修改与任务直接相关的内容，不做顺手重构。
- 不写入真实密钥、Token、生产路径；所有测试数据必须脱敏。
- 修改后运行上述测试；GUI 启动若受限，说明原因并给出替代验证。
- 涉及 `.git` 复制、`safe.directory` 配置或依赖安装时，先说明并获取批准；不执行破坏性 git 命令。

## 十二、完成后输出要求

- 修改了哪些文件。
- 完成了哪些需求（对照验收标准逐项说明）。
- 运行了哪些测试，结果如何。
- 哪些内容未测试（如 EXE 构建、GUI 启动），原因是什么。
- Git 迁移后的仓库状态摘要（`git log` 首条、`git status` 概要）。
- 是否存在未完成项或风险。
- 回滚方式是什么。
- 是否建议进入验收。
