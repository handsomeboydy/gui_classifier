import sys
from multi_mode_launcher import dispatch_mode_or_continue, spawn_self
dispatch_mode_or_continue()


import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.scrolledtext as scrolledtext
import json
import subprocess  # 用于启动拓展工具（保留）
import extension   # 保留：帮助 pyinstaller 收集模块

import classify
import classify_autonomous
import classify_channels

APP_VERSION = "1.0.1"  # 必须与当前 Release tag 对齐
import updater_github
from reporting import ResultRecorder
from preflight import LEVEL_ERROR, LEVEL_WARNING, preflight


# Config file for persisting ledger library path（保持你原来的用法）
CONFIG_FILE = os.path.join(os.getcwd(), 'gui_classifier_config.json')

# === 单 EXE 多模式：分发调用保持模块顶部唯一一次，且在创建主窗口前执行 ===
import threading
from tkinter import messagebox
# =======================================================

def check_update_ui(self, silent=False):
    def worker():
        try:
            if silent and (not updater_github.should_check_now()):
                return
            updater_github.mark_checked()

            latest, notes, tmp_exe = updater_github.prepare_update(APP_VERSION)
            if not tmp_exe:
                if not silent:
                    self.after(0, lambda: messagebox.showinfo("检查更新", "当前已是最新版本。"))
                return

            def ask():
                ok = messagebox.askyesno(
                    "发现新版本",
                    f"当前版本：{APP_VERSION}\n最新版本：{latest}\n\n更新说明：\n{notes}\n\n是否立即更新？"
                )
                if ok:
                    updater_github.apply_update_and_restart(tmp_exe)
                    messagebox.showinfo("更新", "已开始更新，程序将自动重启。")
                    self.destroy()  # 退出让 bat 覆盖 exe
            self.after(0, ask)

        except Exception as e:
            if not silent:
                self.after(0, lambda: messagebox.showwarning("检查更新失败", str(e)))

    threading.Thread(target=worker, daemon=True).start()

def redirect_stdout_to_widget(widget):
    class TextRedirector(object):
        def __init__(self, widget):
            self.widget = widget

        def write(self, text):
            self.widget.configure(state='normal')
            self.widget.insert('end', text)
            self.widget.see('end')
            self.widget.configure(state='disabled')

        def flush(self):
            pass

    return TextRedirector(widget)


class ClassifierGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("智能作业一班_御3T分图工具")
        self.geometry("600x550")

        self.src_dirs = []
        self._out_auto_set = False

        self._build_widgets()
        self._build_menu()          # ✅ 新增：菜单栏
        self._load_config()

        # ✅ 可选：启动后静默检查更新（有更新才弹）
        self.after(800, lambda: check_update_ui(self, silent=True))

    def _build_menu(self):
        menubar = tk.Menu(self)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="检查更新", command=lambda: check_update_ui(self, silent=False))
        help_menu.add_separator()
        help_menu.add_command(label="关于", command=lambda: messagebox.showinfo("关于", f"版本：{APP_VERSION}"))
        menubar.add_cascade(label="帮助", menu=help_menu)

        self.config(menu=menubar)


    def _build_widgets(self):
        row = 0
        # 台账库文件夹
        tk.Label(self, text="台账库文件夹:").grid(row=row, column=0, sticky="e")
        self.ledger_lib_var = tk.StringVar()
        tk.Entry(self, textvariable=self.ledger_lib_var, width=40).grid(row=row, column=1)
        tk.Button(self, text="浏览", command=self._browse_ledger_lib).grid(row=row, column=2)

        # 源照片文件夹（支持多选）
        row += 1
        tk.Label(self, text="源照片文件夹:").grid(row=row, column=0, sticky="ne")
        self.src_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE, width=40, height=4)
        self.src_listbox.grid(row=row, column=1, sticky="nsew")
        btn_frame = tk.Frame(self)
        tk.Button(btn_frame, text="添加", width=8, command=self._add_src).pack(pady=2)
        tk.Button(btn_frame, text="删除", width=8, command=self._remove_selected_src).pack(pady=2)
        btn_frame.grid(row=row, column=2, sticky="nw")

        # 输出根目录
        row += 1
        tk.Label(self, text="输出根目录:").grid(row=row, column=0, sticky="e")
        self.out_var = tk.StringVar()
        tk.Entry(self, textvariable=self.out_var, width=40).grid(row=row, column=1)
        tk.Button(self, text="浏览", command=self._browse_out).grid(row=row, column=2)

        # 线路名称（集成二级下拉）
        row += 1
        tk.Label(self, text="线路名称:").grid(row=row, column=0, sticky="e")
        frame = tk.Frame(self)
        frame.grid(row=row, column=1, columnspan=2, sticky="we")
        self.line_var = tk.StringVar()
        entry = tk.Entry(frame, textvariable=self.line_var)
        entry.pack(side="left", fill="x", expand=True)
        btn = tk.Button(frame, text="▼", width=2, command=self._show_line_menu)
        btn.pack(side="left")

        # 距离阈值
        row += 1
        tk.Label(self, text="距离阈值（米）:").grid(row=row, column=0, sticky="e")
        self.thresh_var = tk.StringVar(value="50")
        tk.Entry(self, textvariable=self.thresh_var).grid(row=row, column=1, columnspan=2, sticky="we")

        # 分图模式
        row += 1
        tk.Label(self, text="分图模式:").grid(row=row, column=0, sticky="e")
        self.mode_var = tk.StringVar(value="manual")
        tk.Radiobutton(self, text="手动飞行", variable=self.mode_var, value="manual").grid(row=row, column=1, sticky="w")
        tk.Radiobutton(self, text="自主飞行", variable=self.mode_var, value="auto").grid(row=row, column=2, sticky="w")

        # 结果清单脱敏开关（D-04）
        row += 1
        tk.Label(self, text="结果清单:").grid(row=row, column=0, sticky="e")
        self.sanitize_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self, text="脱敏输出（路径只留文件名，坐标保留两位小数）", variable=self.sanitize_var).grid(row=row, column=1, columnspan=2, sticky="w")

        # 按钮行：开始分类 & 拓展工具（各占 GUI 宽度的 1/6）
        row += 1
        btn_frame2 = tk.Frame(self)
        btn_frame2.grid(row=row, column=0, columnspan=3, pady=10, sticky="we")
        for c in range(7):
            btn_frame2.grid_columnconfigure(c, weight=1)

        self.start_btn = tk.Button(
            btn_frame2,
            text="开始分类",
            command=lambda: self._start_classify(False),
            bg="#4CAF50",
            fg="white"
        )
        self.start_btn.grid(row=0, column=1, sticky="we")

        self.analyze_btn = tk.Button(
            btn_frame2,
            text="仅分析（试运行）",
            command=lambda: self._start_classify(True),
            bg="#607D8B",
            fg="white"
        )
        self.analyze_btn.grid(row=0, column=2, sticky="we")


        update_btn = tk.Button(
            btn_frame2,
            text="检查更新",
            command=lambda: check_update_ui(self, silent=False),
            bg="#FF9800",
            fg="white"
        )
        update_btn.grid(row=0, column=3, sticky="we")


        ext_btn = tk.Button(
            btn_frame2,
            text="拓展工具",
            command=self._open_extension,
            bg="#2196F3",
            fg="white"
        )
        ext_btn.grid(row=0, column=5, sticky="we")

        # 日志输出区
        row += 1
        tk.Label(self, text="日志输出:").grid(row=row, column=0, sticky="nw")
        self.log_text = scrolledtext.ScrolledText(self, state='disabled', height=15)
        self.log_text.grid(row=row, column=1, columnspan=2, sticky="nsew")

        self.grid_rowconfigure(row, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def _load_config(self):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            path = cfg.get('ledger_lib_path', '')
            if path and os.path.isdir(path):
                self.ledger_lib_var.set(path)
        except Exception:
            pass

    def _save_config(self):
        try:
            cfg = {'ledger_lib_path': self.ledger_lib_var.get()}
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f)
            print(f"已保存台账库路径至配置文件: {CONFIG_FILE}\n")
        except Exception as e:
            print(f"保存配置失败: {e}\n")

    def _show_line_menu(self):
        """弹出二级菜单：一级为指定顺序的文件夹，二级为该目录下.xlsx文件（即使为空也显示文件夹）"""
        base = self.ledger_lib_var.get().strip()
        if not os.path.isdir(base):
            messagebox.showwarning("路径无效", "请先设置有效的台账库文件夹")
            return

        # 指定显示顺序
        order = [
            "1班", "2班", "3班", "4班", "5班", "6班",
            "8班", "9班", "10班",
            "11班（广宁）", "12班（怀集）", "13班（封开）", "14班（德庆）"
        ]

        menu = tk.Menu(self, tearoff=0)
        for folder in order:
            sub = tk.Menu(menu, tearoff=0)
            full_path = os.path.join(base, folder)
            if os.path.isdir(full_path):
                for fn in sorted(os.listdir(full_path)):
                    if fn.lower().endswith('.xlsx'):
                        def cmd(f=fn, fld=folder):
                            name = f
                            if name.endswith('经纬度台账.xlsx'):
                                name = name[:-len('经纬度台账.xlsx')]
                            else:
                                name = os.path.splitext(name)[0]
                            self.line_var.set(name)
                        sub.add_command(label=fn, command=cmd)
            # 即使目录为空或不存在，也显示该文件夹（空 submenu）
            menu.add_cascade(label=folder, menu=sub)

        x = self.winfo_pointerx()
        y = self.winfo_pointery()
        menu.tk_popup(x, y)

    def _browse_ledger_lib(self):
        path = filedialog.askdirectory()
        if path:
            self.ledger_lib_var.set(path)
            self._save_config()

    def _add_src(self):
        if self.src_dirs:
            initial = os.path.dirname(self.src_dirs[-1])
            path = filedialog.askdirectory(initialdir=initial)
        else:
            path = filedialog.askdirectory()
        if path and path not in self.src_dirs:
            self.src_dirs.append(path)
            self.src_listbox.insert('end', path)
            # 默认输出根目录为源文件夹的同级目录（仅首次生效）
            if not self._out_auto_set and not self.out_var.get():
                parent = os.path.dirname(path)
                self.out_var.set(parent)
                self._out_auto_set = True

    def _remove_selected_src(self):
        selections = list(self.src_listbox.curselection())
        for idx in reversed(selections):
            path = self.src_listbox.get(idx)
            if path in self.src_dirs:
                self.src_dirs.remove(path)
            self.src_listbox.delete(idx)

    def _browse_out(self):
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)
            self._out_auto_set = True

    def _start_classify(self, analyze_only=False):
        if getattr(self, "_busy", False):
            return
        self._busy = True
        self.start_btn.configure(state="disabled")
        self.analyze_btn.configure(state="disabled")
        self._save_config()
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')
        threading.Thread(target=self._do_classify, args=(analyze_only,), daemon=True).start()

    def _finish_run(self):
        self._busy = False
        self.start_btn.configure(state="normal")
        self.analyze_btn.configure(state="normal")

    def _do_classify(self, analyze_only=False):
        sys_stdout = sys.stdout
        sys.stderr = sys.stdout = redirect_stdout_to_widget(self.log_text)
        try:
            self._run_classify(analyze_only)
        finally:
            sys.stdout = sys.stderr = sys_stdout
            self.after(0, self._finish_run)

    def _run_classify(self, analyze_only=False):
        ledger_lib = self.ledger_lib_var.get().strip()
        out = self.out_var.get().strip()
        line = self.line_var.get().strip()
        try:
            thresh = float(self.thresh_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "距离阈值必须是数字")
            return

        if not ledger_lib or not self.src_dirs or not out or not line:
            messagebox.showwarning("参数不全", "请填写所有参数并添加至少一个源照片文件夹")
            return

        ledger_file = os.path.join(ledger_lib, f"{line}经纬度台账.xlsx")
        if not os.path.isfile(ledger_file):
            messagebox.showerror("参数错误", f"未找到台账文件: {ledger_file}")
            return

        mode = self.mode_var.get()
        pre = preflight(ledger_file, self.src_dirs, out, line, thresh, mode)
        print("\n===== 预检结果 =====")
        for issue in pre.rows():
            print(f"[{issue['级别']}] {issue['类别']}: {issue['说明']}")
        if pre.has_errors():
            errors = [i['说明'] for i in pre.rows() if i['级别'] == LEVEL_ERROR]
            messagebox.showerror("预检未通过", "\n".join(errors))
            return
        warnings = [i['说明'] for i in pre.rows() if i['级别'] == LEVEL_WARNING]
        if warnings:
            ok = messagebox.askyesno("预检警告", "存在以下警告：\n\n" + "\n".join(warnings) + "\n\n是否继续？")
            if not ok:
                return

        conflict_policy = "覆盖"
        if not analyze_only:
            choice = messagebox.askyesnocancel(
                "冲突处理",
                "目标路径存在同名文件时如何处理？\n\n是(Y)=覆盖（默认）\n否(N)=跳过\n取消=放弃本次运行"
            )
            if choice is None:
                return
            conflict_policy = "覆盖" if choice else "跳过"

        recorder = ResultRecorder()
        try:
            if mode == "manual":
                for src_folder in self.src_dirs:
                    print(f"\n====== 开始处理源文件夹: {src_folder} ======\n")
                    classify.classify(
                        ledger_file=ledger_file,
                        src_folder=src_folder,
                        output_root=out,
                        line_name=line,
                        threshold=thresh,
                        recorder=recorder,
                        conflict_policy=conflict_policy,
                        dry_run=analyze_only
                    )
            else:
                for src_folder in self.src_dirs:
                    print(f"\n====== 开始处理源文件夹: {src_folder} ======\n")
                    print("\n------ 提取通道照片 ------\n")
                    classify_channels.classify_channels(
                        ledger_file=ledger_file,
                        src_folder=src_folder,
                        output_root=out,
                        line_name=line,
                        threshold=thresh,
                        recorder=recorder,
                        conflict_policy=conflict_policy,
                        dry_run=analyze_only
                    )
                    print("通道分类完成。\n")
                    classify_autonomous.classify_autonomous(
                        ledger_file=ledger_file,
                        src_folder=src_folder,
                        output_root=out,
                        line_name=line,
                        threshold=thresh,
                        recorder=recorder,
                        conflict_policy=conflict_policy,
                        dry_run=analyze_only
                    )
                self._cleanup_skip_ir(out, line)
            csv_path = recorder.write_csv(out, line, sanitize=self.sanitize_var.get())
            summary = recorder.summary()
            print("\n总体处理完成。")
            print(f"结果清单位置: {csv_path}")
            print(f"扫描文件总数: {summary['总数']}")
            for result, count in summary["分类结果"].items():
                print(f"  {result}: {count}")
            for reason, count in summary["结果原因"].items():
                print(f"  原因[{reason}]: {count}")
            if analyze_only:
                messagebox.showinfo("仅分析完成", f"仅分析完成（未复制任何照片）\n清单位置:\n{csv_path}")
            else:
                messagebox.showinfo("完成", f"照片分类已完成\n清单位置:\n{csv_path}")
        except Exception as e:
            try:
                recorder.write_csv(out, line, sanitize=self.sanitize_var.get())
            except Exception:
                pass
            print(f"运行出错: {e}\n")
            messagebox.showerror("运行出错", str(e))

    def _cleanup_skip_ir(self, output_root: str, line_name: str):
        """自主分类全部完成后删除两个阶段之间的临时文件。"""
        skip_path = os.path.join(output_root, line_name, "skip_ir.txt")
        try:
            os.remove(skip_path)
            print(f"已删除临时文件: {skip_path}")
        except FileNotFoundError:
            pass

    def _open_extension(self):
        """
        打开拓展工具 GUI —— 单EXE多模式：拉起自身并进入 extension 模式
        """
        spawn_self("--tool=extension")


if __name__ == "__main__":
    app = ClassifierGUI()
    app.mainloop()
