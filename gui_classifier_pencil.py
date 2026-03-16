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

# Config file for persisting ledger library path（保持你原来的用法）
CONFIG_FILE = os.path.join(os.getcwd(), 'gui_classifier_config.json')

# === 单 EXE 多模式：加入分发器（必须在创建主窗口前调用） ===
from multi_mode_launcher import dispatch_mode_or_continue, spawn_self
dispatch_mode_or_continue()
# =======================================================


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
        self.geometry("1200x800")
        self.configure(bg="#F2F3F0")  # 背景色

        # 支持多个源文件夹
        self.src_dirs = []
        # 标记输出目录是否已手动设置
        self._out_auto_set = False

        self._build_widgets()
        self._load_config()

    def _create_card(self, parent, **kwargs):
        """创建卡片样式的Frame"""
        card = tk.Frame(
            parent,
            bg="#FFFFFF",
            relief="flat",
            bd=1,
            highlightbackground="#CBCCC9",
            highlightthickness=1,
            **kwargs
        )
        return card

    def _create_input_field(self, parent, placeholder="", textvariable=None):
        """创建输入框"""
        frame = tk.Frame(parent, bg="#FFFFFF", relief="flat", bd=1, highlightbackground="#CBCCC9", highlightthickness=1)
        entry = tk.Entry(
            frame,
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14),
            relief="flat",
            bd=0,
            insertbackground="#111111",
            textvariable=textvariable
        )
        if placeholder and not textvariable:
            entry.insert(0, placeholder)
            entry.config(fg="#666666")
            entry.bind("<FocusIn>", lambda e, p=placeholder: self._on_entry_focus_in(e, p))
            entry.bind("<FocusOut>", lambda e, p=placeholder: self._on_entry_focus_out(e, p))
        entry.pack(side="left", fill="both", expand=True, padx=16, pady=8)
        return frame, entry

    def _on_entry_focus_in(self, event, placeholder):
        """输入框获得焦点时清除占位符"""
        if event.widget.get() == placeholder:
            event.widget.delete(0, tk.END)
            event.widget.config(fg="#111111")

    def _on_entry_focus_out(self, event, placeholder):
        """输入框失去焦点时恢复占位符"""
        if not event.widget.get():
            event.widget.insert(0, placeholder)
            event.widget.config(fg="#666666")

    def _create_button(self, parent, text, command, bg_color, fg_color="white", width=None):
        """创建按钮"""
        btn = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=fg_color,
            font=("JetBrains Mono", 14, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=24,
            pady=12,
            width=width
        )
        btn.bind("<Enter>", lambda e: btn.config(bg=self._lighten_color(bg_color)))
        btn.bind("<Leave>", lambda e: btn.config(bg=bg_color))
        return btn

    def _lighten_color(self, color):
        """轻微提亮颜色（用于悬停效果）"""
        color_map = {
            "#4CAF50": "#66BB6A",
            "#2196F3": "#42A5F5"
        }
        return color_map.get(color, color)

    def _build_widgets(self):
        # 主容器：左右分栏
        main_container = tk.Frame(self, bg="#F2F3F0")
        main_container.pack(fill="both", expand=True, padx=32, pady=32)

        # 左侧面板：配置参数
        left_panel = tk.Frame(main_container, bg="#F2F3F0", width=600)
        left_panel.pack(side="left", fill="both", padx=(0, 24))

        # 配置参数卡片
        config_card = self._create_card(left_panel)
        config_card.pack(fill="x", pady=(0, 24))

        # 卡片内容容器
        card_content = tk.Frame(config_card, bg="#FFFFFF")
        card_content.pack(fill="both", padx=24, pady=24)

        # 标题
        title_label = tk.Label(
            card_content,
            text="配置参数",
            bg="#FFFFFF",
            fg="#111111",
            font=("JetBrains Mono", 20, "bold")
        )
        title_label.pack(anchor="w", pady=(0, 24))

        # 台账库文件夹
        ledger_group = tk.Frame(card_content, bg="#FFFFFF")
        ledger_group.pack(fill="x", pady=(0, 20))
        
        tk.Label(
            ledger_group,
            text="台账库文件夹",
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14, "bold")
        ).pack(anchor="w", pady=(0, 6))
        
        ledger_row = tk.Frame(ledger_group, bg="#FFFFFF")
        ledger_row.pack(fill="x")
        
        self.ledger_lib_var = tk.StringVar()
        ledger_input_frame, ledger_entry = self._create_input_field(ledger_row, "请选择台账库文件夹", self.ledger_lib_var)
        ledger_input_frame.pack(side="left", fill="x", expand=True, padx=(0, 12))
        
        ledger_btn = self._create_button(ledger_row, "浏览", self._browse_ledger_lib, "#2196F3", width=10)
        ledger_btn.pack(side="left")

        # 源照片文件夹
        src_group = tk.Frame(card_content, bg="#FFFFFF")
        src_group.pack(fill="x", pady=(0, 20))
        
        tk.Label(
            src_group,
            text="源照片文件夹",
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14, "bold")
        ).pack(anchor="w", pady=(0, 6))
        
        src_row = tk.Frame(src_group, bg="#FFFFFF")
        src_row.pack(fill="x")
        
        src_list_frame = tk.Frame(src_row, bg="#FFFFFF", relief="flat", bd=1, highlightbackground="#CBCCC9", highlightthickness=1)
        src_list_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))
        
        self.src_listbox = tk.Listbox(
            src_list_frame,
            selectmode=tk.MULTIPLE,
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14),
            relief="flat",
            bd=0,
            height=5
        )
        self.src_listbox.pack(fill="both", expand=True, padx=16, pady=12)
        
        src_btn_frame = tk.Frame(src_row, bg="#FFFFFF")
        src_btn_frame.pack(side="left")
        
        add_btn = self._create_button(src_btn_frame, "添加", self._add_src, "#4CAF50", width=10)
        add_btn.pack(pady=(0, 8))
        
        remove_btn = self._create_button(src_btn_frame, "删除", self._remove_selected_src, "#2196F3", width=10)
        remove_btn.pack()

        # 输出根目录
        out_group = tk.Frame(card_content, bg="#FFFFFF")
        out_group.pack(fill="x", pady=(0, 20))
        
        tk.Label(
            out_group,
            text="输出根目录",
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14, "bold")
        ).pack(anchor="w", pady=(0, 6))
        
        out_row = tk.Frame(out_group, bg="#FFFFFF")
        out_row.pack(fill="x")
        
        self.out_var = tk.StringVar()
        out_input_frame, out_entry = self._create_input_field(out_row, "请选择输出根目录", self.out_var)
        out_input_frame.pack(side="left", fill="x", expand=True, padx=(0, 12))
        
        out_btn = self._create_button(out_row, "浏览", self._browse_out, "#2196F3", width=10)
        out_btn.pack(side="left")

        # 线路名称
        line_group = tk.Frame(card_content, bg="#FFFFFF")
        line_group.pack(fill="x", pady=(0, 20))
        
        tk.Label(
            line_group,
            text="线路名称",
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14, "bold")
        ).pack(anchor="w", pady=(0, 6))
        
        self.line_var = tk.StringVar()
        line_input_frame, line_entry = self._create_input_field(line_group, "请选择线路名称", self.line_var)
        line_input_frame.pack(fill="x")
        
        # 添加下拉按钮
        line_dropdown_btn = tk.Button(
            line_input_frame,
            text="▼",
            command=self._show_line_menu,
            bg="#FFFFFF",
            fg="#666666",
            font=("Geist", 12),
            relief="flat",
            bd=0,
            width=3,
            cursor="hand2"
        )
        line_dropdown_btn.pack(side="right", padx=(0, 8))

        # 距离阈值
        thresh_group = tk.Frame(card_content, bg="#FFFFFF")
        thresh_group.pack(fill="x", pady=(0, 20))
        
        tk.Label(
            thresh_group,
            text="距离阈值（米）",
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14, "bold")
        ).pack(anchor="w", pady=(0, 6))
        
        self.thresh_var = tk.StringVar(value="50")
        thresh_input_frame, thresh_entry = self._create_input_field(thresh_group, "", self.thresh_var)
        thresh_input_frame.pack(fill="x")

        # 分图模式
        mode_group = tk.Frame(card_content, bg="#FFFFFF")
        mode_group.pack(fill="x", pady=(0, 20))
        
        tk.Label(
            mode_group,
            text="分图模式",
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14, "bold")
        ).pack(anchor="w", pady=(0, 12))
        
        mode_radio_frame = tk.Frame(mode_group, bg="#FFFFFF")
        mode_radio_frame.pack(anchor="w")
        
        self.mode_var = tk.StringVar(value="manual")
        
        manual_radio = tk.Radiobutton(
            mode_radio_frame,
            text="手动飞行",
            variable=self.mode_var,
            value="manual",
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14),
            selectcolor="#111111",
            activebackground="#FFFFFF",
            activeforeground="#111111"
        )
        manual_radio.pack(side="left", padx=(0, 24))
        
        auto_radio = tk.Radiobutton(
            mode_radio_frame,
            text="自主飞行",
            variable=self.mode_var,
            value="auto",
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 14),
            selectcolor="#111111",
            activebackground="#FFFFFF",
            activeforeground="#111111"
        )
        auto_radio.pack(side="left")

        # 操作按钮
        action_frame = tk.Frame(left_panel, bg="#F2F3F0")
        action_frame.pack(fill="x", pady=(0, 0))
        
        start_btn = self._create_button(action_frame, "开始分类", self._start_classify, "#4CAF50", width=15)
        start_btn.pack(side="left", padx=(0, 16))
        
        ext_btn = self._create_button(action_frame, "拓展工具", self._open_extension, "#2196F3", width=15)
        ext_btn.pack(side="left")

        # 右侧面板：日志输出
        right_panel = tk.Frame(main_container, bg="#F2F3F0")
        right_panel.pack(side="right", fill="both", expand=True)

        # 日志卡片
        log_card = self._create_card(right_panel)
        log_card.pack(fill="both", expand=True)

        log_content = tk.Frame(log_card, bg="#FFFFFF")
        log_content.pack(fill="both", padx=24, pady=24)

        # 日志标题
        log_title = tk.Label(
            log_content,
            text="日志输出",
            bg="#FFFFFF",
            fg="#111111",
            font=("JetBrains Mono", 20, "bold")
        )
        log_title.pack(anchor="w", pady=(0, 20))

        # 日志内容区域
        log_text_frame = tk.Frame(
            log_content,
            bg="#FFFFFF",
            relief="flat",
            bd=1,
            highlightbackground="#CBCCC9",
            highlightthickness=1
        )
        log_text_frame.pack(fill="both", expand=True)

        self.log_text = scrolledtext.ScrolledText(
            log_text_frame,
            state='disabled',
            bg="#FFFFFF",
            fg="#111111",
            font=("Geist", 12),
            relief="flat",
            bd=0,
            wrap=tk.WORD
        )
        self.log_text.pack(fill="both", expand=True, padx=16, pady=16)
        self.log_text.insert('1.0', "等待开始分类...")

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

    def _start_classify(self):
        self._save_config()
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')
        threading.Thread(target=self._do_classify, daemon=True).start()

    def _do_classify(self):
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

        sys_stdout = sys.stdout
        sys.stderr = sys.stdout = redirect_stdout_to_widget(self.log_text)

        try:
            if self.mode_var.get() == "manual":
                for src_folder in self.src_dirs:
                    print(f"\n====== 开始处理源文件夹: {src_folder} ======\n")
                    classify.classify(
                        ledger_file=ledger_file,
                        src_folder=src_folder,
                        output_root=out,
                        line_name=line,
                        threshold=thresh
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
                        threshold=thresh
                    )
                    print("通道分类完成。\n")
                    classify_autonomous.classify_autonomous(
                        ledger_file=ledger_file,
                        src_folder=src_folder,
                        output_root=out,
                        line_name=line,
                        threshold=thresh
                    )
            print("\n总体分类完成。")
            messagebox.showinfo("完成", "照片分类已完成")
        except Exception as e:
            print(f"运行出错: {e}\n")
            messagebox.showerror("运行出错", str(e))
        finally:
            sys.stdout = sys.stderr = sys_stdout

    def _open_extension(self):
        """
        打开拓展工具 GUI —— 单EXE多模式：拉起自身并进入 extension 模式
        """
        spawn_self("--tool=extension")


if __name__ == "__main__":
    app = ClassifierGUI()
    app.mainloop()

