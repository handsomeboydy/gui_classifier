import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.scrolledtext as scrolledtext
import sys
import json

import classify
import classify_channels
import classify_autonomous

CONFIG_FILE = os.path.join(os.getcwd(), 'gui_classifier_config.json')


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
        self.geometry("600x500")
        self._build_widgets()
        self._load_config()

    def _build_widgets(self):
        row = 0
        tk.Label(self, text="台账库文件夹:").grid(row=row, column=0, sticky="e")
        self.ledger_lib_var = tk.StringVar()
        tk.Entry(self, textvariable=self.ledger_lib_var, width=40).grid(row=row, column=1)
        tk.Button(self, text="浏览", command=self._browse_ledger_lib).grid(row=row, column=2)

        row += 1
        tk.Label(self, text="源照片文件夹:").grid(row=row, column=0, sticky="e")
        self.src_var = tk.StringVar()
        tk.Entry(self, textvariable=self.src_var, width=40).grid(row=row, column=1)
        tk.Button(self, text="浏览", command=self._browse_src).grid(row=row, column=2)

        row += 1
        tk.Label(self, text="输出根目录:").grid(row=row, column=0, sticky="e")
        self.out_var = tk.StringVar()
        tk.Entry(self, textvariable=self.out_var, width=40).grid(row=row, column=1)
        tk.Button(self, text="浏览", command=self._browse_out).grid(row=row, column=2)

        row += 1
        tk.Label(self, text="线路名称:").grid(row=row, column=0, sticky="e")
        self.line_var = tk.StringVar()
        tk.Entry(self, textvariable=self.line_var).grid(row=row, column=1, columnspan=2, sticky="we")

        row += 1
        tk.Label(self, text="距离阈值（米）:").grid(row=row, column=0, sticky="e")
        self.thresh_var = tk.StringVar(value="50")
        tk.Entry(self, textvariable=self.thresh_var).grid(row=row, column=1, columnspan=2, sticky="we")

        row += 1
        tk.Label(self, text="分图模式:").grid(row=row, column=0, sticky="e")
        self.mode_var = tk.StringVar(value="manual")
        tk.Radiobutton(self, text="手动飞行", variable=self.mode_var, value="manual").grid(row=row, column=1, sticky="w")
        tk.Radiobutton(self, text="自主飞行", variable=self.mode_var, value="auto").grid(row=row, column=2, sticky="w")

        row += 1
        tk.Button(self, text="开始分类", command=self._start_classify, bg="#4CAF50", fg="white") \
            .grid(row=row, column=0, columnspan=3, pady=10, sticky="we")

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

    def _browse_ledger_lib(self):
        path = filedialog.askdirectory()
        if path:
            self.ledger_lib_var.set(path)
            self._save_config()

    def _browse_src(self):
        path = filedialog.askdirectory()
        if path:
            self.src_var.set(path)

    def _browse_out(self):
        path = filedialog.askdirectory()
        if path:
            self.out_var.set(path)

    def _start_classify(self):
        self._save_config()
        self.log_text.configure(state='normal')
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state='disabled')
        threading.Thread(target=self._do_classify, daemon=True).start()

    def _do_classify(self):
        ledger_lib = self.ledger_lib_var.get().strip()
        src = self.src_var.get().strip()
        out = self.out_var.get().strip()
        line = self.line_var.get().strip()
        try:
            thresh = float(self.thresh_var.get())
        except ValueError:
            messagebox.showerror("参数错误", "距离阈值必须是数字")
            return
        if not all((ledger_lib, src, out, line)):
            messagebox.showwarning("参数不全", "请填写所有参数")
            return

        ledger_file = os.path.join(ledger_lib, f"{line}经纬度台账.xlsx")
        if not os.path.isfile(ledger_file):
            messagebox.showerror("参数错误", f"未找到台账文件: {ledger_file}")
            return

        sys_stdout = sys.stdout
        sys.stderr = sys.stdout = redirect_stdout_to_widget(self.log_text)

        try:
            if self.mode_var.get() == "manual":
                classify.classify(
                    ledger_file=ledger_file,
                    src_folder=src,
                    output_root=out,
                    line_name=line,
                    threshold=thresh
                )
            else:
                classify_channels.classify_channels(
                    ledger_file=ledger_file,
                    src_folder=src,
                    output_root=out,
                    line_name=line,
                    threshold=thresh
                )
                print("通道分类完成。\n")
                classify_autonomous.classify_autonomous(
                    ledger_file=ledger_file,
                    src_folder=src,
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


if __name__ == "__main__":
    app = ClassifierGUI()
    app.mainloop()
