import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox

from multi_mode_launcher import spawn_self  # 新增：使用多模式自启动

# 按钮名称与对应脚本文件名（沿用你的命名）
BUTTONS = [
    ("直接取坐标", "button1.py"),
    ("地线取坐标", "button2.py"),
    ("激光雷达分图", "button3.py"),
]

def run_script(script_name):
    """
    兼容旧逻辑的占位：保留函数（以免外部还有引用），
    但不再直接解释执行 .py，而是改用“自身多模式”。
    """
    mode = "--tool=" + os.path.splitext(script_name)[0]  # button1.py -> --tool=button1
    spawn_self(mode)

def main():
    root = tk.Tk()
    root.title("扩展工具")
    root.geometry("400x300")

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=20)

    for name, script in BUTTONS:
        btn = tk.Button(
            btn_frame,
            text=name,
            width=20,
            height=2,
            command=lambda s=script: run_script(s)
        )
        btn.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
