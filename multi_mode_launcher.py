# -*- coding: utf-8 -*-
# 单 EXE 多模式启动器（做法B核心）
import os, sys, subprocess, runpy

def spawn_self(mode_flag: str):
    """
    以“自身”为程序，附带 --tool=xxx 参数启动新进程。
    源码运行：python gui_classifier.py --tool=xxx
    打包运行：gui_classifier.exe --tool=xxx
    """
    try:
        if getattr(sys, "frozen", False):
            app = sys.executable                      # 当前 exe
            args = [app, mode_flag]
            cwd  = os.path.dirname(app)
        else:
            app = sys.executable                      # python.exe
            script = os.path.abspath(sys.argv[0])     # gui_classifier.py
            args = [app, script, mode_flag]
            cwd  = os.path.dirname(script)
        subprocess.Popen(args, cwd=cwd)
    except Exception as e:
        try:
            from tkinter import messagebox
            messagebox.showerror("错误", f"无法启动子进程：{e}")
        except Exception:
            print(f"[multi_mode_launcher] 启动失败: {e}")

def dispatch_mode_or_continue():
    """
    若命令行含 --tool=xxx，则直接在当前进程执行对应模块的 __main__ 逻辑，
    然后退出；否则继续回到主程序的正常启动流程。
    这里不要求 button1/2/3 有 main()；用 run_module(..., '__main__') 直接跑顶层代码。
    """
    for arg in sys.argv[1:]:
        if arg.startswith("--tool="):
            tool = arg.split("=", 1)[1].strip()
            if tool in {"extension", "button1", "button2", "button3"}:
                runpy.run_module(tool, run_name="__main__")
                sys.exit(0)
            else:
                print(f"[multi_mode_launcher] 未知 tool: {tool}（忽略）")
