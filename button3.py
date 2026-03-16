import os
import glob
import shutil
import json
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import tkinter.scrolledtext as scrolledtext
import exifread
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime

CONFIG_FILE = os.path.join(os.getcwd(), 'button3_config.json')

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_exif(img_path):
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)
    try:
        def _deg(v):
            d, m, s = [float(x.num)/float(x.den) for x in v.values]
            return d + m/60 + s/3600
        lat = _deg(tags['GPS GPSLatitude'])
        if tags['GPS GPSLatitudeRef'].printable != 'N': lat = -lat
        lon = _deg(tags['GPS GPSLongitude'])
        if tags['GPS GPSLongitudeRef'].printable != 'E': lon = -lon
    except KeyError:
        return None, None, None
    tstr = tags.get('EXIF DateTimeOriginal') or tags.get('Image DateTime')
    ts = None
    if tstr:
        try:
            ts = datetime.strptime(tstr.printable, '%Y:%m:%d %H:%M:%S')
        except:
            pass
    return lat, lon, ts

def load_ledger(path, line_name):
    df = pd.read_excel(path, dtype=str)
    df = df[df['线路名称'] == line_name].copy()
    df['经度'] = pd.to_numeric(df['经度'], errors='coerce')
    df['纬度'] = pd.to_numeric(df['纬度'], errors='coerce')
    return df[['杆塔编号', '经度', '纬度']]

def classify_radar(ledger_file, src_folder, output_root, line_name, dupe_threshold=30.0,logger=None):
    if logger is None:
        logger = print

    # 1. 读台账
    ledger = load_ledger(ledger_file, line_name)
    # ledger 返回 ['杆塔编号','经度','纬度']
    tower_ids = ledger['杆塔编号'].tolist()
    coords = {row['杆塔编号']:(row['纬度'], row['经度'])
              for _, row in ledger.iterrows()}
    print(f"加载杆塔 {len(tower_ids)} 根")

    # 2. 构建相邻两杆塔中点圆心和半径
    segments = []  # 存 (小号杆塔, center_lat, center_lon, radius_m)
    for a, b in zip(tower_ids, tower_ids[1:]):
        lat1, lon1 = coords[a]
        lat2, lon2 = coords[b]
        c_lat = (lat1 + lat2) / 2
        c_lon = (lon1 + lon2) / 2
        r = haversine(lat1, lon1, lat2, lon2) / 2
        segments.append((a, c_lat, c_lon, r))
    print(f"构建 {len(segments)} 段中点圆")

    # 3. 收集并排序照片
    imgs = []
    for p in glob.glob(os.path.join(src_folder, '*.*')):
        lat, lon, ts = get_exif(p)
        if lat is None or lon is None or ts is None:
            continue
        imgs.append({'path': p, 'lat': lat, 'lon': lon, 'time': ts})
    imgs.sort(key=lambda x: x['time'])
    print(f"读取 {len(imgs)} 张有效照片")

    # 4. 分组：落在哪个圆内就归到对应 small-id
    groups = {a: [] for a, _, _, _ in segments}
    for img in imgs:
        for a, clat, clon, rad in segments:
            if haversine(img['lat'], img['lon'], clat, clon) <= rad:
                groups[a].append(img)
                break

    # 5. 去重 & 输出
    total = 0
    for a, lst in groups.items():
        lst.sort(key=lambda x: x['time'])
        last = None
        for img in lst:
            if last is None or haversine(img['lat'], img['lon'], last['lat'], last['lon']) >= dupe_threshold:
                dst_dir = os.path.join(output_root, line_name, str(a))
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy(img['path'], dst_dir)
                logger(f"[已分类] {img['path']} → {dst_dir}")
                total += 1
                last = img
        print(f"杆塔{a}最终{len(groups[a])}张")  # 或者 len(filtered) if you want filtered count

    print(f"分图完成，共 {total} 张")
    return total


class RadarGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('激光雷达照片分图工具 GUI')
        self.geometry('600x500')
        self.src_dirs = []
        self._out_auto_set = False
        self._build_widgets()
        self._load_config()

    def _build_widgets(self):
        row = 0
        tk.Label(self, text='台账库文件夹:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.ledger_lib_var = tk.StringVar()
        tk.Entry(self, textvariable=self.ledger_lib_var, width=40).grid(row=row, column=1, padx=5)
        tk.Button(self, text='浏览', command=self._browse_ledger_lib).grid(row=row, column=2, padx=5)

        row += 1
        tk.Label(self, text='源照片文件夹:').grid(row=row, column=0, sticky='ne', padx=5, pady=5)
        self.src_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE, width=40, height=4)
        self.src_listbox.grid(row=row, column=1, sticky='nsew', padx=5)
        btn_frame = tk.Frame(self)
        tk.Button(btn_frame, text='添加', width=8, command=self._add_src).pack(pady=2)
        tk.Button(btn_frame, text='删除', width=8, command=self._remove_src).pack(pady=2)
        btn_frame.grid(row=row, column=2, sticky='n', padx=5)

        row += 1
        tk.Label(self, text='输出根目录:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.out_var = tk.StringVar()
        tk.Entry(self, textvariable=self.out_var, width=40).grid(row=row, column=1, padx=5)
        tk.Button(self, text='浏览', command=self._browse_out).grid(row=row, column=2, padx=5)

        row += 1
        tk.Label(self, text='线路名称:').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.line_var = tk.StringVar()
        tk.Entry(self, textvariable=self.line_var, width=40).grid(row=row, column=1, padx=5)

        row += 1
        tk.Label(self, text='重复过滤阈值 (m):').grid(row=row, column=0, sticky='e', padx=5, pady=5)
        self.dupe_var = tk.StringVar(value='30.0')
        tk.Entry(self, textvariable=self.dupe_var, width=10).grid(row=row, column=1, sticky='w', padx=5)

        row += 1
        tk.Button(self, text='开始分图', command=self._on_start, width=20).grid(row=row, column=1, pady=10)

        row += 1
        tk.Label(self, text='输出日志:').grid(row=row, column=0, sticky='nw', padx=5, pady=5)
        self.log_text = scrolledtext.ScrolledText(self, width=70, height=10, state='disabled')
        self.log_text.grid(row=row, column=1, columnspan=2, sticky='nsew', padx=5, pady=5)
        self.grid_rowconfigure(row, weight=1)
        self.grid_columnconfigure(2, weight=1)

    def _log(self, msg):
        self.log_text['state'] = 'normal'
        self.log_text.insert('end', msg + '\n')
        self.log_text.see('end')
        self.log_text['state'] = 'disabled'

    def _browse_ledger_lib(self):
        p = filedialog.askdirectory()
        if p:
            self.ledger_lib_var.set(p); self._save_config()
    def _save_config(self):
        try:
            json.dump({'ledger_lib_path': self.ledger_lib_var.get()},
                      open(CONFIG_FILE,'w',encoding='utf-8'))
        except: pass
    def _load_config(self):
        try:
            cfg = json.load(open(CONFIG_FILE,'r',encoding='utf-8'))
            p = cfg.get('ledger_lib_path','')
            if os.path.isdir(p): self.ledger_lib_var.set(p)
        except: pass

    def _add_src(self):
        init = os.path.dirname(self.src_dirs[-1]) if self.src_dirs else None
        p = filedialog.askdirectory(initialdir=init) if init else filedialog.askdirectory()
        if p and p not in self.src_dirs:
            self.src_dirs.append(p); self.src_listbox.insert('end', p)
            if not self._out_auto_set and not self.out_var.get():
                self.out_var.set(os.path.dirname(p)); self._out_auto_set = True

    def _remove_src(self):
        for i in reversed(self.src_listbox.curselection()):
            p = self.src_listbox.get(i)
            self.src_dirs.remove(p); self.src_listbox.delete(i)

    def _browse_out(self):
        p = filedialog.askdirectory()
        if p: self.out_var.set(p); self._out_auto_set = True

    def _on_start(self):
        self.log_text.configure(state='normal'); self.log_text.delete('1.0','end'); self.log_text.configure(state='disabled')
        lib = self.ledger_lib_var.get().strip()
        if not lib or not os.path.isdir(lib):
            messagebox.showwarning('提示','请选择台账库文件夹'); return
        self._log(f'台账库: {lib}')

        if not self.src_dirs:
            messagebox.showwarning('提示','请添加源照片文件夹'); return
        out = self.out_var.get().strip()
        if not out:
            messagebox.showwarning('提示','请选择输出根目录'); return
        self._log(f'输出根目录: {out}')

        line = self.line_var.get().strip()
        if not line:
            messagebox.showwarning('提示','请输入线路名称'); return
        self._log(f'线路名称: {line}')

        try:
            dt = float(self.dupe_var.get())
        except:
            messagebox.showerror('错误','过滤阈值须为数字'); return
        self._log(f'重复阈值: {dt} m')

        # 找台账文件
        ledger_file = None
        for fn in os.listdir(lib):
            if fn.lower().endswith(('.xlsx','.xls')):
                fp = os.path.join(lib,fn)
                try:
                    df = pd.read_excel(fp, dtype=str)
                    if '线路名称' in df.columns and not df[df['线路名称']==line].empty:
                        ledger_file = fp; break
                except: pass
        if not ledger_file:
            messagebox.showerror('错误',f'未找到线路"{line}"的台账'); return
        self._log(f'使用台账: {ledger_file}')

        total = 0
        for src in self.src_dirs:
            self._log(f'开始: {src}')
            c = classify_radar(ledger_file, src, out, line, dt)
            self._log(f'{src} 完成 {c} 张'); total += c
        self._log(f'全部完成，共 {total} 张'); messagebox.showinfo('完成',f'共分类 {total} 张')
 
if __name__ == '__main__':
    app = RadarGUI()
    app.mainloop()