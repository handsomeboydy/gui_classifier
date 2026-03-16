# button2.py
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ExifTags, UnidentifiedImageError

def get_exif_data(img):
    exif_data = {}
    info = img._getexif()
    if not info:
        return exif_data
    for tag, val in info.items():
        decoded = ExifTags.TAGS.get(tag, tag)
        if decoded == 'GPSInfo':
            gps = {}
            for t, v in val.items():
                gps[ExifTags.GPSTAGS.get(t, t)] = v
            exif_data['GPSInfo'] = gps
        else:
            exif_data[decoded] = val
    return exif_data

def dms_to_decimal(dms, ref):
    deg    = float(dms[0])
    minute = float(dms[1])
    sec    = float(dms[2])
    dec = deg + minute/60 + sec/3600
    if ref in ('S', 'W'):
        dec = -dec
    return dec

def get_coordinates(path):
    try:
        img = Image.open(path)
        exif = get_exif_data(img)
        gps = exif.get('GPSInfo')
        if not gps:
            raise ValueError('图片中未包含 GPS 信息')
        lat = dms_to_decimal(gps['GPSLatitude'], gps['GPSLatitudeRef'])
        lon = dms_to_decimal(gps['GPSLongitude'], gps['GPSLongitudeRef'])
        return lat, lon
    except (UnidentifiedImageError, ValueError) as e:
        messagebox.showerror('错误', f'获取坐标失败：\n{e}')
        return None, None
    except Exception as e:
        messagebox.showerror('错误', f'意外错误：\n{e}')
        return None, None

def main():
    root = tk.Tk()
    root.title('两张照片中心点坐标计算')
    root.geometry('600x250')

    tk.Label(root, text='请选择两张带 GPS 信息的照片：').pack(pady=5)

    path1_var = tk.StringVar()
    path2_var = tk.StringVar()

    frm = tk.Frame(root)
    frm.pack(pady=5)
    tk.Entry(frm, textvariable=path1_var, width=40).grid(row=0, column=0)
    tk.Button(frm, text='浏览1', command=lambda: path1_var.set(
        filedialog.askopenfilename(filetypes=[('图像文件','*.jpg;*.jpeg;*.png;*.tif;*.tiff')])
    )).grid(row=0, column=1, padx=5)

    tk.Entry(frm, textvariable=path2_var, width=40).grid(row=1, column=0, pady=5)
    tk.Button(frm, text='浏览2', command=lambda: path2_var.set(
        filedialog.askopenfilename(filetypes=[('图像文件','*.jpg;*.jpeg;*.png;*.tif;*.tiff')])
    )).grid(row=1, column=1, padx=5)

    center_var = tk.StringVar()
    tk.Entry(root, textvariable=center_var, width=60, state='readonly').pack(pady=10)

    def confirm_center():
        p1 = path1_var.get().strip()
        p2 = path2_var.get().strip()
        if not p1 or not p2:
            messagebox.showwarning('提示', '请先选择两张照片')
            return
        lat1, lon1 = get_coordinates(p1)
        lat2, lon2 = get_coordinates(p2)
        if None in (lat1, lon1, lat2, lon2):
            return
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        text = f"经度：{mid_lon:.6f}   纬度：{mid_lat:.6f}"
        center_var.set(text)
        root.clipboard_clear()
        root.clipboard_append(text)
        messagebox.showinfo('已复制', '中心点坐标已复制到剪贴板')

    tk.Button(root, text='确定', command=confirm_center).pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    main()
