# button1.py
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
    root.title('单张照片坐标提取')
    root.geometry('500x200')

    tk.Label(root, text='请选择一张带 GPS 信息的照片：').pack(pady=5)

    path_var = tk.StringVar()
    tk.Entry(root, textvariable=path_var, width=50).pack()

    def choose_file():
        p = filedialog.askopenfilename(
            filetypes=[('图像文件', '*.jpg;*.jpeg;*.png;*.tif;*.tiff')]
        )
        if p:
            path_var.set(p)

    tk.Button(root, text='浏览', command=choose_file).pack(pady=5)

    coord_var = tk.StringVar()
    entry = tk.Entry(root, textvariable=coord_var, width=50, state='readonly')
    entry.pack(pady=5)

    def confirm():
        p = path_var.get().strip()
        if not p:
            messagebox.showwarning('提示', '请先选择照片')
            return
        lat, lon = get_coordinates(p)
        if lat is None or lon is None:
            return
        text = f"经度：{lon:.6f}   纬度：{lat:.6f}"
        coord_var.set(text)
        root.clipboard_clear()
        root.clipboard_append(text)
        messagebox.showinfo('已复制', '坐标已复制到剪贴板')

    tk.Button(root, text='确定', command=confirm).pack(pady=5)

    root.mainloop()

if __name__ == '__main__':
    main()
