import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import math
import os
import struct
import time

bmp_image = None

def validate_bmp(bmp_bytes):
    errors = []
    if not bmp_bytes or len(bmp_bytes) < 54:
        return ["File too small to be a valid BMP (needs at least 54 bytes)"]
    if bmp_bytes[0:2] != b"BM":
        errors.append("Not a BMP file (missing 'BM')")
    file_size = int.from_bytes(bmp_bytes[2:6], "little")
    if file_size != len(bmp_bytes):
        errors.append(f"Header file size ({file_size}) does not match actual size ({len(bmp_bytes)})")
    dib_size = int.from_bytes(bmp_bytes[14:18], "little")
    if dib_size != 40:
        errors.append(f"Unsupported DIB header size: {dib_size} (expected 40)")
    width = int.from_bytes(bmp_bytes[18:22], "little")
    height = int.from_bytes(bmp_bytes[22:26], "little")
    if width <= 0:
        errors.append(f"Invalid width: {width}")
    if height <= 0:
        errors.append(f"Invalid height: {height}")
    bpp = int.from_bytes(bmp_bytes[28:30], "little")
    if bpp not in (1, 4, 8, 24):
        errors.append(f"Unsupported bits-per-pixel: {bpp} (allowed: 1, 4, 8, 24)")
    return errors

class BmpImage:
    def __init__(self, filepath=None, bmp_bytes=None, scale=50, brightness=50, bpp=None, file_size=None):
        self.filepath = filepath
        self.bmp_bytes = bmp_bytes
        self.original_width = self._parse_width()
        self.original_height = self._parse_height()
        self.bpp = bpp
        self.file_size = file_size
        self.width = self.original_width
        self.height = self.original_height
        self.scale = scale
        self.brightness = brightness
        self.original_bytes = self.parse_rgb()
        self.r = True; self.g = True; self.b = True
        self.refresh()

    @classmethod
    def from_raw_rgb(cls, rgb_bytes, width, height, filepath=None, bpp=24, file_size=None):
        obj = cls.__new__(cls)
        obj.filepath = filepath
        obj.bmp_bytes = None
        obj.original_width = width
        obj.original_height = height
        obj.width = width
        obj.height = height
        obj.scale = 50
        obj.brightness = 50
        obj.original_bytes = rgb_bytes
        obj.r = obj.g = obj.b = True
        obj.bpp = bpp
        obj.file_size = file_size
        obj.refresh()
        return obj

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, "rb") as f:
            bmp_bytes = f.read()
        errors = validate_bmp(bmp_bytes)
        if errors:
            raise ValueError("BMP validation failed:\n" + "\n".join(f"- {e}" for e in errors))
        return cls(filepath=filepath, bmp_bytes=bmp_bytes)

    def get_file_size(self):
        if self.filepath and os.path.exists(self.filepath):
            try:
                return os.path.getsize(self.filepath)
            except OSError:
                pass
        if self.file_size:
            return self.file_size
        elif self.bmp_bytes:
            return int.from_bytes(self.bmp_bytes[2:6], "little")
        return 0

    def _parse_width(self):
        if self.bmp_bytes:
            return int.from_bytes(self.bmp_bytes[18:22], "little")
        return None

    def _parse_height(self):
        if self.bmp_bytes:
            return int.from_bytes(self.bmp_bytes[22:26], "little")
        return None

    def get_original_width(self): return self.original_width
    def get_original_height(self): return self.original_height
    def get_width(self): return self.width
    def get_height(self): return self.height

    def get_bits_per_pixel(self):
        if self.bpp:
            return self.bpp
        elif self.bmp_bytes:
            return int.from_bytes(self.bmp_bytes[28:30], 'little')
        return None

    def get_pixel_offset(self):
        return int.from_bytes(self.bmp_bytes[10:14], "little")

    def get_pixel_array(self):
        if self.bmp_bytes:
            offset = self.get_pixel_offset()
            return self.bmp_bytes[offset:]

    def get_colour_palette(self):
        dib_header_size = int.from_bytes(self.bmp_bytes[14:18], "little")
        bpp = self.get_bits_per_pixel()
        if bpp <= 8:
            num_colors = 2 ** bpp
            palette = []
            palette_start = 14 + dib_header_size
            for i in range(num_colors):
                b = self.bmp_bytes[palette_start + i*4]
                g = self.bmp_bytes[palette_start + i*4 + 1]
                r = self.bmp_bytes[palette_start + i*4 + 2]
                palette.append((r, g, b))
            return palette
        else:
            return []

    def get_stride(self):
        return ((self.get_original_width() * self.get_bits_per_pixel() + 31) // 32) * 4

    def parse_rgb(self):
        out = bytearray()
        width = self.get_original_width()
        height = self.get_original_height()
        if self.bmp_bytes:
            stride = self.get_stride()
            bits_per_pixel = self.get_bits_per_pixel()
            if bits_per_pixel <= 8:
                colour_palette = self.get_colour_palette()
            for i in range(height*stride - stride, -1, -stride):
                row = self.get_pixel_array()[i:i+stride]
                if (bits_per_pixel == 24):
                    bpp = bits_per_pixel // 8
                    for j in range(0, width * bpp, bpp):
                        b = row[j+0]; g = row[j+1]; r = row[j+2]
                        out += bytes((r, g, b))
                elif (bits_per_pixel == 8):
                    for j in range(0, width):
                        (r,g,b) = colour_palette[row[j]]
                        out += bytes((r, g, b))
                elif (bits_per_pixel == 4):
                    row_data_bytes = (width + 1) // 2
                    for j in range(row_data_bytes):
                        byte_val = row[j]
                        high = (byte_val >> 4) & 0x0F
                        (r,g,b) = colour_palette[high]
                        out += bytes((r, g, b))
                        if j * 2 + 1 < width:
                            low = byte_val & 0x0F
                            (r,g,b) = colour_palette[low]
                            out += bytes((r, g, b))
                elif (bits_per_pixel == 1):
                    row_data_bytes = (width + 7) // 8
                    for j in range(row_data_bytes):
                        byte_val = row[j]; base = j * 8
                        for bit in range(7, -1, -1):
                            x = base + (7 - bit)
                            if x >= width: break
                            idx = (byte_val >> bit) & 0x01
                            r, g, b = colour_palette[idx]
                            out += bytes((r, g, b))
        return bytes(out)

    def refresh_channels(self):
        w, h = self.get_width(), self.get_height()
        buf = self.get_displayable_image()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).astype(np.uint8)
        if not self.r: img[...,0] = 0
        if not self.g: img[...,1] = 0
        if not self.b: img[...,2] = 0
        self.display_bytes = img.tobytes()

    def refresh_scale(self):
        if self.scale <= 0:
            self.display_bytes = b""
            self.width = 0
            self.height = 0
            return
        W = self.get_original_width(); H = self.get_original_height()
        s = float(self.scale) / 50.0
        w = max(1, int(round(W * s))); h = max(1, int(round(H * s)))
        self.width = w; self.height = h
        buf = self.original_bytes; stride = W * 3; out = bytearray()
        if s <= 1.0:
            sx = W / w
            sy = H / h
            for y_out in range(h):
                ys = int(y_out * sy)
                high_y = max(ys + 1, math.floor((y_out + 1) * sy))
                for x_out in range(w):
                    xs = int(x_out * sx)
                    high_x = max(xs + 1, math.floor((x_out + 1) * sx))
                    r_run = g_run = b_run = 0.0
                    cnt = 0
                    for y_src in range(ys, high_y):
                        row_base = y_src * stride
                        for x_src in range(xs, high_x):
                            idx = row_base + x_src * 3
                            r_run += buf[idx + 0]; g_run += buf[idx + 1]; b_run += buf[idx + 2]; cnt += 1
                    out += bytes((int(round(r_run/cnt)), int(round(g_run/cnt)), int(round(b_run/cnt))))
        else:
            inv = 1.0 / s
            for y_out in range(h):
                y_src = min(H - 1, int(y_out * inv)); row_base = y_src * stride
                for x_out in range(w):
                    x_src = min(W - 1, int(x_out * inv)); idx = row_base + x_src * 3
                    out += buf[idx:idx+3]
        self.display_bytes = out

    rgb2yuv = np.array([
        [0.299,  0.587,  0.114],
        [-0.299, -0.587,  0.886],
        [0.701, -0.587, -0.114]
    ], dtype=np.float32)
    yuv2rgb = np.linalg.inv(rgb2yuv).astype(np.float32)

    def refresh_brightness(self):
        brightness = float(self.brightness)*2 / 100.0
        w, h = self.get_width(), self.get_height()
        buf = self.get_displayable_image()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
        yuv = img @ self.rgb2yuv.T
        yuv[..., 0] *= brightness
        out = yuv @ self.yuv2rgb.T
        out = np.clip(out, 0.0, 1.0)
        self.display_bytes = (out * 255.0 + 0.5).astype(np.uint8).tobytes()

    def refresh(self):
        if self.original_bytes:
            self.refresh_scale()
            self.refresh_brightness()
            self.refresh_channels()

    def set_scale(self, scale): self.scale = int(scale); self.refresh()
    def set_brightness(self, brightness): self.brightness = int(brightness); self.refresh()
    def set_r(self, state): self.r = state; self.refresh()
    def set_g(self, state): self.g = state; self.refresh()
    def set_b(self, state): self.b = state; self.refresh()
    def get_displayable_image(self): return self.display_bytes

# -------- .cmpt365 file --------
# magic[8]        = b"CMPT365\0"
# width           u32
# height          u32
# original_size   u32
# compressed_size u32
# payload: encoded bytes
CMPT365_MAGIC = b"CMPT365\x00"
HEADER_STRUCT = "<8sIIII"

def pack_header(width, height, original_size, compressed_size):
    return struct.pack(
        HEADER_STRUCT,
        CMPT365_MAGIC,
        width,
        height,
        original_size,
        compressed_size,
    )

def unpack_header(data: bytes):
    magic, width, height, original_size, compressed_size = struct.unpack(
        HEADER_STRUCT, data[:struct.calcsize(HEADER_STRUCT)]
    )
    return {
        "magic": magic,
        "width": width,
        "height": height,
        "original_size": original_size,
        "compressed_size": compressed_size,
    }

MAX_BITS = 16
MAX_DICT = (1 << MAX_BITS)
# Encode function adapted and ported to Python from GeeksForGeeks (LZW) and CMPT 365 course slides.
# https://www.geeksforgeeks.org/computer-networks/lzw-lempel-ziv-welch-compression-technique/
def lzw_encode(src_bytes: bytes) -> bytes:
    if not src_bytes:
        return bytes()

    table = {bytes([i]): i for i in range(256)}
    p = src_bytes[:1]
    next_code = 256
    out: list[int] = []

    for b in src_bytes[1:]:
        c = bytes([b])
        pc = p + c
        if pc in table:
            p = pc
        else:
            out.append(table[p])
            if next_code < MAX_DICT:
                table[pc] = next_code
                next_code += 1
            p = c

    out.append(table[p])
    encoded = struct.pack("<" + "H" * len(out), *out)
    return bytes(encoded)

# Decode function adapted and ported to Python from GeeksForGeeks (LZW) and CMPT 365 course slides.
# https://www.geeksforgeeks.org/computer-networks/lzw-lempel-ziv-welch-compression-technique/
def lzw_decode(src_encoded: bytes) -> bytes:
    if not src_encoded:
        return b""

    num_codes = len(src_encoded) // 2
    codes = list(struct.unpack("<" + "H" * num_codes, src_encoded))

    table = {i: bytes([i]) for i in range(256)}
    next_code = 256

    old = codes[0]
    s = table[old]
    out = bytearray(s)
    c = s[:1]

    for n in codes[1:]:
        if n in table:
            s = table[n]
        else:
            s = table[old] + c

        out.extend(s)
        c = s[:1]

        if next_code < MAX_DICT:
            table[next_code] = table[old] + c
            next_code += 1

        old = n

    return bytes(out)

def show_message_on_canvas(message):
    canvas.delete("all")
    canvas.create_text(
        int(canvas.winfo_width() or 600) // 2,
        int(canvas.winfo_height() or 350) // 2,
        text=message,
        fill="white",
        anchor="center"
    )

def clear_compression_stats():
    original_size_var.set("—")
    compressed_size_var.set("—")
    ratio_var.set("—")
    time_ms_var.set("—")

def post_open_success(img: "BmpImage", path: str):
    global bmp_image
    bmp_image = img
    file_path_entry.delete(0, tk.END)
    file_path_entry.insert(0, path)
    brightness_scale.set(50)
    scale_scale.set(50)
    r_toggle_var.set(True)
    g_toggle_var.set(True)
    b_toggle_var.set(True)
    clear_compression_stats()
    display_image(bmp_image)

def open_cmpt365_file(path: str):
    try:
        with open(path, "rb") as f:
            head_bytes = f.read(struct.calcsize(HEADER_STRUCT))
            hdr = unpack_header(head_bytes)
            payload = f.read()
    except Exception as e:
        clear_compression_stats()
        messagebox.showerror("Open Error", f"Failed to open .cmpt365:\n{e}")
        return

    if hdr["magic"] != CMPT365_MAGIC:
        clear_compression_stats()
        show_message_on_canvas("Invalid .cmpt365 file (bad magic).")
        return

    if hdr["compressed_size"] != len(payload):
        clear_compression_stats()
        messagebox.showerror(
            "Open Error",
            f"Invalid .cmpt365: expected {hdr['compressed_size']} bytes of payload, "
            f"got {len(payload)}."
        )
        return

    try:
        rgb_bytes = lzw_decode(payload)
    except Exception as e:
        clear_compression_stats()
        messagebox.showerror("Decode Error", f"LZW decode failed:\n{e}")
        return

    total_size = os.path.getsize(path)

    img = BmpImage.from_raw_rgb(
        rgb_bytes=rgb_bytes,
        width=hdr["width"],
        height=hdr["height"],
        filepath=path,
        bpp=24,
        file_size=total_size
    )
    post_open_success(img, path)

def open_bmp_file(path: str):
    global bmp_image

    try:
        img = BmpImage.from_file(path)
    except ValueError as e:
        bmp_image = None
        file_path_entry.delete(0, tk.END)
        file_size_var.set("—")
        width_var.set("—")
        height_var.set("—")
        bpp_var.set("—")
        clear_compression_stats()
        show_message_on_canvas(str(e))
        return
    except Exception as e:
        bmp_image = None
        clear_compression_stats()
        messagebox.showerror("Open Error", f"Failed to open BMP:\n{e}")
        return

    post_open_success(img, path)

def browse_file():
    path = filedialog.askopenfilename(
        filetypes=[
            ("Supported Images", "*.bmp *.cmpt365"),
            ("Bitmap Images", "*.bmp"),
            ("CMPT365 File", "*.cmpt365"),
        ]
    )
    if not path:
        return

    try:
        with open(path, "rb") as f:
            head = f.read(max(54, struct.calcsize(HEADER_STRUCT)))
    except Exception as e:
        clear_compression_stats()
        messagebox.showerror("Open Error", f"Failed to open file:\n{e}")
        return

    if head.startswith(CMPT365_MAGIC):
        open_cmpt365_file(path)
    else:
        open_bmp_file(path)


def display_image(bmp_obj):
    rgb_bytes = bmp_obj.get_displayable_image()
    width = bmp_obj.get_width(); height = bmp_obj.get_height()
    file_size_var.set(str(bmp_image.get_file_size()))
    width_var.set(str(bmp_image.get_original_width()))
    height_var.set(str(bmp_image.get_original_height()))
    bpp_var.set(str(bmp_image.get_bits_per_pixel()))
    expected_len = width * height * 3
    if rgb_bytes is None or len(rgb_bytes) != expected_len:
        print(f"Unexpected buffer size: got {len(rgb_bytes) if rgb_bytes else None}, expected {expected_len}")
        return
    img = Image.frombytes("RGB", (width, height), rgb_bytes)
    photo = ImageTk.PhotoImage(img)
    canvas.delete("all"); canvas.config(width=width, height=height)
    canvas.image = photo
    canvas.create_image(0, 0, anchor="nw", image=photo)

def on_brightness_change(value):
    if bmp_image:
        bmp_image.set_brightness(value); display_image(bmp_image)

def on_scale_change(value):
    if bmp_image:
        bmp_image.set_scale(value); display_image(bmp_image)

def on_toggle_r():
    if bmp_image:
        bmp_image.set_r(r_toggle_var.get()); display_image(bmp_image)

def on_toggle_g():
    if bmp_image:
        bmp_image.set_g(g_toggle_var.get()); display_image(bmp_image)

def on_toggle_b():
    if bmp_image:
        bmp_image.set_b(b_toggle_var.get()); display_image(bmp_image)

def compress_to_cmpt365():
    if not bmp_image:
        messagebox.showwarning("No Image", "Open a BMP file first.")
        return

    src = bmp_image.original_bytes
    original_size = bmp_image.get_file_size()
    width = bmp_image.get_original_width()
    height = bmp_image.get_original_height()

    t0 = time.perf_counter()
    encoded = lzw_encode(src)
    t1 = time.perf_counter()

    hdr = pack_header(width, height, original_size, len(encoded))

    default_name = os.path.splitext(os.path.basename(bmp_image.filepath or "output"))[0] + ".cmpt365"
    save_path = filedialog.asksaveasfilename(defaultextension=".cmpt365",
                                             initialfile=default_name,
                                             filetypes=[("CMPT365 File", "*.cmpt365")])
    if not save_path:
        messagebox.showerror("Compression Cancelled", "File not saved.")
        return

    try:
        with open(save_path, "wb") as f:
            f.write(hdr)
            f.write(encoded)
    except Exception as e:
        messagebox.showerror("Write Error", f"Failed to save file:\n{e}")
        return

    compressed_os_size = os.path.getsize(save_path)

    original_size_var.set(str(original_size))
    compressed_size_var.set(str(compressed_os_size))
    ratio = (original_size / compressed_os_size) if compressed_os_size else float('inf')
    ratio_var.set(f"{ratio:.3f}x")
    time_ms_var.set(f"{(t1 - t0)*1000:.2f}")
    messagebox.showinfo("Saved", f"Saved {save_path} successfully")

# -------- UI --------
root = tk.Tk()
root.title("BMP Viewer - CMPT365 PA2")
root.geometry("800x740")
root.resizable(True, True)

top_frame = tk.Frame(root); top_frame.pack(fill="x", padx=10, pady=(10, 6))
canvas = tk.Canvas(top_frame, width=600, height=350, bg="grey"); canvas.pack(fill="x")

controls_frame = tk.Frame(root); controls_frame.pack(fill="x", padx=10, pady=(0, 10))
tk.Label(controls_frame, text="File Path:").grid(row=0, column=0, padx=(0, 8), pady=5, sticky="w")
file_path_entry = tk.Entry(controls_frame); file_path_entry.grid(row=0, column=1, padx=0, pady=5, sticky="ew")
browse_btn = tk.Button(controls_frame, text="Browse", command=browse_file)
browse_btn.grid(row=0, column=2, padx=(8, 0), pady=5, sticky="w")
controls_frame.columnconfigure(1, weight=1)

grid_frame = tk.Frame(root); grid_frame.pack(fill="x", padx=10, pady=(0, 8))
tk.Label(grid_frame, text="File Size:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
file_size_var = tk.StringVar(value="—"); tk.Label(grid_frame, textvariable=file_size_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
tk.Label(grid_frame, text="Width:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
width_var = tk.StringVar(value="—"); tk.Label(grid_frame, textvariable=width_var).grid(row=0, column=3, sticky="w", padx=5, pady=5)
tk.Label(grid_frame, text="Height:").grid(row=0, column=4, sticky="e", padx=5, pady=5)
height_var = tk.StringVar(value="—"); tk.Label(grid_frame, textvariable=height_var).grid(row=0, column=5, sticky="w", padx=5, pady=5)
tk.Label(grid_frame, text="Bits Per Pixel:").grid(row=0, column=6, sticky="e", padx=5, pady=5)
bpp_var = tk.StringVar(value="—"); tk.Label(grid_frame, textvariable=bpp_var).grid(row=0, column=7, sticky="w", padx=5, pady=5)

r_toggle_var = tk.BooleanVar(value=True); g_toggle_var = tk.BooleanVar(value=True); b_toggle_var = tk.BooleanVar(value=True)
tk.Checkbutton(grid_frame, text="R", variable=r_toggle_var, command=on_toggle_r).grid(row=0, column=8, padx=(15, 5), pady=5, sticky="w")
tk.Checkbutton(grid_frame, text="G", variable=g_toggle_var, command=on_toggle_g).grid(row=0, column=9, padx=5, pady=5, sticky="w")
tk.Checkbutton(grid_frame, text="B", variable=b_toggle_var, command=on_toggle_b).grid(row=0, column=10, padx=5, pady=5, sticky="w")

sliders_frame = tk.Frame(root); sliders_frame.pack(fill="x", padx=10, pady=(0, 8))
tk.Label(sliders_frame, text="Brightness").grid(row=0, column=0, sticky="w", padx=(0, 8))
brightness_scale = tk.Scale(sliders_frame, from_=0, to=100, orient="horizontal", command=on_brightness_change)
brightness_scale.set(50); brightness_scale.grid(row=0, column=1, sticky="ew", padx=(0, 12))
tk.Label(sliders_frame, text="Scale").grid(row=0, column=2, sticky="w", padx=(0, 8))
scale_scale = tk.Scale(sliders_frame, from_=0, to=100, orient="horizontal", command=on_scale_change)
scale_scale.set(50); scale_scale.grid(row=0, column=3, sticky="ew")
sliders_frame.columnconfigure(1, weight=1); sliders_frame.columnconfigure(3, weight=1)

comp_frame = tk.LabelFrame(root, text="Compression (LZW)")
comp_frame.pack(fill="x", padx=10, pady=(0, 10))
tk.Button(comp_frame, text="Compress (.cmpt365)", command=compress_to_cmpt365).grid(row=0, column=0, padx=(0, 8), pady=8, sticky="w")

tk.Label(comp_frame, text="Original Size:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
original_size_var = tk.StringVar(value="—"); tk.Label(comp_frame, textvariable=original_size_var).grid(row=1, column=1, sticky="w", padx=5, pady=3)
tk.Label(comp_frame, text="Compressed Size:").grid(row=1, column=2, sticky="e", padx=5, pady=3)
compressed_size_var = tk.StringVar(value="—"); tk.Label(comp_frame, textvariable=compressed_size_var).grid(row=1, column=3, sticky="w", padx=5, pady=3)
tk.Label(comp_frame, text="Ratio:").grid(row=2, column=0, sticky="e", padx=5, pady=3)
ratio_var = tk.StringVar(value="—"); tk.Label(comp_frame, textvariable=ratio_var).grid(row=2, column=1, sticky="w", padx=5, pady=3)
tk.Label(comp_frame, text="Time (ms):").grid(row=2, column=2, sticky="e", padx=5, pady=3)
time_ms_var = tk.StringVar(value="—"); tk.Label(comp_frame, textvariable=time_ms_var).grid(row=2, column=3, sticky="w", padx=5, pady=3)
for i in range(4): comp_frame.columnconfigure(i, weight=1)

root.mainloop()
