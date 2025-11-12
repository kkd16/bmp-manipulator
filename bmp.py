import numpy as np
from typing import Self
import math
import os

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
    def validate(cls, bmp_bytes) -> list[str]:
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


    @classmethod
    def from_file(cls, filepath) -> Self:
        with open(filepath, "rb") as f:
            bmp_bytes = f.read()
        errors = BmpImage.validate(bmp_bytes)
        if errors:
            raise ValueError("BMP validation failed:\n" + "\n".join(f"- {e}" for e in errors))
        return cls(filepath=filepath, bmp_bytes=bmp_bytes)

    def get_file_size(self) -> int:
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

    def _parse_width(self) -> int:
        if self.bmp_bytes:
            return int.from_bytes(self.bmp_bytes[18:22], "little")
        return 0

    def _parse_height(self) -> int:
        if self.bmp_bytes:
            return int.from_bytes(self.bmp_bytes[22:26], "little")
        return 0

    def get_original_width(self) -> int:
        if self.original_width is not None:
            return self.original_width
        return 0

    def get_original_height(self) -> int:
        if self.original_height is not None:
            return self.original_height
        return 0

    def get_width(self) -> int:
        if self.width is not None:
            return self.width
        return 0

    def get_height(self) -> int:
        if self.height is not None:
            return self.height
        return 0

    def get_bits_per_pixel(self) -> int:
        if self.bpp:
            return self.bpp
        elif self.bmp_bytes:
            return int.from_bytes(self.bmp_bytes[28:30], 'little')
        return 0

    def get_pixel_offset(self) -> int:
        if self.bmp_bytes:
            return int.from_bytes(self.bmp_bytes[10:14], "little")
        return 0

    def get_pixel_array(self) -> bytes:
        if self.bmp_bytes:
            offset = self.get_pixel_offset()
            return self.bmp_bytes[offset:]
        return bytes()

    def get_colour_palette(self) -> list[tuple[int, int, int]]:
        if self.bmp_bytes:
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
        return []

    def get_stride(self) -> int:
        return ((self.get_original_width() * self.get_bits_per_pixel() + 31) // 32) * 4

    def parse_rgb(self) -> bytes:
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

    def refresh_channels(self) -> None:
        w, h = self.get_width(), self.get_height()
        buf = self.get_displayable_image()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).astype(np.uint8)
        if not self.r: img[...,0] = 0
        if not self.g: img[...,1] = 0
        if not self.b: img[...,2] = 0
        self.display_bytes = img.tobytes()

    def refresh_scale(self) -> None:
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

    def refresh_brightness(self) -> None:
        brightness = float(self.brightness)*2 / 100.0
        w, h = self.get_width(), self.get_height()
        buf = self.get_displayable_image()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
        yuv = img @ self.rgb2yuv.T
        yuv[..., 0] *= brightness
        out = yuv @ self.yuv2rgb.T
        out = np.clip(out, 0.0, 1.0)
        self.display_bytes = (out * 255.0 + 0.5).astype(np.uint8).tobytes()

    def refresh(self) -> None:
        if self.original_bytes:
            self.refresh_scale()
            self.refresh_brightness()
            self.refresh_channels()

    def set_scale(self, scale) -> None: 
        self.scale = int(scale)
        self.refresh()
    
    def set_brightness(self, brightness) -> None: 
        self.brightness = int(brightness)
        self.refresh()

    def set_r(self, state) -> None: 
        self.r = state
        self.refresh()

    def set_g(self, state) -> None: 
        self.g = state
        self.refresh()

    def set_b(self, state) -> None:
        self.b = state
        self.refresh()

    def get_displayable_image(self) -> bytes:
        if self.display_bytes:
            return bytes(self.display_bytes)
        return bytes()