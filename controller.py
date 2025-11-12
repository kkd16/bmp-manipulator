import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os, struct, time

from bmp import BmpImage
from compression import CMPT365_MAGIC, HEADER_STRUCT, pack_header, unpack_header, lzw_encode, lzw_decode

class Controller:
    def __init__(self, root: tk.Tk, widgets: dict, tkvars: dict):
        self.root = root
        self.w = widgets
        self.v = tkvars
        self.bmp_image: BmpImage | None = None

    def bind_handlers(self):
        self.w["browse_btn"]["command"] = self.browse_file
        self.w["compress_btn"]["command"] = self.compress_to_cmpt365
        self.w["brightness_scale"]["command"] = self.on_brightness_change
        self.w["scale_scale"]["command"] = self.on_scale_change
        self.w["r_btn"]["command"] = self.on_toggle_r
        self.w["g_btn"]["command"] = self.on_toggle_g
        self.w["b_btn"]["command"] = self.on_toggle_b

    def show_message_on_canvas(self, message: str):
        canvas = self.w["canvas"]
        canvas.delete("all")
        canvas.create_text(
            int(canvas.winfo_width() or 600) // 2,
            int(canvas.winfo_height() or 350) // 2,
            text=message,
            fill="white",
            anchor="center",
        )

    def clear_compression_stats(self):
        self.v["original_size_var"].set("—")
        self.v["compressed_size_var"].set("—")
        self.v["ratio_var"].set("—")
        self.v["time_ms_var"].set("—")

    def post_open_success(self, img: BmpImage, path: str):
        self.bmp_image = img
        e = self.w["file_path_entry"]
        e.delete(0, tk.END)
        e.insert(0, path)
        self.w["brightness_scale"].set(50)
        self.w["scale_scale"].set(50)
        self.v["r_toggle_var"].set(True)
        self.v["g_toggle_var"].set(True)
        self.v["b_toggle_var"].set(True)
        self.clear_compression_stats()
        self.display_image(img)

    def open_cmpt365_file(self, path: str):
        try:
            with open(path, "rb") as f:
                head_bytes = f.read(struct.calcsize(HEADER_STRUCT))
                hdr = unpack_header(head_bytes)
                payload = f.read()
        except Exception as e:
            self.clear_compression_stats()
            messagebox.showerror("Open Error", f"Failed to open .cmpt365:\n{e}")
            return

        if hdr["magic"] != CMPT365_MAGIC:
            self.clear_compression_stats()
            self.show_message_on_canvas("Invalid .cmpt365 file (bad magic).")
            return

        if hdr["compressed_size"] != len(payload):
            self.clear_compression_stats()
            messagebox.showerror(
                "Open Error",
                f"Invalid .cmpt365: expected {hdr['compressed_size']} bytes of payload, "
                f"got {len(payload)}.",
            )
            return

        try:
            bmp_bytes = lzw_decode(payload)
        except Exception as e:
            self.clear_compression_stats()
            messagebox.showerror("Decode Error", f"LZW decode failed:\n{e}")
            return

        errors = BmpImage.validate(bmp_bytes)
        if errors:
            self.clear_compression_stats()
            self.show_message_on_canvas(
                "Decoded payload is not a valid BMP:\n" + "\n".join(f"- {e}" for e in errors)
            )
            return

        total_size = os.path.getsize(path)
        img = BmpImage(filepath=path, bmp_bytes=bmp_bytes, file_size=total_size)
        self.post_open_success(img, path)

    def open_bmp_file(self, path: str):
        try:
            img = BmpImage.from_file(path)
        except ValueError as e:
            self.bmp_image = None
            self.w["file_path_entry"].delete(0, tk.END)
            self.v["file_size_var"].set("—")
            self.v["width_var"].set("—")
            self.v["height_var"].set("—")
            self.v["bpp_var"].set("—")
            self.clear_compression_stats()
            self.show_message_on_canvas(str(e))
            return
        except Exception as e:
            self.bmp_image = None
            self.clear_compression_stats()
            messagebox.showerror("Open Error", f"Failed to open BMP:\n{e}")
            return

        self.post_open_success(img, path)

    def browse_file(self):
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
            self.clear_compression_stats()
            messagebox.showerror("Open Error", f"Failed to open file:\n{e}")
            return

        if head.startswith(CMPT365_MAGIC):
            self.open_cmpt365_file(path)
        else:
            self.open_bmp_file(path)

    def display_image(self, bmp_obj: BmpImage):
        rgb_bytes = bmp_obj.get_displayable_image()
        width = bmp_obj.get_width()
        height = bmp_obj.get_height()

        if self.bmp_image:
            self.v["file_size_var"].set(str(self.bmp_image.get_file_size()))
            self.v["width_var"].set(str(self.bmp_image.get_original_width()))
            self.v["height_var"].set(str(self.bmp_image.get_original_height()))
            self.v["bpp_var"].set(str(self.bmp_image.get_bits_per_pixel()))

        expected_len = width * height * 3
        if rgb_bytes is None or len(rgb_bytes) != expected_len:
            print(
                f"Unexpected buffer size: got {len(rgb_bytes) if rgb_bytes else None}, expected {expected_len}"
            )
            return

        img = Image.frombytes("RGB", (width, height), rgb_bytes)
        photo = ImageTk.PhotoImage(img)
        canvas = self.w["canvas"]
        canvas.delete("all")
        canvas.config(width=width, height=height)
        canvas.image = photo
        canvas.create_image(0, 0, anchor="nw", image=photo)

    def on_brightness_change(self, value):
        if self.bmp_image:
            self.bmp_image.set_brightness(value)
            self.display_image(self.bmp_image)

    def on_scale_change(self, value):
        if self.bmp_image:
            self.bmp_image.set_scale(value)
            self.display_image(self.bmp_image)

    def on_toggle_r(self):
        if self.bmp_image:
            self.bmp_image.set_r(self.v["r_toggle_var"].get())
            self.display_image(self.bmp_image)

    def on_toggle_g(self):
        if self.bmp_image:
            self.bmp_image.set_g(self.v["g_toggle_var"].get())
            self.display_image(self.bmp_image)

    def on_toggle_b(self):
        if self.bmp_image:
            self.bmp_image.set_b(self.v["b_toggle_var"].get())
            self.display_image(self.bmp_image)

    def compress_to_cmpt365(self):
        if not self.bmp_image:
            messagebox.showwarning("No Image", "Open a BMP file first.")
            return

        src = getattr(self.bmp_image, "bmp_bytes", None)
        if not src:
            messagebox.showwarning(
                "No BMP Data",
                "Current image has no BMP source bytes to compress.",
            )
            return

        width = self.bmp_image.get_original_width()
        height = self.bmp_image.get_original_height()
        original_size = len(src)

        t0 = time.perf_counter()
        encoded = lzw_encode(src)
        t1 = time.perf_counter()

        hdr = pack_header(width, height, original_size, len(encoded))

        default_name = os.path.splitext(os.path.basename(self.bmp_image.filepath or "output"))[0] + ".cmpt365"
        save_path = filedialog.asksaveasfilename(
            defaultextension=".cmpt365",
            initialfile=default_name,
            filetypes=[("CMPT365 File", "*.cmpt365")],
        )
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

        self.v["original_size_var"].set(str(original_size))
        self.v["compressed_size_var"].set(str(compressed_os_size))
        ratio = (original_size / compressed_os_size) if compressed_os_size else float("inf")
        self.v["ratio_var"].set(f"{ratio:.3f}x")
        self.v["time_ms_var"].set(f"{(t1 - t0) * 1000:.2f}")
        messagebox.showinfo("Saved", f"Saved {save_path} successfully")
