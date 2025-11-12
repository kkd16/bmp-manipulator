import tkinter as tk

from controller import Controller 

def build_ui():
    root = tk.Tk()
    root.title("BMP Viewer - CMPT365 PA2")
    root.geometry("800x740")
    root.resizable(True, True)

    top_frame = tk.Frame(root); top_frame.pack(fill="x", padx=10, pady=(10, 6))
    canvas = tk.Canvas(top_frame, width=600, height=350, bg="grey"); canvas.pack(fill="x")

    controls_frame = tk.Frame(root); controls_frame.pack(fill="x", padx=10, pady=(0, 10))
    tk.Label(controls_frame, text="File Path:").grid(row=0, column=0, padx=(0, 8), pady=5, sticky="w")
    file_path_entry = tk.Entry(controls_frame); file_path_entry.grid(row=0, column=1, padx=0, pady=5, sticky="ew")
    browse_btn = tk.Button(controls_frame, text="Browse")
    browse_btn.grid(row=0, column=2, padx=(8, 0), pady=5, sticky="w")
    controls_frame.columnconfigure(1, weight=1)

    grid_frame = tk.Frame(root); grid_frame.pack(fill="x", padx=10, pady=(0, 8))
    file_size_var  = tk.StringVar(value="—"); width_var = tk.StringVar(value="—")
    height_var     = tk.StringVar(value="—"); bpp_var   = tk.StringVar(value="—")
    tk.Label(grid_frame, text="File Size:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    tk.Label(grid_frame, textvariable=file_size_var).grid(row=0, column=1, sticky="w", padx=5, pady=5)
    tk.Label(grid_frame, text="Width:").grid(row=0, column=2, sticky="e", padx=5, pady=5)
    tk.Label(grid_frame, textvariable=width_var).grid(row=0, column=3, sticky="w", padx=5, pady=5)
    tk.Label(grid_frame, text="Height:").grid(row=0, column=4, sticky="e", padx=5, pady=5)
    tk.Label(grid_frame, textvariable=height_var).grid(row=0, column=5, sticky="w", padx=5, pady=5)
    tk.Label(grid_frame, text="Bits Per Pixel:").grid(row=0, column=6, sticky="e", padx=5, pady=5)
    tk.Label(grid_frame, textvariable=bpp_var).grid(row=0, column=7, sticky="w", padx=5, pady=5)

    r_toggle_var = tk.BooleanVar(value=True)
    g_toggle_var = tk.BooleanVar(value=True)
    b_toggle_var = tk.BooleanVar(value=True)
    r_btn = tk.Checkbutton(grid_frame, text="R", variable=r_toggle_var)
    g_btn = tk.Checkbutton(grid_frame, text="G", variable=g_toggle_var)
    b_btn = tk.Checkbutton(grid_frame, text="B", variable=b_toggle_var)
    r_btn.grid(row=0, column=8, padx=(15, 5), pady=5, sticky="w")
    g_btn.grid(row=0, column=9, padx=5, pady=5, sticky="w")
    b_btn.grid(row=0, column=10, padx=5, pady=5, sticky="w")

    sliders_frame = tk.Frame(root); sliders_frame.pack(fill="x", padx=10, pady=(0, 8))
    tk.Label(sliders_frame, text="Brightness").grid(row=0, column=0, sticky="w", padx=(0, 8))
    brightness_scale = tk.Scale(sliders_frame, from_=0, to=100, orient="horizontal")
    brightness_scale.set(50); brightness_scale.grid(row=0, column=1, sticky="ew", padx=(0, 12))
    tk.Label(sliders_frame, text="Scale").grid(row=0, column=2, sticky="w", padx=(0, 8))
    scale_scale = tk.Scale(sliders_frame, from_=0, to=100, orient="horizontal")
    scale_scale.set(50); scale_scale.grid(row=0, column=3, sticky="ew")
    sliders_frame.columnconfigure(1, weight=1); sliders_frame.columnconfigure(3, weight=1)

    comp_frame = tk.LabelFrame(root, text="Compression (LZW)"); comp_frame.pack(fill="x", padx=10, pady=(0, 10))
    compress_btn = tk.Button(comp_frame, text="Compress (.cmpt365)")
    compress_btn.grid(row=0, column=0, padx=(0, 8), pady=8, sticky="w")

    tk.Label(comp_frame, text="Original Size:").grid(row=1, column=0, sticky="e", padx=5, pady=3)
    original_size_var = tk.StringVar(value="—")
    tk.Label(comp_frame, textvariable=original_size_var).grid(row=1, column=1, sticky="w", padx=5, pady=3)

    tk.Label(comp_frame, text="Compressed Size:").grid(row=1, column=2, sticky="e", padx=5, pady=3)
    compressed_size_var = tk.StringVar(value="—")
    tk.Label(comp_frame, textvariable=compressed_size_var).grid(row=1, column=3, sticky="w", padx=5, pady=3)

    tk.Label(comp_frame, text="Ratio:").grid(row=2, column=0, sticky="e", padx=5, pady=3)
    ratio_var = tk.StringVar(value="—")
    tk.Label(comp_frame, textvariable=ratio_var).grid(row=2, column=1, sticky="w", padx=5, pady=3)

    tk.Label(comp_frame, text="Time (ms):").grid(row=2, column=2, sticky="e", padx=5, pady=3)
    time_ms_var = tk.StringVar(value="—")
    tk.Label(comp_frame, textvariable=time_ms_var).grid(row=2, column=3, sticky="w", padx=5, pady=3)
    for i in range(4): comp_frame.columnconfigure(i, weight=1)

    widgets = dict(
        canvas=canvas,
        file_path_entry=file_path_entry,
        browse_btn=browse_btn,
        r_btn=r_btn, g_btn=g_btn, b_btn=b_btn,
        brightness_scale=brightness_scale,
        scale_scale=scale_scale,
        compress_btn=compress_btn,
    )
    tkvars = dict(
        file_size_var=file_size_var, width_var=width_var, height_var=height_var, bpp_var=bpp_var,
        r_toggle_var=r_toggle_var, g_toggle_var=g_toggle_var, b_toggle_var=b_toggle_var,
        original_size_var=original_size_var, compressed_size_var=compressed_size_var,
        ratio_var=ratio_var, time_ms_var=time_ms_var,
    )
    return root, widgets, tkvars

def main():
    root, widgets, tkvars = build_ui()
    ctrl = Controller(root=root, widgets=widgets, tkvars=tkvars)
    ctrl.bind_handlers()
    root.mainloop()

if __name__ == "__main__":
    main()
