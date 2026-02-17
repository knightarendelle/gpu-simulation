import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import shutil

if os.path.exists("out/renders"):
    shutil.rmtree("out/renders")

if os.path.exists("out/nbody.gif"):
    os.remove("out/nbody.gif")

# ---------- Frame Renderer ----------

def render_frame(csv_path, out_png, lim=10.0):
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)

    x = data[:, 0]
    y = data[:, 1]

    plt.figure(figsize=(8, 8))
    plt.style.use("dark_background")

    # Black background explicitly
    ax = plt.gca()
    ax.set_facecolor("black")

    plt.scatter(
        x, y,
        s=14,                # slightly larger balls
        c="#00ff66",         # neon green
        alpha=0.75,
        edgecolors="none"
    )

    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)

    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.savefig(out_png, dpi=200, facecolor="black")
    plt.close()


# ---------- GIF Builder ----------

def main():
    frame_paths = sorted(glob.glob("out/frames/frame_*.csv"))

    if not frame_paths:
        print("No frames found in out/frames/")
        return

    os.makedirs("out/renders", exist_ok=True)

    png_paths = list()

    print("Rendering frames...")
    for csv_path in frame_paths:
        png_path = os.path.join(
            "out/renders",
            os.path.basename(csv_path).replace(".csv", ".png")
        )

        render_frame(csv_path, png_path)
        png_paths.append(png_path)

    print("Building GIF...")
    images = [Image.open(p) for p in png_paths]

    images[0].save(
        "out/nbody.gif",
        save_all=True,
        append_images=images[1:],
        duration=160,  # lower = faster animation
        loop=0,
        optimize=True
    )

    print("Done. Output: out/nbody.gif")


if __name__ == "__main__":
    main()
