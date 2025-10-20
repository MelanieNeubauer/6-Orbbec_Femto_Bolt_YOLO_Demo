import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import open3d as o3d

from camera_interface import CameraInterface
from utils import frame_to_bgr_image


class MultiStreamViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Orbbec Live Viewer – RGB / Depth / Mask")

        self.cam = CameraInterface()
        self.cam.setup_streams()

        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        self.update_video()

    def update_video(self):
        try:
            color_frame, depth_frame = self.cam.get_frames()

            if color_frame is not None and depth_frame is not None:
                # === RGB image ===
                rgb = frame_to_bgr_image(color_frame)

                # === Depth colormap ===
                depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    (depth_frame.get_height(), depth_frame.get_width())
                )
                depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # === Fake mask preview (for demo, threshold on depth) ===
                mask = (depth > 0).astype(np.uint8) * 255
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # === Combine three streams side by side ===
                h, w = rgb.shape[:2]
                target_size = (480, 360)  # resize all for display
                combined = np.hstack([
                    cv2.resize(rgb, target_size),
                    cv2.resize(depth_colored, target_size),
                    cv2.resize(mask_bgr, target_size)
                ])

                # Convert to Tkinter image
                img = Image.fromarray(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
        except Exception as e:
            print(f"[ERROR] update_video failed: {e}")

        self.root.after(30, self.update_video)


if __name__ == "__main__":
    root = tk.Tk()
    app = MultiStreamViewer(root)
    root.mainloop()
