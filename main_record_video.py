import cv2
import numpy as np
import time
from pyorbbecsdk import *

# ======== SETUP CAMERA ========
pipeline = Pipeline()
config = Config()

color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
color_profile = color_profiles.get_default_video_stream_profile()

depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
depth_profile = depth_profiles.get_default_video_stream_profile()

config.enable_stream(color_profile)
config.enable_stream(depth_profile)
config.set_align_mode(OBAlignMode.SW_MODE)

pipeline.enable_frame_sync()
pipeline.start(config)

# Setup exposure
device = pipeline.get_device()
device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)
device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 5000)


print("[INFO] Camera started. Recording RGB stream...")

# ======== GET FRAME SIZE ========
frames = pipeline.wait_for_frames(1000)
color_frame = frames.get_color_frame().as_video_frame()
width, height = color_frame.get_width(), color_frame.get_height()

# ======== VIDEO WRITER ========
fps = 30
video_path = "dataset/videos/orbbec_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))


# ======== RECORD LOOP ========
duration = 5  # seconds
start_time = time.time()

try:
    while time.time() - start_time < duration:
        frames = pipeline.wait_for_frames(1000)
        if not frames:
            continue

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        fmt = color_frame.get_format()
        frame_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)

        # --- Decode depending on format ---
        if fmt == OBFormat.RGB:
            frame = frame_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
        elif fmt == OBFormat.BGR:
            frame = frame_data.reshape((color_frame.get_height(), color_frame.get_width(), 3))
        elif fmt == OBFormat.YUYV:
            yuyv = frame_data.reshape((color_frame.get_height(), color_frame.get_width(), 2))
            frame = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
        elif fmt == OBFormat.MJPG:
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
        else:
            print(f"[WARNING] Unsupported format: {fmt}, skipping frame.")
            continue

        out.write(frame)
except KeyboardInterrupt:
    print("[INFO] Recording stopped manually.")
finally:
    out.release()
    pipeline.stop()
    print(f"[INFO] Video saved to '{video_path}'")
