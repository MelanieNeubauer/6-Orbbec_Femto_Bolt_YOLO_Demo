from pyorbbecsdk import *
import open3d as o3d

class CameraInterface:
    def __init__(self):
        self.pipeline = Pipeline()
        self.config = Config()
        self.color_profile = None
        self.intrinsics = None
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    def print_default_camera_settings(self, device):
        print("\n=== DEFAULT CAMERA SETTINGS ===")
        props_to_check = [
            OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT,
            OBPropertyID.OB_PROP_COLOR_GAIN_INT,
            OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
            OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT,
            OBPropertyID.OB_PROP_IR_GAIN_INT,
            OBPropertyID.OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL,
        ]
        for prop_id in props_to_check:
            try:
                val = device.get_int_property(prop_id)
                print(f"{prop_id.name}: {val}")
            except Exception:
                try:
                    val = device.get_bool_property(prop_id)
                    print(f"{prop_id.name}: {val}")
                except Exception:
                    print(f"{prop_id.name}: Not supported")

        print("=== END SETTINGS ===\n")

    def setup_streams(self):
        # Setup color stream (prefer RGB format)
        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        self.color_profile = color_profiles.get_default_video_stream_profile()
        
        # Setup depth stream with default profile
        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()

        device = self.pipeline.get_device()
        # self.print_supported_properties(device)
        # self.print_default_camera_settings(device)
        # print("=== OBPropertyID Members ===")
        # for attr in dir(OBPropertyID):
        #    if not attr.startswith("__"):
        #        val = getattr(OBPropertyID, attr)
        #        print(f"{attr}: {val}")


        # --- Configure color stream properties ---
        try:
            device.set_bool_property(OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, False)  # Disable auto exposure
        except OBError:
            print("[WARNING] Could not disable color auto exposure")

        try:
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT, 100)  # Set manual exposure in microseconds
        except OBError:
            print("[WARNING] Could not set color exposure")

        try:
            device.set_int_property(OBPropertyID.OB_PROP_COLOR_GAIN_INT, 64)  # Set manual gain
        except OBError:
            print("[WARNING] Could not set color gain")

        print("[INFO] Color stream exposure and gain set.")

        # --- Configure depth stream properties ---
        #try:
        #    device.set_int_property(OBPropertyID.OB_PROP_DEPTH_EXPOSURE_INT, 10000)  # microseconds
        #except OBError:
        #    print("[WARNING] Could not set depth exposure")

        #try:
        #    device.set_int_property(OBPropertyID.OB_PROP_DEPTH_GAIN_INT, 16)         # gain value
        #except OBError:
        #    print("[WARNING] Could not set depth gain")

        #try:
        #    device.set_int_property(OBPropertyID.OB_PROP_IR_EXPOSURE_INT, 2000)      # microseconds
        #except OBError:
        #    print("[WARNING] Could not set IR exposure")

        #try:
        #    device.set_int_property(OBPropertyID.OB_PROP_IR_GAIN_INT, 24)            # gain value
        #except OBError:
        #    print("[WARNING] Could not set IR gain")

        #print("[INFO] Depth and IR exposure/gain successfully configured.")

        self.config.enable_stream(self.color_profile)
        self.config.enable_stream(depth_profile)
        # Start the pipeline with the configured streams
        # Start the pipeline with the configured streams

        # Get the depth sensor to apply hardware alignment settings
        self.config.set_align_mode(OBAlignMode.SW_MODE)
        self.pipeline.enable_frame_sync()
        self.pipeline.start(self.config)
        self.print_default_camera_settings(device)

        # Retrieve intrinsics from first frame
        frames = self.pipeline.wait_for_frames(1000)
        if not frames:
            raise RuntimeError("Unable to retrieve frames for intrinsics.")

        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("Unable to retrieve color frame for intrinsics.")

        color_frame = color_frame.as_video_frame()
        color_profile = color_frame.get_stream_profile().as_video_stream_profile()
        color_intrinsics = color_profile.get_intrinsic()

        self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=color_intrinsics.width,
            height=color_intrinsics.height,
            fx=color_intrinsics.fx,
            fy=color_intrinsics.fy,
            cx=color_intrinsics.cx,
            cy=color_intrinsics.cy
        )
        print("[INFO] Camera intrinsics retrieved successfully.")


    def get_frames(self):
        frames = self.pipeline.wait_for_frames(1000)
        if frames:
            aligned_frames = self.align_filter.process(frames).as_frame_set()
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if color_frame and depth_frame:
                return color_frame, depth_frame
        return None, None

    def get_intrinsics(self):
        if self.intrinsics is None:
            raise RuntimeError("Intrinsics not initialized. Call setup_streams() first.")
        return self.intrinsics

    def stop(self):
        self.pipeline.stop()
