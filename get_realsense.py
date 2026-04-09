# First import the library
import pyrealsense2 as rs # type: ignore

import numpy as np


# Create a context object. This object owns the handles to all connected realsense devices
pipeline = rs.pipeline()
pipeline.start()

try:
    while True:
        # Create a pipeline object. This object configures the streaming camera and owns it's handle
        frames = pipeline.wait_for_frames()
        
        depth = frames.get_depth_frame()
        depth_data = depth.as_frame().get_data()
        depth_image = np.asanyarray(depth_data)

        rgb = frames.get_color_frame()
        rgb_data = rgb.as_frame().get_data()
        rgb_image = np.asanyarray(rgb_data)
        print(rgb_image.shape)

finally:
    pipeline.stop()