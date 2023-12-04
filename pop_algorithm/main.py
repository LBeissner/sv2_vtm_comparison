# project file import
from sick_VisionaryT_Mini_wrapper.cam import get_data
import pop_algorithm.POPAlgorithm as POPAlgorithm
import cv2 as cv

# * settings for the data generator
# Login "service" pass "CUST_SERV"
# Default IP of Visionary Devices: 169.254.214.10:2114 - Visionary-T Mini
#
IP = "169.254.214.10:2114"
PORT = 2114
SOURCE = "ssr"
SSR_DATA = "ssr_files\H03_hip_level_mtk_filter.ssr"
LOOP_DATA = False
START_FRAME = 0
FRAME_COUNT = 0

data_generator = get_data(
    source=SOURCE,
    ip=IP,
    port=PORT,
    ssr=SSR_DATA,
    ssr_loop=LOOP_DATA,
    ssr_start=START_FRAME,
    ssr_cnt=FRAME_COUNT,
)

# setup
POPAlgorithm.setup(
    calibration_frames=100,
    on_the_run_frames=30,
    image_dimensions_xy=(512, 424),
    perception_depth=9000,
    height_thresholds_mm=(0, 2000),
    distance_threshold_px=10,
    distance_threshold_mm=10,
    relative_armpit_height=0.7,
    minimum_contour_area=80,
    filter_sensitivity=5,
)

# calibration loop
POPAlgorithm.calibration_loop(data_generator)

# main loop
for cam_data, status in data_generator:
    POPAlgorithm.detection_loop(cam_data)

    # stream results
    POPAlgorithm.show_results()

    k = cv.waitKey(30) & 0xFF
    if k == 27:
        break

cv.destroyAllWindows()
