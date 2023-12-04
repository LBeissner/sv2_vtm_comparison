# project file import
from sick_data_analyzer.data_analyzer import DataAnalyzer

# * settings for the data generator
# Login "service" pass "CUST_SERV"
# Default IP of Visionary Devices: 169.254.214.10:2114 - Visionary-T Mini
# ip address for comparison with safeVisionary2: 192.168.1.2:2114

FILE = r"sv2_record_13,5m_filtered"


analyzer = DataAnalyzer()

# analyzer.visualize_point_cloud(file_name=FILE, resolution=3)
# analyzer.visualize_maps(file_name=FILE, show_roi=True, overlay_model=True, roi_border=5)
analyzer.analyze_experimental_series(
    filter_mode="filtered", data_mode="distance", y_limit=100, correction_enabled=True
)

# analyzer.compare_point_clouds(
#     distance=9.5, filter_mode="filtered", frames=150, roi_border=1
# )
