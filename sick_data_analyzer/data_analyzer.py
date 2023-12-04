import numpy as np
import cv2
import re
import scipy
import open3d as o3d
import keyboard
from pathlib import Path
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from sick_camera_wrapper.camera import SafeVisionary2, VisionaryTMini
from sick_camera_wrapper.cam_data import CamData, CamImage
from sick_data_analyzer.model import PlaneModel
from sick_data_analyzer.color_scheme import ColorScheme
from sick_data_analyzer.visualizer import PointCloudVisualizer, Figure2D, DataSet2D

DEBUGGING: bool = True


def average(
    stack: list[np.ndarray], mode: str = "mean", fill_zero: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    options = ["mean", "median"]

    if mode not in options:
        raise ValueError(
            f"Invalid mode selected! Please select one of the following:\t{options}"
        )

    stack_array = np.asarray(stack, dtype=float)

    # replace all zero values with nan to ignore "empty" pixels
    stack_array[stack_array == 0.0] = np.nan

    if mode == "mean":
        # average and standard deviation
        stack_mean = np.nanmean(stack_array, axis=0, keepdims=True)
        stack_std = np.nanstd(stack_array, axis=0, keepdims=True)
        if fill_zero:
            stack_mean = np.nan_to_num(stack_mean)
            stack_std = np.nan_to_num(stack_std)
        return stack_mean, stack_std

    elif mode == "median":
        # median and median absolute deviation
        stack_median = np.nanmedian(stack_array, axis=0, keepdims=True)
        stack_mad = np.nanmedian(
            np.abs(stack_array - stack_median), axis=0, keepdims=True
        )
        if fill_zero:
            stack_median = np.nan_to_num(stack_median)
            stack_mad = np.nan_to_num(stack_mad)
        return stack_median, stack_mad


class DataAnalyzer(object):
    __ip: str = None
    __port: np.uint16 = None

    # record data
    __PARENT_PATH: Path = Path.cwd() / "camera_data_storage"
    __FILTER_MODES: list[str] = ["filtered", "unfiltered"]
    __DISTANCES: list[float] = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5]

    def __init__(self, ip: str = None, port: np.uint16 = None):
        self.__ip = ip
        self.__port = port

    def __camera_from_file_name(self, file_name: str):
        substrings = re.split("_", file_name)
        if substrings[0] == "sv2":
            if self.__ip is None and self.__port is None:
                return SafeVisionary2()
            if self.__ip is not None and self.__port is not None:
                return SafeVisionary2(self.__ip, self.__port)
            elif self.__ip is not None:
                return SafeVisionary2(ip=self.__ip)
            elif self.__port is not None:
                return SafeVisionary2(port=self.__port)

        elif substrings[0] == "vtm":
            if self.__ip is None and self.__port is None:
                return VisionaryTMini()
            if self.__ip is not None and self.__port is not None:
                return VisionaryTMini(self.__ip, self.__port)
            elif self.__ip is not None:
                return VisionaryTMini(ip=self.__ip)
            elif self.__port is not None:
                return VisionaryTMini(port=self.__port)
        else:
            raise ValueError(
                f"Invalid file prefix! Constructor does not recogize {substrings[0]}!"
            )

    def analyze_experimental_series(
        self,
        filter_mode: str = "filtered",
        data_mode: str = "distance",
        y_limit: np.uint16 = 100,
        correction_enabled: bool = False,
    ):

        if filter_mode not in self.__FILTER_MODES:
            raise AttributeError(
                f"Invalid filter mode selected! Select one of the following:\t{self.__FILTER_MODES}"
            )

        print("\n<< Start Analysis >>")

        sv2_pixel_set = DataSet2D(
            "safeVisionary2 Pixel Values",
            "Average Distance [mm]",
            "Deviation [mm]",
            ColorScheme.gray.hex(),
            y_ticks=None,
            y_limits=(0, y_limit),
        )
        sv2_cog_set = DataSet2D(
            "safeVisionary2 Centers of Gravity",
            "Average Distance [mm]",
            "Deviation [mm]",
            ColorScheme.yellow.hex(),
            y_ticks=None,
            y_limits=(0, y_limit),
        )

        vtm_pixel_set = DataSet2D(
            "Visionary-T Mini Pixel Values",
            "Average Distance [mm]",
            "Deviation [mm]",
            ColorScheme.gray.hex(),
            y_ticks=None,
            y_limits=(0, y_limit),
        )
        vtm_cog_set = DataSet2D(
            "Visionary-T Mini Centers of Gravity",
            "Average Distance [mm]",
            "Deviation [mm]",
            ColorScheme.aquamarine.hex(),
            y_ticks=None,
            y_limits=(0, y_limit),
        )

        camera_directories = self.__PARENT_PATH.glob("*")
        for camera_directory in camera_directories:
            camera_model = camera_directory.name

            # if DEBUGGING:
            #     if camera_model == "safeVisionary2":
            #         continue

            print(f"\n> Start to analyze {camera_model} records.")

            records = camera_directory.glob("*")

            pixel_set = DataSet2D(
                "Pixel",
                "Average Distance [mm]",
                "Deviation [mm]",
                ColorScheme.aquamarine.hex(),
                y_limits=(0, y_limit),
            )
            cog_set = DataSet2D(
                "COG",
                "Average Distance [mm]",
                "Deviation [mm]",
                ColorScheme.orange.hex(),
                y_limits=(0, y_limit),
            )

            for record in records:
                file_name = record.name

                if ("_" + filter_mode) in file_name:
                    print(f"\t> Processing record '{file_name}' ...")

                    roi, _ = self.__generate_model(file_name)

                    initial_transform = None

                    if correction_enabled:
                        initial_transform = self.__find_camera_pose(file_name)

                    # get the average distance map and the distance deviation map
                    _, distance_map_avg, distance_map_dev = self.__average_camera_data(
                        file_name,
                        frames=150,
                        roi=roi,
                        output=data_mode,
                        avg_mode="median",
                        transformation_matrix=initial_transform,
                    )

                    # calculate the cog coordinates
                    record_avg, _ = average(
                        np.ravel(distance_map_avg), mode="mean", fill_zero=False
                    )
                    record_dev, _ = average(
                        np.ravel(distance_map_dev), mode="mean", fill_zero=False
                    )

                    # add averages, deviations and cog coordinates to the data sets
                    pixel_set.add_data(distance_map_avg, distance_map_dev)
                    cog_set.add_data(record_avg, record_dev)

                    if DEBUGGING:
                        pixel_subset = DataSet2D(
                            "Pixel Subset",
                            "Average Distance [mm]",
                            "Deviation [mm]",
                            ColorScheme.aquamarine.hex(),
                            x_ticks=None,
                            x_limits=None,
                            y_limits=(0, y_limit),
                        )
                        pixel_subset.add_data(distance_map_avg, distance_map_dev)
                        cog_subset = DataSet2D(
                            "COG Subset",
                            "Average Distance [mm]",
                            "Deviation [mm]",
                            ColorScheme.orange.hex(),
                            x_ticks=None,
                            x_limits=None,
                            y_limits=(0, y_limit),
                        )
                        cog_subset.add_data(record_avg, record_dev)

                        print(file_name)
                        print(cog_subset)

                        figure = Figure2D(window_title=f"{camera_model}")
                        figure.plot_datasets_to_axis(
                            0, 0, [pixel_subset, cog_subset], "Pixel Deviations & COGs"
                        )
                        figure.plot()
                        figure.clear()

                    print(f"\t> Record '{file_name}' analyzed.")

            print(f"> All {camera_model} records analyzed.")

            if camera_model == "safeVisionary2":
                sv2_pixel_set += pixel_set
                sv2_cog_set += cog_set
            elif camera_model == "Visionary-T Mini":
                vtm_pixel_set += pixel_set
                vtm_cog_set += cog_set

        print(sv2_cog_set)
        print(vtm_cog_set)

        comparison_figure = Figure2D(
            window_title=f"Camera Comparison {filter_mode}", rows=1, columns=2
        )

        sv2_pixel_set.y_limits = (0, y_limit)
        sv2_cog_set.y_limits = (0, y_limit)
        vtm_pixel_set.y_limits = (0, y_limit)
        vtm_cog_set.y_limits = (0, y_limit)

        comparison_figure.plot_datasets_to_axis(0, 0, [sv2_pixel_set, sv2_cog_set])
        comparison_figure.plot_datasets_to_axis(0, 1, [vtm_pixel_set, vtm_cog_set])

        comparison_figure.plot()

        sv2_p = sv2_cog_set.add_trendline()
        vtm_p = vtm_cog_set.add_trendline()

        x = np.linspace(0, 14000)

        sv2_trendline = DataSet2D(
            "Trendline",
            "Average Distance [mm]",
            "Deviation [mm]",
            ColorScheme.light_gray.hex(),
            x_data=x,
            y_data=np.exp(sv2_p(x)),
            x_ticks=None,
            x_limits=None,
            y_limits=None,
            marker="",
            linestyle="dashed",
        )

        vtm_trendline = DataSet2D(
            "Trendline",
            "Average Distance [mm]",
            "Deviation [mm]",
            ColorScheme.gray.hex(),
            x_data=x,
            y_data=np.exp(vtm_p(x)),
            x_ticks=None,
            x_limits=None,
            y_limits=None,
            marker="",
            linestyle="dashed",
        )

        trend_figure = Figure2D(
            window_title=f"Camera Comparison Trendlines {filter_mode}",
            rows=1,
            columns=1,
        )
        trend_figure.plot_datasets_to_axis(
            0,
            0,
            [sv2_trendline, vtm_trendline, sv2_cog_set, vtm_cog_set],
            log_scale_y=True,
        )

        trend_figure.plot()

        print(np.round(np.array(sv2_cog_set.x_data), 2))
        print(np.round(np.array(sv2_cog_set.y_data), 2), "\n")
        print(np.round(np.array(vtm_cog_set.x_data), 2))
        print(np.round(np.array(vtm_cog_set.y_data), 2))

        print("\n<< End Analysis >>\n")

    def visualize_point_cloud(
        self,
        file_name: str,
        loop: bool = True,
        show_roi: bool = True,
        overlay_model: bool = True,
        roi_border: np.uint8 = 0,
        resolution: float = 2,
    ):
        visualizer = PointCloudVisualizer()

        camera = self.__camera_from_file_name(file_name)

        if show_roi or overlay_model:
            if overlay_model:
                roi, model_points = self.__generate_model(file_name, border=roi_border)
            else:
                roi = self.__select_roi(file_name)

        if show_roi:
            data_generator = camera.play_file(
                file_name=file_name, output_mode="Point Cloud", loop=loop, roi=roi
            )
        else:
            data_generator = camera.play_file(
                file_name=file_name, output_mode="Point Cloud", loop=loop
            )

        for point_cloud_points, _ in data_generator:

            point_clouds: dict = {}

            # highlight the largest plane on full point clouds
            if not show_roi:
                # segment inliers and outliers of the largest plane detected in the point cloud
                inlier_points, outlier_points = visualizer.segment_plane(
                    point_cloud_points
                )
                point_clouds.update(
                    {"inliers": inlier_points, "outliers": outlier_points}
                )
            else:
                point_clouds.update({"point cloud": point_cloud_points})

            if overlay_model:
                point_clouds.update({"model": model_points})

            terminate = visualizer.show_point_clouds(point_clouds, resolution)

            # break loop on window closure
            if terminate:
                break

    def visualize_maps(
        self,
        file_name: str,
        loop: bool = True,
        show_roi: bool = True,
        overlay_model: bool = False,
        roi_border: np.uint8 = 2,
    ):

        camera = self.__camera_from_file_name(file_name)

        if show_roi or overlay_model:
            if overlay_model:
                roi, _ = self.__generate_model(file_name, border=roi_border)
            else:
                roi = self.__select_roi(file_name)

        if show_roi:
            data_generator = camera.play_file(
                file_name=file_name, output_mode="CamData", loop=loop, roi=roi
            )
        else:
            data_generator = camera.play_file(
                file_name=file_name, output_mode="CamData", loop=loop
            )

        for camera_data, _ in data_generator:
            camera_image = CamImage.from_cam_data(cam_data=camera_data)

            print(camera_image)

            cv2.imshow("Depth Image", camera_image.depth())
            cv2.imshow("Intensity Image", camera_image.intensity(add_cross_hair=True))
            cv2.imshow("Status Image", camera_image.state())

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                break

    def __generate_model(self, file_name: str, border: np.uint8 = 0):

        # initialize camera and the data generator
        camera = self.__camera_from_file_name(file_name)
        data_generator = camera.play_file(
            file_name=file_name, output_mode="CamData", loop=False
        )

        # get one frame to retrieve intrinsic camera parameters
        camera_data, _ = next(data_generator)
        principal_point = camera_data.get_principal_point()

        # construct a plane model and calculate its roi and 3d points
        model = PlaneModel.from_file_name(file_name)
        roi = model.roi(principal_point, border)
        model_points = model.points()

        return roi, model_points

    def __select_roi(self, file_name: str) -> np.ndarray:
        """
            Opens the region of interest selection dialogue and returns its boundaries. The ROI needs to include the principal point of the image.

        Args:
            image (np.ndarray): The image to be shown in the selection window.
            image_cp (tuple[int, int]): The image principal point.

        Returns:
            np.ndarray: The top left and bottom right vertices of the rectangular ROI arranged like [[x0 x1] [y0 y1]].
        """

        camera = self.__camera_from_file_name(file_name)

        data_generator = camera.play_file(
            file_name=file_name, output_mode="CamData", loop=False
        )
        camera_data, _ = next(data_generator)

        camera_images: CamImage = CamImage.from_cam_data(camera_data)
        image = camera_images.intensity(add_cross_hair=True)

        roi_vertices: np.ndarray = None

        while True:

            # selection window
            window_name = "Select the Region of Interest:"
            print("The ROI must be selected around the principal point!")
            roi = cv2.selectROI(window_name, image, showCrosshair=False)
            cv2.destroyWindow(window_name)

            # origin (top left vertex) x & y, roi width & height
            x0, y0, roi_width, roi_height = roi

            # bottom right vertex
            x1, y1 = (x0 + roi_width), (y0 + roi_height)

            # image principal point
            cx, cy = camera_images.principal_point

            # condition that the principal point is located in the roi
            cp_in_roi = x0 < cx and cx < x1 and y0 < cy and cy < y1

            # reopen selection window, if invalid ROI is selected
            if not cp_in_roi:
                print("Selected ROI invalid!")

            # exit program, if selection
            if keyboard.is_pressed("c") or keyboard.is_pressed("C"):
                exit()

            # pack the top left and bottom right vertices of the ROI and
            # calculate the position of the principal point in the snippet
            else:
                roi_vertices = np.array([[x0, x1], [y0, y1]], dtype=np.uint16)
                break

        return roi_vertices

    def __average_camera_data(
        self,
        file_name: str,
        frames: np.uint16,
        roi: np.ndarray = None,
        output: str = "depth",
        avg_mode: str = "mean",
        fill_zero: bool = True,
        return_stack: bool = False,
        transformation_matrix: np.ndarray = None,
    ) -> tuple[CamData, list, list]:

        options = ["depth", "distance", "intensity"]
        if output not in options:
            raise ValueError(f"No valid output selected! Select one from:\t{options}")

        camera = self.__camera_from_file_name(file_name)

        # initialise data generator
        data_generator = camera.play_file(
            file_name=file_name, output_mode="CamData", roi=roi, loop=True
        )
        depth_stack = []
        intensity_stack = []

        frame = 0
        for camera_data, _ in data_generator:
            if output == "distance":
                camera_data.depth_to_distance_map(transformation=transformation_matrix)
            depth_stack.append(camera_data.dist_img)
            intensity_stack.append(camera_data.ints_img)

            frame += 1

            if frame == frames:
                break

        depth_avg, depth_div = average(depth_stack, avg_mode, fill_zero)
        intensity_avg, intensity_div = average(intensity_stack, avg_mode, fill_zero)

        camera_data.dist_img = depth_avg
        camera_data.ints_img = intensity_avg

        if return_stack:
            return camera_data, depth_stack, intensity_stack

        elif output == "depth" or output == "distance":
            return camera_data, depth_avg, depth_div

        elif output == "intensity":
            return camera_data, intensity_avg, intensity_div

    def __check_pixel_normal_distribution(
        self, image_stack: list[np.ndarray], pthreshold: float = 0.75
    ):

        pixel_stack = np.asarray(image_stack)
        pixel_stack = pixel_stack.T.reshape((-1, pixel_stack.shape[0]))

        normal_distributed = 0
        not_normal_distributed = 0
        total = 0

        for pixel in pixel_stack:
            sample = pixel[pixel != 0]
            if sample.size >= 20:
                _, pvalue = scipy.stats.normaltest(sample)
                if pvalue > pthreshold:
                    normal_distributed += 1
                else:
                    not_normal_distributed += 1
                total += 1

        print(
            f"{normal_distributed} of {total} Pixels normal distributed:\t{normal_distributed / total} %"
        )

    def __find_camera_pose(self, file_name: str, frames: np.uint16 = 150):

        # roughly define a region of interest
        roi, _ = self.__generate_model(file_name, border=2)

        # get average camera data and
        avg_camera_data, _, _ = self.__average_camera_data(
            file_name, frames=frames, output="depth", avg_mode="median", roi=roi
        )
        avg_camera_image = CamImage.from_cam_data(avg_camera_data)
        binary_image = avg_camera_image.apply_otsu_threshold()

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            cv2.destroyAllWindows()

        # get outmost contours
        contours, _ = cv2.findContours(
            binary_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
        )

        # find largest contour
        target_contour = contours[0]
        for contour in contours:
            if cv2.contourArea(contour) > cv2.contourArea(target_contour):
                target_contour = contour

        approx = []
        k = 0.01

        while len(approx) != 4:
            epsilon = k * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            k += 0.01

            if k == 10:
                break

        if len(approx) != 4:
            raise ValueError("Invalid number of contour vertices!")

        index = np.flip(np.squeeze(approx), axis=1)
        bitmask = np.zeros(binary_image.shape, np.float32)
        bitmask[index[:, 0], index[:, 1]] = 1

        image_coordinates = np.flip(np.array(np.nonzero(bitmask)).T, axis=1)

        avg_camera_data.dist_img *= bitmask
        point_cloud = avg_camera_data.point_cloud()

        world_coordinates = point_cloud[point_cloud[:, 2] > 0]

        if world_coordinates.shape[0] > 3:

            _, r_vector, t_vector = cv2.solvePnP(
                objectPoints=world_coordinates.astype(np.float32),
                imagePoints=image_coordinates.astype(np.float32),
                cameraMatrix=avg_camera_data.get_camera_matrix(),
                distCoeffs=avg_camera_data.get_distortion_vector(),
                flags=cv2.SOLVEPNP_AP3P,
            )

            transformation_matrix = np.identity(4)
            transformation_matrix[:3, :3] = Rotation.from_rotvec(
                np.squeeze(r_vector)
            ).as_matrix()
            transformation_matrix[:3, 3] = np.squeeze(t_vector)

            if DEBUGGING:
                for i in range(world_coordinates.shape[0]):
                    print(image_coordinates[i, :])
                    print(world_coordinates[i, :])

                image = avg_camera_image.intensity(target_shape=None)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                image = cv2.drawContours(
                    image, [approx], -1, ColorScheme.orange.ocv_tuple(), 1
                )
                avg_camera_image.set_intensity(image)
                cv2.imshow("Depth Image", avg_camera_image.depth())
                cv2.imshow("Intensity Image", avg_camera_image.intensity())

                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    cv2.destroyAllWindows()

            return transformation_matrix
        else:
            return np.eye(4)

    def compare_point_clouds(
        self,
        distance: float,
        filter_mode: str,
        frames: np.uint16 = 150,
        resolution: float = 2.0,
        roi_border: np.uint8 = 1,
    ):

        if filter_mode not in self.__FILTER_MODES:
            raise AttributeError(
                f"Invalid filter mode selected! Select one of the following:\t{self.__FILTER_MODES}"
            )

        if distance not in self.__DISTANCES:
            raise AttributeError(
                f"Invalid distance selected! Select one of the following:\t{self.__DISTANCES}"
            )

        if DEBUGGING:
            visualizer = PointCloudVisualizer()

        sv2_camera = SafeVisionary2()
        vtm_camera = VisionaryTMini()

        # setup file names
        distance_text = re.sub("[.]", ",", str(distance))
        sv2_file_name = f"sv2_record_{distance_text}m_{filter_mode}"
        vtm_file_name = f"vtm_record_{distance_text}m_{filter_mode}"

        sv2_transform = self.__find_camera_pose(sv2_file_name)
        vtm_transform = self.__find_camera_pose(vtm_file_name)

        sv2_roi, _ = self.__generate_model(sv2_file_name, border=roi_border)
        vtm_roi, _ = self.__generate_model(vtm_file_name, border=roi_border)

        # initialize data generators
        sv2_data_generator = sv2_camera.play_file(
            file_name=sv2_file_name,
            output_mode="Point Cloud",
            roi=sv2_roi,
            loop=True,
        )

        vtm_data_generator = vtm_camera.play_file(
            file_name=vtm_file_name,
            output_mode="Point Cloud",
            roi=vtm_roi,
            loop=True,
        )

        sv2_point_cloud = o3d.geometry.PointCloud()
        vtm_point_cloud = o3d.geometry.PointCloud()

        avg_frame_distances = []

        for _ in range(frames):

            sv2_point_cloud_points, _ = next(sv2_data_generator)
            vtm_point_cloud_points, _ = next(vtm_data_generator)

            (
                sv2_point_cloud_points,
                vtm_point_cloud_points,
            ) = self.__match_point_cloud_sizes(
                sv2_point_cloud_points, vtm_point_cloud_points
            )

            sv2_point_cloud_points = np.c_[
                sv2_point_cloud_points, np.ones(sv2_point_cloud_points.shape[0])
            ]
            sv2_point_cloud_points = (sv2_transform @ sv2_point_cloud_points.T).T[:, :3]

            vtm_point_cloud_points = np.c_[
                vtm_point_cloud_points, np.ones(vtm_point_cloud_points.shape[0])
            ]
            vtm_point_cloud_points = (vtm_transform @ vtm_point_cloud_points.T).T[:, :3]

            if DEBUGGING:
                print(sv2_point_cloud_points.shape, vtm_point_cloud_points.shape)

            print(sv2_point_cloud_points.shape, vtm_point_cloud_points.shape)

            transformation = self.__register_point_clouds(
                source_points=vtm_point_cloud_points,
                target_points=sv2_point_cloud_points,
                threshold=205.0,
                initial_transform=np.eye(4),
            )

            sv2_point_cloud.points = o3d.utility.Vector3dVector(sv2_point_cloud_points)
            sv2_point_cloud.transform(transformation)
            vtm_point_cloud.points = o3d.utility.Vector3dVector(vtm_point_cloud_points)

            distances_sv2_to_vtm = sv2_point_cloud.compute_point_cloud_distance(
                vtm_point_cloud
            )
            distances_vtm_to_sv2 = vtm_point_cloud.compute_point_cloud_distance(
                sv2_point_cloud
            )

            avg_distance_sv2_to_vtm = np.average(distances_sv2_to_vtm)
            avg_distance_vtm_to_sv2 = np.average(distances_vtm_to_sv2)

            avg_frame_distances.append(
                [avg_distance_sv2_to_vtm, avg_distance_vtm_to_sv2]
            )

            if DEBUGGING:
                _ = visualizer.show_point_clouds(
                    {"sv2": sv2_point_cloud_points, "vtm": vtm_point_cloud_points},
                    resolution=resolution,
                )

        result = np.nanmean(np.array(avg_frame_distances), axis=0)

        print(np.around(result, 2))

    @staticmethod
    def __register_point_clouds(
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: np.ndarray,
        threshold: float,
    ):
        """_summary_

        Args:
            source (o3d.geometry.PointCloud): _description_
            target (o3d.geometry.PointCloud): _description_
            threshold (float): _description_
            log (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_points)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_points)

        evaluation = o3d.pipelines.registration.evaluate_registration(
            source=source,
            target=target,
            max_correspondence_distance=threshold,
            transformation=initial_transform,
        )

        registration_p2p = o3d.pipelines.registration.registration_icp(
            source=source,
            target=source,
            max_correspondence_distance=threshold,
            init=initial_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        if DEBUGGING:
            print(
                "\n<< Point Cloud Registration >>"
                + f"\n\tEvaluation Result:\tFitness:\t{evaluation.fitness}"
                + f"\n\t\t\t\tRMSE:\t\t{evaluation.inlier_rmse}"
                + f"\n\tTransformation:\t\t{registration_p2p.transformation[0,:]}"
                + f"\n\t\t\t\t{registration_p2p.transformation[1,:]}"
                + f"\n\t\t\t\t{registration_p2p.transformation[2,:]}"
                + f"\n\t\t\t\t{registration_p2p.transformation[3,:]}"
                + "\n"
            )

        return registration_p2p.transformation

    @staticmethod
    def __match_point_cloud_sizes(source_points: np.ndarray, target_points: np.ndarray):
        source_lt_zero = np.asarray(source_points[:, 2] <= 0).nonzero()[0]
        target_lt_zero = np.asarray(target_points[:, 2] <= 0).nonzero()[0]

        deletion_indices = np.unique(np.hstack([source_lt_zero, target_lt_zero]))
        matched_source_points = np.delete(source_points, deletion_indices, axis=0)
        matched_target_points = np.delete(target_points, deletion_indices, axis=0)

        return matched_source_points, matched_target_points
