from __future__ import annotations

import open3d as o3d
import numpy as np
import keyboard
from sick_data_analyzer.color_scheme import ColorScheme
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from dataclasses import dataclass, field
from typing import Union
from math import floor, ceil


class PointCloudVisualizer:
    __visualizer: o3d.visualization.Visualizer
    __point_clouds: dict[str, o3d.geometry.PointCloud] = {}
    __terminated: bool = False
    __update_called: bool = False

    def __init__(self, debug_mode: bool = False, resolution: float = 0.5):

        if debug_mode:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

        # initialize visualizer
        self.__visualizer = o3d.visualization.Visualizer()

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=100.0, origin=[0, 0, 0]
        )
        self.__visualizer.add_geometry(
            mesh_frame,
        )

    def __add_point_cloud(self, key: str, pcl_points: np.ndarray[np.float64]):
        """
            Creates a new open3d point cloud object and stores it to the update queue.
            This needs to be called before the first update_renderer() call.

        Args:
            pcl_points (np.ndarray[np.float64]): NumPy array of dimensions (n, 3) containing the point cloud points ordered xyz.
        """
        if not self.__update_called:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(pcl_points)
            self.__visualizer.add_geometry(point_cloud, True)
            self.__point_clouds.update({key: point_cloud})

    def __update_point_cloud(self, key: str, pcl_points: np.ndarray[np.float64]):
        """
            Updates the points wrapped in the selected point cloud. This method needs to
            be called, if changes to the point cloud were made and need to be visualized.

        Args:
            key (str): Key of the point cloud to be updated. The key is selected when calling add_point_cloud().
            pcl_points (np.ndarray[np.float64]): The modified or new set of points to be stored to the point cloud object.
        """
        self.__point_clouds[key].points = o3d.utility.Vector3dVector(pcl_points)

    def __paint_point_cloud(self, key: str, color: np.ndarray[np.float64]):
        """
            Paints all points in the selected point cloud uniformely. The color is specified by a rgb color vector.

        Args:
            key (str): Key of the point cloud to be updated. The key is selected when calling add_point_cloud().
            color (np.ndarray[np.float64]): The rgb color vector to paint the points.
        """
        self.__point_clouds[key].paint_uniform_color(color)

    def __update_renderer(self) -> bool:
        """
            Updates the open3d renderer and keeps the coordinate system visible during frame change.
            It also contains contains the query of the "close window" key q. This method needs to be
            called after all add_point_cloud(), update_point_cloud() and paint_point_cloud() calls.

        Returns:
            bool: Boolean for loop termination.
        """

        for point_cloud in self.__point_clouds.values():
            self.__visualizer.update_geometry(point_cloud)

        window_active = self.__visualizer.poll_events()
        self.__visualizer.update_renderer()

        # boolean to prevent excution of add_point_cloud() after the first update_renderer() call
        self.__update_called = True

        if not window_active or keyboard.is_pressed("q"):
            self.__terminated = True
            self.__visualizer.destroy_window()

        return self.__terminated

    @staticmethod
    def segment_plane(point_cloud_points: np.ndarray):

        # filter points in or behind the camera lense (z <= 0)
        # these points interfere with the plane segmentation algorithm
        points = point_cloud_points[point_cloud_points[:, 2] > 0]

        # create a point cloud object
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # use RANSAC to find the largest plane in the point clouds bottom half
        _, inlier_indices = point_cloud.segment_plane(
            distance_threshold=50, ransac_n=5, num_iterations=1000
        )  # 50, 5, 1000

        # divide point cloud in inliers and outliers
        inlier_point_cloud = point_cloud.select_by_index(inlier_indices)
        outlier_point_cloud = point_cloud.select_by_index(inlier_indices, invert=True)

        inlier_points = np.asarray(inlier_point_cloud.points)
        outlier_points = np.asarray(outlier_point_cloud.points)

        return inlier_points, outlier_points

    def show_point_clouds(
        self, point_clouds: dict[str, np.ndarray], resolution: float = 0.5
    ):

        # open the display window
        self.__visualizer.create_window(
            "Pointcloud from Depth Frame:\tPress 'q' to exit!"
        )

        # option class for setting display background color and point size
        option = self.__visualizer.get_render_option()
        option.background_color = ColorScheme.anthracite.o3d_vector()
        option.point_size = resolution

        color_table = ColorScheme.table()

        i = 0
        for key, points in point_clouds.items():
            if i == len(color_table):
                print(
                    f"[WARNING] Not enough colors for the number of point clouds entered! Affected point clouds are displayed in {str(color_table[-1])}"
                )
                i = -1
                break
            self.__add_point_cloud(key, points)
            self.__update_point_cloud(key, points)
            self.__paint_point_cloud(key, color_table[i].o3d_vector())
            i += 1

        return self.__update_renderer()


@dataclass
class DataSet2D(object):
    title: str
    x_label: str
    y_label: str

    color: str
    x_limits: tuple[float, float] = (0, 14000)
    x_ticks: list[int] = field(
        default_factory=lambda: [1500, 3500, 5500, 7500, 9500, 11500, 13500]
    )
    y_limits: tuple[float, float] = (0, 100)
    y_ticks: list[int] = field(default_factory=list)
    marker: str = "."
    linestyle: str = ""

    x_data: list = field(default_factory=list)
    y_data: list = field(default_factory=list)

    def __add__(self, other: DataSet2D):
        merged_set = DataSet2D(
            title=self.title,
            x_label=self.x_label,
            y_label=self.y_label,
            color=self.color,
            x_limits=(
                min(self.x_limits[0], other.x_limits[0]),
                max(self.x_limits[1], other.x_limits[1]),
            ),
            y_limits=(
                min(self.y_limits[0], other.y_limits[0]),
                max(self.y_limits[1], other.y_limits[1]),
            ),
            marker=self.marker,
            linestyle=self.linestyle,
        )
        merged_set.add_data(self.x_data, self.y_data)
        merged_set.add_data(other.x_data, other.y_data)

        return merged_set

    def format_x_axis(self, x_limits: tuple[int, int], x_ticks: list[int]):
        self.x_limits = x_limits
        self.x_ticks = x_ticks

    def format_y_axis(self, y_limits: tuple[int, int], y_ticks: list[int]):
        self.y_limits = y_limits
        self.y_ticks = y_ticks

    def __str__(self):
        return (
            f"\n<< DataSet >>"
            + f"\n\tName:\t{self.title}"
            + f"\n\tColor:\t{str(self.color)}"
            + f"\n\tLabels:\tX: {self.x_label}"
            + f"\n\t\tY: {self.y_label}"
            + f"\n\tRanges:\tX: {floor(min(self.x_data))} - {ceil(max(self.x_data))}"
            + f"\n\t\tY: {floor(min(self.y_data))} - {ceil(max(self.y_data))}"
            + "\n"
        )

    def add_data(
        self, x_data: Union[np.ndarray, list], y_data: Union[np.ndarray, list]
    ):

        if isinstance(x_data, list) and isinstance(y_data, list):
            x_data = np.array(x_data)
            y_data = np.array(y_data)

        elif isinstance(x_data, np.ndarray) and isinstance(y_data, np.ndarray):
            x_data = np.ravel(x_data)
            y_data = np.ravel(y_data)
        else:
            raise ValueError(
                f"X and Y data need to be of the same input type! They currently are:\nX: {type(x_data)}\tY: {type(y_data)} "
            )

        (x_non_nan,) = np.asarray(x_data != np.nan).nonzero()
        self.x_data += x_data[x_non_nan].tolist()
        self.y_data += y_data[x_non_nan].tolist()

    def delete_data(self, index: np.uint8):
        del self.x_data[index]
        del self.y_data[index]

    def plot_to_axis(self, axis: Axis, log_scale_y: bool = False):
        keywords = {
            "color": self.color,
            "marker": self.marker,
            "linestyle": self.linestyle,
        }

        axis.title.set_text(self.title)
        axis.plot(self.x_data, self.y_data, **keywords)

        axis.set_xlabel(self.x_label)
        axis.set_ylabel(self.y_label)

        if log_scale_y:
            axis.set_yscale("log")

        if self.x_limits is not None:
            axis.set_xlim(self.x_limits[0], self.x_limits[1])
        if self.y_limits is not None:
            axis.set_ylim(self.y_limits[0], self.y_limits[1])

        plt.sca(axis)

        if self.x_ticks:
            plt.xticks(self.x_ticks)
        if self.y_ticks:
            plt.yticks(self.y_ticks)

    def add_trendline(self, exponential: bool = True):
        x = np.array(self.x_data)
        y = np.array(self.y_data)

        if exponential:
            y = np.log(y)

        index = np.isfinite(x) & np.isfinite(y)

        z = np.polyfit(x[index], y[index], 1)
        p = np.poly1d(z)
        print(p)

        return np.poly1d(z)

        pass


class Figure2D(object):
    __data_sets: dict[str:DataSet2D] = {}
    __figure: plt.figure
    __axes: Axis.axes

    def __init__(
        self, window_title: str = "Figure", rows: np.uint8 = 1, columns: int = 1
    ):
        if rows < 1 or columns < 1:
            raise ValueError(
                "Invalid number of rows or columns! Please Set at least 1 row and 1 column."
            )

        self.__figure, self.__axes = plt.subplots(rows, columns, squeeze=False)
        self.__figure.canvas.manager.set_window_title(window_title)

    def update_data_set(self, dataset: DataSet2D):
        if dataset.title not in self.__data_sets.keys():
            print(f"[INFO] Set does not exist! Creating a data set '{dataset.title}'!")
        self.__data_sets.update({dataset.title: dataset})

    def plot_datasets_to_axis(
        self,
        row,
        column,
        datasets: list[DataSet2D],
        new_title: str = None,
        log_scale_y: bool = False,
    ):
        for dataset in datasets:
            dataset.plot_to_axis(self.__axes[row, column], log_scale_y=log_scale_y)
        self.__axes[row, column].title.set_text(new_title)

    def plot(self, window_width: np.uint16 = 600, window_height: np.uint16 = 400):
        self.__figure.set_figwidth(window_width / 100)
        self.__figure.set_figheight(window_height / 100)
        self.__figure.tight_layout()
        plt.show()

    def clear(self):
        self.__data_sets = {}
