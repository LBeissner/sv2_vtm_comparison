# analytics
import cProfile
import io
import pstats
from datetime import date
from os import listdir
from pathlib import Path

# 3rd party libraries
import cv2 as cv
import numpy as np
import open3d as o3d
from numba import njit
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA

# project file import
# ! MODIFICATION import of the new camera classes
from sick_camera_wrapper.camera import SafeVisionary2, VisionaryTMini
from sick_camera_wrapper.cam_data import CamData


# * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ o ////////////////////////////////////////////////// * #
# * OBJECT POINT CLOUD


class ObjectPointCloud:
    __pca = PCA()
    __hog = cv.HOGDescriptor()
    __hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

    __stride: np.uint8 = 8
    __padding: np.uint8 = 8
    __scale: np.float64 = 1.01

    def __init__(self, id: np.uint16):
        self._id: np.uint16 = id
        self._points_xyz: np.ndarray
        self._conversion_factors: np.ndarray
        self._frame_ticker: np.uint16 = 0

        self._muted: np.bool_ = False
        self._marked_for_deletion = False
        self._is_human = False

        self._current_centroid_xyz: np.ndarray
        self._previous_centroid_xyz: np.ndarray

        self._previous_pca_origin_xz: np.ndarray
        self._current_pca_origin_xz: np.ndarray

        self._path_image: np.ndarray

        self._displacement_radial_mm: np.float32 = 0.0
        self._velocity_mms: np.float32 = 0.0
        self._distance_traveled_mm: np.float32 = 0.0
        self._distance_x_mm: np.float32 = 0.0
        self._distance_y_mm: np.float32 = 0.0
        self._distance_to_lens_mm: np.uint16 = 0

    def __str__(self):
        minima = np.round(np.amin(self._points_xyz, axis=0), 0)
        maxima = np.round(np.amax(self._points_xyz, axis=0), 0)
        string = (
            f"[INFO]\tObject ID:\t{self._id}"
            + f"\n\tNumber of Points:\t{self._points_xyz.shape[0]}"
            + f"\n\tMin XYZ:\t{minima[0]}\t{minima[1]}\t{minima[2]}"
            + f"\n\tMax XYZ:\t{maxima[0]}\t{maxima[1]}\t{maxima[2]}"
            + f"\n\tHuman:\t{self._is_human}"
            + f"\n\tMuted:\t{self._muted}"
        )
        return string

    def is_muted(self):
        return self._muted

    def is_marked_for_deletion(self):
        return self._marked_for_deletion

    def _get_points_from_contour(self, contour):

        # calculate offset on x axis
        offset_x = projection_dimensions_xy[0] // 2

        # get boundaries from contour and subtract the offset
        boundaries_xz = contour[2:4]
        boundaries_xz[:, 0] -= offset_x

        # scale boundaries and add the y component
        boundaries_xz = (boundaries_xz / scene_scaling_factors_xz).T
        boundaries_xz[1] += scene_boundaries_xyz[2, 0]
        boundaries_xyz = np.insert(
            boundaries_xz, obj=1, values=scene_boundaries_xyz[1], axis=0
        ).T

        # get scene points in original scale and their conversion factors
        points_xyz = np.copy(scene_points_xyz)
        conversion_factors = np.copy(scene_conversion_factors)

        # cut object from scene
        to_keep = np.all(
            (points_xyz >= boundaries_xyz[0, np.newaxis, :])
            & (points_xyz < boundaries_xyz[1, np.newaxis, :]),
            axis=1,
        )
        self._points_xyz = points_xyz[to_keep]
        self._conversion_factors = conversion_factors[to_keep]

    def update_from_contours(self, contours: np.ndarray):
        contour_index, *_ = np.where(contours[:, 0, 0] == self._id)
        if contour_index.size:
            self._get_points_from_contour(np.squeeze(contours[contour_index]))
        else:
            self._marked_for_deletion = True

    @classmethod
    def create_from_contour(cls, contour):
        object = cls(contour[0, 0])
        object._get_points_from_contour(contour)
        return object

    def analyze(self):
        # retrieve object points and the ground level
        points_xyz = np.copy(self._points_xyz)
        ground_level_y = scene_boundaries_xyz[1, 0]

        # clean the point cloud from outliers to get the estimated head height of the object
        head_height_y = np.percentile(points_xyz[:, 1], q=99.5, axis=0)

        # estimate shoulder height and remove points beneath it
        shoulder_height_y = ground_level_y + armpit_height_estimate * (
            head_height_y - ground_level_y
        )
        to_keep = points_xyz[:, 1] > shoulder_height_y
        points_xyz = points_xyz[to_keep]

        # get the distance between object centroid and camera lens; always round up
        self._current_centroid_xyz = np.median(points_xyz, axis=0)
        self._distance_to_lens_mm = np.ceil(np.linalg.norm(self._current_centroid_xyz))

        # every X frames, measure the moved distance and velocity
        if not current_frame % movement_frame_delta:

            # if the object existed in the frame prior ...
            if hasattr(self, "_previous_centroid_xyz"):

                # determin displacement in radial direction
                self._displacement_radial_mm = np.linalg.norm(
                    self._current_centroid_xyz
                ) - np.linalg.norm(self._previous_centroid_xyz)

                # project the points in xz plane
                current_centroid_xz = np.delete(
                    self._current_centroid_xyz, obj=1, axis=0
                )
                previous_centroid_xz = np.delete(
                    self._previous_centroid_xyz, obj=1, axis=0
                )

                # determin movement direction and moved distance
                movement_direction_xz = current_centroid_xz - previous_centroid_xz
                euclidean_distance_xz = np.linalg.norm(movement_direction_xz)

                # get the velocity in mm/s with an estimated frame rate of 25 fps
                movement_velocity_mms = (
                    euclidean_distance_xz * 25 / movement_frame_delta
                )
                self._velocity_mms = np.round(
                    movement_velocity_mms / np.power(10, 0), 2
                )
                self._distance_traveled_mm = euclidean_distance_xz

                # update status of the object according to the threshold
                if euclidean_distance_xz < euclidean_distance_threshold:
                    self._muted = True
                else:
                    self._muted = False

            self._previous_centroid_xyz = np.copy(self._current_centroid_xyz)
            self._frame_ticker = 0

        # pjoject the points in the xz plane and calculate the centroid
        points_xz = np.delete(points_xyz, obj=1, axis=1)

        # execute pca
        ObjectPointCloud.__pca.fit(points_xz)

        # get the principal vectors
        # * __pca.mean_: vector origin
        # * __pca.components_: direction vectors
        # * __pca.explained_variance_: lengths in x and z squared
        self._current_pca_origin_xz = ObjectPointCloud.__pca.mean_
        self._pca_vectors_xz = np.concatenate(
            [
                ObjectPointCloud.__pca.components_,
                -ObjectPointCloud.__pca.components_,
            ],
            axis=0,
        )

        self._previous_pca_origin_xz = np.copy(self._current_pca_origin_xz)

    def detect_human(self, intensity_map):
        if not self._muted:
            points_xyz = self._points_xyz
            points_xyz = scene_rotation_matrix.inv().apply(points_xyz)
            pixel_coordinates_xy = jit_world_to_pixel_coordinates(
                points_xyz, self._conversion_factors
            )
            detector_window_xy = np.array([64, 128]) + 2 * ObjectPointCloud.__padding

            image_borders_xy = np.array(
                [
                    np.amin(pixel_coordinates_xy, axis=0),
                    np.amax(pixel_coordinates_xy, axis=0),
                ],
                dtype=np.uint16,
            )

            mask = np.zeros(np.flip(projection_dimensions_xy))
            mask[pixel_coordinates_xy[:, 1], pixel_coordinates_xy[:, 0]] = 255

            padding_xy = np.array([16, 16])

            image_dimensions_xy = image_borders_xy[1] - image_borders_xy[0]
            if image_dimensions_xy[0] < detector_window_xy[0]:
                padding_xy[0] += (detector_window_xy[0] - image_dimensions_xy[0]) // 2

            if image_dimensions_xy[1] < detector_window_xy[1]:
                padding_xy[1] += (detector_window_xy[1] - image_dimensions_xy[1]) // 2

            image_borders_xy[1] = image_borders_xy[1] + 2 * padding_xy

            temp_image = cv.copyMakeBorder(
                intensity_map,
                top=padding_xy[1],
                bottom=padding_xy[1],
                left=padding_xy[0],
                right=padding_xy[0],
                borderType=cv.BORDER_REFLECT_101,
            )

            temp_image = temp_image[
                image_borders_xy[0, 1] : image_borders_xy[1, 1],
                image_borders_xy[0, 0] : image_borders_xy[1, 0],
            ]

            temp_image = np.power(temp_image, gamma_exponent)
            temp_image = (temp_image / temp_image.max() * 255).astype(np.uint8)

            stride = ObjectPointCloud.__stride
            padding = ObjectPointCloud.__padding
            scale = ObjectPointCloud.__scale

            regions, _ = ObjectPointCloud.__hog.detectMultiScale(
                temp_image,
                winStride=(stride, stride),
                padding=(padding, padding),
                scale=scale,
            )

            if len(regions):
                self._is_human = True
                self._is_muted = True

            return temp_image
        return None

    def show(self):

        print(self)

        offset_xz = np.delete(np.copy(scene_boundaries_xyz), obj=1, axis=0)
        offset_xz = offset_xz.T[0]

        previous_position_xz = (
            (np.copy(self._previous_pca_origin_xz) - offset_xz)
            * scene_scaling_factors_xz
        ).astype(np.uint16)
        current_position_xz = (
            (np.copy(self._current_pca_origin_xz) - offset_xz)
            * scene_scaling_factors_xz
        ).astype(np.uint16)

        length_px = 20

        directions_xz = np.copy(self._pca_vectors_xz)
        directions_xz[::2] *= length_px * 2
        directions_xz[1::2] *= length_px

        current_orientation_xz = (directions_xz + current_position_xz).astype(np.uint16)

        if hasattr(self, "_path_image"):
            self._path_image = cv.line(
                self._path_image,
                previous_position_xz,
                current_position_xz,
                color=(84, 82, 0),
                thickness=2,
                lineType=cv.LINE_AA,
            )
        else:
            self._path_image = calibration_top_down_view.astype(np.uint8) * 128
        image = cv.cvtColor(np.copy(self._path_image), cv.COLOR_GRAY2BGR)

        image = cv.line(
            image,
            pt1=current_position_xz,
            pt2=current_orientation_xz[0],
            color=(255, 255, 255),
            thickness=1,
            lineType=cv.LINE_AA,
        )

        image = cv.line(
            image,
            pt1=current_position_xz,
            pt2=current_orientation_xz[1],
            color=(255, 255, 255),
            thickness=1,
            lineType=cv.LINE_AA,
        )

        image = cv.line(
            image,
            pt1=current_position_xz,
            pt2=current_orientation_xz[2],
            color=(255, 255, 255),
            thickness=1,
            lineType=cv.LINE_AA,
        )

        image = cv.line(
            image,
            pt1=current_position_xz,
            pt2=current_orientation_xz[3],
            color=(255, 255, 255),
            thickness=1,
            lineType=cv.LINE_AA,
        )

        image = cv.circle(
            image,
            center=current_position_xz,
            radius=2,
            color=(188, 185, 0),
            thickness=2,
            lineType=cv.LINE_AA,
        )

        cv.imshow(f"{self._id}", image)
        cv.setWindowTitle(
            f"{self._id}",
            f"Object {self._id}:     Distance: {self._distance_to_lens_mm} mm\tVelocity: {self._velocity_mms} mm/s",
        )

        return image


# * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ o ////////////////////////////////////////////////// * #
# * GLOBAL LISTS AND VARIABLES

calibration_euler_angles_xyz = []
calibration_ground_levels_y = []
calibration_top_down_views = []
calibration_intensity_maps = []
scene_objects: list[ObjectPointCloud] = []
object_result_images: list[np.ndarray] = []
object_human_images: list[np.ndarray] = []
scene_object_detected = False
current_frame = 0
calibration_ticker = 0

# * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ o ////////////////////////////////////////////////// * #
# * DETECTOR SETUP


def setup(
    calibration_frames: np.uint16 = 200,
    on_the_run_frames: np.uint16 = 90,
    image_dimensions_xy: tuple[np.uint16, np.uint16] = (512, 424),
    perception_depth: np.uint16 = 9000,
    height_thresholds_mm: tuple[np.uint16, np.uint16] = (0, 2000),
    distance_threshold_px: np.uint8 = 10,
    distance_threshold_mm: np.uint16 = 8,
    relative_armpit_height: np.float64 = 0.7,
    frame_delta: np.uint8 = 5,
    minimum_contour_area: np.uint16 = 100,
    filter_sensitivity: np.uint8 = 5,
    gamma: np.float16 = 0.5,
):
    global total_calibration_frames
    total_calibration_frames = calibration_frames

    global projection_dimensions_xy
    projection_dimensions_xy = image_dimensions_xy

    global calibration_height_thresholds
    calibration_height_thresholds = height_thresholds_mm

    global chebyshev_distance_threshold
    chebyshev_distance_threshold = distance_threshold_px

    global euclidean_distance_threshold
    euclidean_distance_threshold = distance_threshold_mm

    global armpit_height_estimate
    armpit_height_estimate = relative_armpit_height

    global contour_area_minimum
    contour_area_minimum = minimum_contour_area

    global intensity_filter_sensitivity
    intensity_filter_sensitivity = filter_sensitivity

    global calibration_on_the_run_frames
    calibration_on_the_run_frames = on_the_run_frames

    global scene_perception_depth
    scene_perception_depth = perception_depth

    global movement_frame_delta
    movement_frame_delta = frame_delta

    global gamma_exponent
    gamma_exponent = gamma


# * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ o ////////////////////////////////////////////////// * #
# * COORDINATE SYSTEM TRANSFORM FUNCTIONS


@njit
def jit_pixel_to_world_coordinates(
    distance_image: np.ndarray,
):

    # code based on the sick_wrapper.cam.dist_to_npy method (c) Karsten Flores
    # * changed to work with numba njit

    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    k1 = miscellaneous[0]
    k2 = miscellaneous[1]
    f2rc = miscellaneous[2]

    m_c2w = extrinsic_matrix.reshape(4, 4)
    m_d = distance_image

    # [x, y] to [X_d, Y_d], rotation of 180 degrees (ICS to CCS)
    m_xp = (cx - coordinates_grid_xy[0, :, :]) / fx
    m_yp = (cy - coordinates_grid_xy[1, :, :]) / fy

    m_r2 = m_xp * m_xp + m_yp * m_yp
    m_k = 1 + (k1 + k2 * m_r2) * m_r2

    m_xd = m_xp * m_k
    m_yd = m_yp * m_k

    # [X_d, Y_d] to [X_c, Y_c, Z_c]
    m_s0 = np.sqrt(m_xd * m_xd + m_yd * m_yd + 1)

    m_xc = m_xd * m_d / m_s0
    m_yc = m_yd * m_d / m_s0
    m_zc = m_d / m_s0 - f2rc

    m_c = np.stack(
        (
            m_xc.ravel(),
            m_yc.ravel(),
            m_zc.ravel(),
            np.ones(distance_image.shape[1] * distance_image.shape[0]),
        ),
        axis=-1,
    )

    # [X_c, Y_c, Z_c] to [X, Y, Z]
    m_w = (m_c2w @ m_c.T)[0:3, :].T
    m_w: np.ndarray = np.zeros(m_w.shape) + m_w

    # addition to retrieve the conversion factors
    d_vector = np.ravel(m_d)
    s_vector = np.ravel(m_s0)
    k_vektor = np.ravel(m_k)

    # the vector positions of zero values in the flattened depth map
    # need to be removed to avoid zero division
    non_zero_indices = d_vector > 0

    d_vector = d_vector[non_zero_indices]
    s_vector = s_vector[non_zero_indices]
    k_vektor = k_vektor[non_zero_indices]

    m_w = m_w[non_zero_indices]

    conversion_factors = s_vector / (k_vektor * d_vector)

    return (m_w, conversion_factors)


@njit
def jit_world_to_pixel_coordinates(
    world_coords_xyz: np.ndarray,
    conversion_factors: np.ndarray,
) -> np.ndarray:

    # get extrinsic matrix
    m_w2c = np.linalg.inv(extrinsic_matrix.reshape(4, 4))

    # get intrinsic camera parameters
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]

    # [X, Y, Z] to [X, Y, Z, 1]
    homogeneous_world_coords_xyz = np.ones((world_coords_xyz.shape[0], 4))
    homogeneous_world_coords_xyz[:, :3] = world_coords_xyz

    # [X, Y, Z, 1] to [X_c, Y_c, Z_c]
    distorted_coords_xyz = (m_w2c @ homogeneous_world_coords_xyz.T).T

    # get coordinates with z != 0 only
    nonzero_indices = np.nonzero(distorted_coords_xyz[:, 2])
    conversion_factors = conversion_factors[nonzero_indices]
    distorted_coords_xyz = distorted_coords_xyz[nonzero_indices].T

    # [X_c, Y_c, Z_c] to [x, y]
    x_distorted = distorted_coords_xyz[0] * conversion_factors
    y_distorted = distorted_coords_xyz[1] * conversion_factors

    x = cx - x_distorted * fx
    y = cy - y_distorted * fy

    pixel_xy = np.empty((x.shape[0], 2), np.float64)
    pixel_xy[:, 0] = x
    pixel_xy[:, 1] = y

    return pixel_xy.astype(np.uint16)


# * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ o ////////////////////////////////////////////////// * #
# * CALIBRATION LOOP


def calculate_scene(
    angles_of_view_xy: tuple[np.uint8, np.uint8] = (70, 60),
) -> np.ndarray:

    aov = np.array(angles_of_view_xy, dtype=np.uint8)
    dimensions_xy = 2 * scene_perception_depth * np.tan(np.radians(aov / 2))
    dimensions_xy = dimensions_xy // 10 * 10

    boundaries_xyz = dimensions_xy / 2
    boundaries_xyz = np.stack([-boundaries_xyz, boundaries_xyz], axis=1)
    boundaries_xyz = np.insert(
        boundaries_xyz, obj=2, values=[0, scene_perception_depth], axis=0
    ).astype(np.int32)

    global scene_corners_xyz
    scene_corners_xyz = np.array(
        np.meshgrid(
            boundaries_xyz[0],
            boundaries_xyz[1],
            boundaries_xyz[2],
            indexing="xy",
        )
    ).T
    scene_corners_xyz = scene_corners_xyz.reshape((-1, 3))


def angle_between_vectors(vector_1: np.ndarray, vector_2: np.ndarray) -> np.float64:
    magnitude_1 = np.linalg.norm(vector_1)
    magnitude_2 = np.linalg.norm(vector_2)
    if not magnitude_1 or not magnitude_2:
        return 0
    vector_1 = vector_1 / magnitude_1
    vector_2 = vector_2 / magnitude_2
    angle = np.arccos(np.dot(vector_1, vector_2))
    return angle


def segment_plane(
    point_cloud: o3d.geometry.PointCloud,
    distance_threshold_mm: np.uint16 = 50,
    ransac_n: np.uint8 = 5,
    iterations: np.uint16 = 1000,
):

    plane_parameters, _ = point_cloud.segment_plane(
        distance_threshold_mm, ransac_n, iterations
    )
    return plane_parameters, distance_threshold_mm


def collect_camera_position_data(
    points_xyz: np.ndarray,
):

    # discard the upper half of the point cloud
    cut = points_xyz.shape[0] // 2
    lower_coords_xyz = points_xyz[cut:, :]

    # clear points too close to origin of coordinate system
    lower_coords_xyz = lower_coords_xyz[lower_coords_xyz[:, 2] > 500]

    # create an object PointCloud (open3D)
    lower_point_cloud = o3d.geometry.PointCloud()
    lower_point_cloud.points = o3d.utility.Vector3dVector(lower_coords_xyz)

    # use RANSAC to find the largest plane in the point clouds bottom half
    plane_parameters, distance_threshold_mm = segment_plane(lower_point_cloud)

    # setup of normal vector and y axis
    n_vector = np.asarray(plane_parameters[:3]).astype(np.float64)
    y_axis = np.asarray([0, 1, 0], dtype=np.float64)

    # distance between plane and lens centre, future intersection point of the floor with the y axis
    ground_level_y = plane_parameters[3] / np.linalg.norm(n_vector)

    # correct the normal vector direction if it points towards the rear quadrants
    if np.sign(n_vector[2]) == -1:
        n_vector *= -1
        ground_level_y *= -1

    ground_level_y += distance_threshold_mm

    # get rotation direction from normal vector
    rotation_direction = np.sign(np.flip(n_vector))
    rotation_direction[2] *= -1

    # calculate the angles between the normal vector and y axis
    # * alpha: rotation around x axis (projection on yz plane)
    # * gamma: rotation around z axis (projection on xy plane)
    euler_angles_xyz = np.array(
        [
            np.pi
            - angle_between_vectors(
                np.delete(n_vector, obj=0, axis=0), np.delete(y_axis, obj=0, axis=0)
            ),
            0,
            np.pi
            - angle_between_vectors(
                np.delete(n_vector, obj=2, axis=0), np.delete(y_axis, obj=2, axis=0)
            ),
        ]
    )

    # apply rotation direction
    euler_angles_xyz *= rotation_direction

    # save euler angles and ground level for final calibration
    global calibration_euler_angles_xyz
    calibration_euler_angles_xyz.append(euler_angles_xyz)

    global calibration_ground_levels_y
    calibration_ground_levels_y.append(ground_level_y)

    # setup rotation and scene boundaries for top down view calibration
    global scene_rotation_matrix
    scene_rotation_matrix = R.from_euler("XYZ", euler_angles_xyz.tolist())

    global scene_boundaries_xyz
    corners_xyz = scene_rotation_matrix.apply(scene_corners_xyz)
    scene_boundaries_xyz = np.stack(
        [np.amin(corners_xyz, axis=0), np.amax(corners_xyz, axis=0)],
        axis=0,
    ).T
    scene_boundaries_xyz[1] = [
        ground_level_y + calibration_height_thresholds[0],
        ground_level_y + calibration_height_thresholds[1],
    ]

    if euler_angles_xyz[0] < np.radians(60):
        scene_boundaries_xyz[2] = [0, scene_perception_depth]


def calibrate_camera_position():
    global scene_rotation_matrix
    final_euler_angles_xyz = np.average(np.array(calibration_euler_angles_xyz), axis=0)
    scene_rotation_matrix = R.from_euler("XYZ", final_euler_angles_xyz.tolist())

    ground_level_y = np.average(np.array(calibration_ground_levels_y), axis=0)

    global scene_boundaries_xyz
    corners_xyz = scene_rotation_matrix.apply(scene_corners_xyz)
    scene_boundaries_xyz = np.stack(
        [np.amin(corners_xyz, axis=0), np.amax(corners_xyz, axis=0)],
        axis=0,
    ).T
    scene_boundaries_xyz[1] = [
        ground_level_y + calibration_height_thresholds[0],
        ground_level_y + calibration_height_thresholds[1],
    ]

    if final_euler_angles_xyz[0] < np.radians(60):
        scene_boundaries_xyz[2] = [0, scene_perception_depth]


def collect_intensity_data(intensity_map):
    global calibration_intensity_maps
    calibration_intensity_maps.append(intensity_map)

    global calibration_ticker
    calibration_ticker += 1

    if scene_object_detected:
        calibration_intensity_maps.clear()
        calibration_ticker = 0


def calibrate_intensity():
    global calibration_ticker
    global scene_intensity_mean
    global scene_intensity_deviation
    if calibration_ticker >= calibration_on_the_run_frames - 1:
        # global scene_intensity_mean
        scene_intensity_mean = np.mean(np.array(calibration_intensity_maps), axis=0)
        # global scene_intensity_deviation
        scene_intensity_deviation = np.std(np.array(calibration_intensity_maps), axis=0)
        calibration_intensity_maps.clear()
        calibration_ticker = 0


def collect_top_down_views(depth_map: np.ndarray):
    # rotate and threshold the current point cloud
    get_scene_points(depth_map)

    # project the point cloud into xz plane
    get_scene_top_down_view()

    global calibration_top_down_views
    calibration_top_down_views.append(np.copy(scene_top_down_view))

    global calibration_top_down_view_no_filter
    calibration_top_down_view_no_filter = np.copy(scene_top_down_view) * 255


def calibrate_top_down_view():

    threshold = total_calibration_frames / 10

    top_down_view = np.sum(np.array(calibration_top_down_views), axis=0).astype(
        np.uint8
    )

    top_down_view[top_down_view < threshold] = 0

    global calibration_top_down_view
    calibration_top_down_view = top_down_view.astype(np.bool_)

    global calibration_top_down_view_image
    calibration_top_down_view_image = calibration_top_down_view.astype(np.uint8) * 255

    cv.imshow("Scene Top Down View", calibration_top_down_view_image)


# ! MODIFICATION deleted get_data typing
def calibration_loop(
    data_generator,
):
    calculate_scene()

    global current_frame

    while current_frame < total_calibration_frames:
        camera_data, _ = next(data_generator)

        intensity_map = camera_data.ints_img
        depth_map = camera_data.dist_img

        # with the first calibration frame get all parameters neccessary to transform coordinates
        if not current_frame:

            # x and y pixel coordinates for cam to world coordiante transformation
            global coordinates_grid_xy
            coordinates_grid_xy = np.array(
                np.meshgrid(
                    range(camera_data.cam_params.width),
                    range(camera_data.cam_params.height),
                    indexing="xy",
                )
            )

            # intrinsic camera parameters
            global intrinsic_matrix
            intrinsic_matrix = np.array(
                [
                    [camera_data.cam_params.fx, 0, camera_data.cam_params.cx],
                    [0, camera_data.cam_params.fy, camera_data.cam_params.cy],
                    [0, 0, 1],
                ]
            )

            # extrinsic camera parameters
            global extrinsic_matrix
            extrinsic_matrix = np.array(camera_data.cam_params.cam2worldMatrix)

            # miscellaneous contains the distortion coefficients and f2rc
            global miscellaneous
            miscellaneous = np.array(
                [
                    camera_data.cam_params.k1,
                    camera_data.cam_params.k2,
                    camera_data.cam_params.f2rc,
                ]
            )

            # first time coordinate transformation for jit compiling
            points_xyz, conversion_factors = jit_pixel_to_world_coordinates(depth_map)
            _ = jit_world_to_pixel_coordinates(points_xyz, conversion_factors)
        else:
            points_xyz, _ = jit_pixel_to_world_coordinates(depth_map)

        collect_camera_position_data(points_xyz)
        collect_intensity_data(intensity_map)
        collect_top_down_views(depth_map)

        current_frame += 1

    calibrate_camera_position()
    calibrate_intensity()
    calibrate_top_down_view()


# * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ o ////////////////////////////////////////////////// * #
# * DETECTION LOOP


def get_scene_points(depth_map: np.ndarray):
    points_xyz, conversion_factors = jit_pixel_to_world_coordinates(depth_map)

    points_xyz = scene_rotation_matrix.apply(points_xyz)
    boundaries_xyz = np.copy(scene_boundaries_xyz).T

    to_keep = np.all(
        (points_xyz >= boundaries_xyz[0, np.newaxis, :])
        & (points_xyz < boundaries_xyz[1, np.newaxis, :]),
        axis=1,
    )

    global scene_points_xyz
    scene_points_xyz = points_xyz[to_keep]
    global scene_conversion_factors
    scene_conversion_factors = conversion_factors[to_keep]


def get_scene_top_down_view():
    if scene_points_xyz.size:
        boundaries_xz = np.delete(np.copy(scene_boundaries_xyz), obj=1, axis=0)

        global scene_scaling_factors_xz
        scene_scaling_factors_xz = np.array(projection_dimensions_xy) / np.sum(
            np.abs(boundaries_xz), axis=1
        )
        offset_xz = boundaries_xz.T[0]

        scaled_points_xz = np.delete(np.copy(scene_points_xyz).T, obj=1, axis=0).T

        scaled_points_xz = (
            (scaled_points_xz - offset_xz) * scene_scaling_factors_xz
        ).astype(np.uint16)

        projection_xz = np.zeros(np.flip(projection_dimensions_xy), dtype=np.uint8)
        projection_xz[scaled_points_xz[:, 1], scaled_points_xz[:, 0]] = 1

        projection_xz = cv.medianBlur(projection_xz, ksize=3)

    else:
        projection_xz = np.zeros(np.flip(projection_dimensions_xy), dtype=np.uint8)

    global scene_top_down_view
    scene_top_down_view = projection_xz

    global scene_top_down_view_image
    scene_top_down_view_image = projection_xz * 255

    # cv.imshow("", scene_top_down_view_image)


def get_intensity_filter(intensity_map):
    global scene_intensity_mean
    intensity_filter_mask = np.logical_or(
        np.array(
            intensity_map
            > scene_intensity_mean
            + intensity_filter_sensitivity * scene_intensity_deviation,
            dtype=np.uint8,
        ),
        np.array(
            intensity_map
            < scene_intensity_mean
            - intensity_filter_sensitivity * scene_intensity_deviation,
            dtype=np.uint8,
        ),
    ).astype(np.uint8)
    return intensity_filter_mask


def get_contour_features(
    projection: np.ndarray,
) -> np.ndarray:

    # use the SUZUKI-ABE algorithm to get contours from the edge image
    contours, _ = cv.findContours(
        np.copy(projection), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    # setup storage for contour features
    contour_features = []

    index = 0

    # per contour calculate two corners of its bounding box and the contour centroid
    for contour in contours:
        moments = cv.moments(contour)

        # minimal contour area filter
        if moments["m00"] < contour_area_minimum:
            continue
        else:
            contour_centroid_xy = (
                np.int64(moments["m10"] / moments["m00"]),
                np.int64(moments["m01"] / moments["m00"]),
            )

            x, y, w, h = cv.boundingRect(contour)
            features = np.stack(
                [[index, 0], contour_centroid_xy, [x, y], [x + w, y + h]]
            )
            contour_features.append(features)
            index += 1

    return np.array(contour_features)


def label_contours():
    if "scene_previous_contours" in globals():
        global scene_previous_contours
        global scene_current_contours

        previous_contours = np.copy(scene_previous_contours)
        current_contours = np.copy(scene_current_contours)

        # store the largest quantity of points
        number_of_indices = np.max(
            [previous_contours.shape[0], current_contours.shape[0]]
        )

        if previous_contours.size and current_contours.size:

            current_centroids_xz = current_contours[:, 1]
            previous_centroids_xz = previous_contours[:, 1]

            chebyshev_distances = np.max(
                np.absolute(
                    previous_centroids_xz[:, np.newaxis] - current_centroids_xz
                ).astype(np.int64),
                axis=2,
            )

            # matching indices of previous contours (pos 0) and current contours (pos 1)
            matching_indices = np.nonzero(
                chebyshev_distances < chebyshev_distance_threshold
            )

            # save existing contours with index of the last frame
            previous_contours[matching_indices[0], 1:] = current_contours[
                matching_indices[1], 1:
            ]

            # delete previous contours that no longer exist
            non_existent_ids = np.delete(
                np.arange(previous_contours.shape[0]),
                obj=matching_indices[0],
                axis=0,
            )

            previous_contours = np.delete(
                previous_contours, obj=non_existent_ids, axis=0
            )

            # update old scene objects ...
            deletion_indices = []
            for index, object in enumerate(scene_objects):
                object.update_from_contours(np.copy(previous_contours))
                if object.is_marked_for_deletion():
                    deletion_indices.append(index)

            # ... and delete objects that no longer exist
            for index in reversed(deletion_indices):
                del scene_objects[index]

            # delete contours already saved
            current_contours = np.delete(
                current_contours, obj=matching_indices[1], axis=0
            )

            # get new ids for new contours
            if current_contours.size:
                new_ids = np.delete(
                    np.arange(number_of_indices),
                    obj=previous_contours[:, 0, 0],
                    axis=0,
                )
                new_ids = new_ids[: current_contours.shape[0]]

                current_contours[:, 0, 0] = new_ids

            # concatenate all contours and save them globally
            current_contours = np.concatenate(
                [previous_contours, current_contours], axis=0
            )
            scene_current_contours = current_contours

            # create new objects for new contours
            start_index = len(scene_objects)
            for contour in current_contours[start_index:]:
                object = ObjectPointCloud.create_from_contour(np.copy(contour))
                scene_objects.append(object)

        else:
            scene_objects.clear()


def detection_loop(cam_data: CamData):
    depth_map = np.copy(cam_data.dist_img)
    intensity_map = np.copy(cam_data.ints_img)

    global current_frame
    current_frame += 1
    print(current_frame)

    # filter mask based on the calibration values for mean intensity in the scene and its standard deviation
    intensity_filter_mask = get_intensity_filter(intensity_map)

    intensity_filter_mask = cv.morphologyEx(
        intensity_filter_mask, cv.MORPH_DILATE, kernel=np.ones((15, 1)), iterations=1
    )
    intensity_filter_mask = cv.morphologyEx(
        intensity_filter_mask, cv.MORPH_ERODE, kernel=np.ones((1, 3)), iterations=1
    )
    intensity_filter_mask = cv.morphologyEx(
        intensity_filter_mask, cv.MORPH_DILATE, kernel=np.ones((9, 9)), iterations=1
    )

    global intensity_filter_mask_image
    intensity_filter_mask_image = intensity_filter_mask * 255

    get_scene_points(depth_map * intensity_filter_mask)

    global scene_points_xyz
    to_delete = np.arange(0, scene_points_xyz.shape[0], 2)
    scene_points_xyz = np.delete(np.copy(scene_points_xyz), obj=to_delete, axis=0)

    global scene_conversion_factors
    scene_conversion_factors = np.delete(
        scene_conversion_factors,
        obj=to_delete,
        axis=0,
    )

    get_scene_top_down_view()

    global scene_top_down_view
    scene_top_down_view = np.logical_and(
        scene_top_down_view, np.logical_not(calibration_top_down_view)
    ).astype(np.uint8)

    scene_top_down_view = cv.morphologyEx(
        scene_top_down_view, cv.MORPH_CLOSE, kernel=np.ones((5, 5)), iterations=1
    )

    global scene_current_contours
    scene_current_contours = get_contour_features(scene_top_down_view)

    label_contours()

    global scene_previous_contours
    scene_previous_contours = np.copy(scene_current_contours)

    global object_human_images

    # execute the class methods
    # * analyze() to get position, orientation and velocity; neccessary to determin movement state of the object
    # * detect_human() to verify object as human
    for object in scene_objects:
        object.analyze()
        human_cutout = object.detect_human(intensity_map)
        if human_cutout is not None:
            object_human_images.append([object._id, human_cutout])

    # check, if all currently detected objects are stationary/muted and set detection flag accordingly
    global scene_object_detected
    if all(object.is_muted() for object in scene_objects):
        scene_object_detected = False
    else:
        scene_object_detected = True

    collect_intensity_data(intensity_map)
    calibrate_intensity()

    global intensity_map_image
    intensity_map = np.power(intensity_map, gamma_exponent)
    intensity_map_image = (intensity_map / intensity_map.max() * 255).astype(np.uint8)

    cv.imshow("Intensity", intensity_map_image)


def show_results(target_id=None):

    print("\nOBJECTS\n")

    scene_image = (
        np.logical_and(
            calibration_top_down_view.astype(np.uint8),
            np.logical_not(scene_top_down_view),
        ).astype(np.uint8)
        * 128
    )

    scene_image += np.copy(scene_top_down_view) * 255
    scene_image = cv.cvtColor(
        scene_image,
        cv.COLOR_GRAY2BGR,
    )

    for contour in scene_current_contours:
        scene_image = cv.circle(
            scene_image, contour[1], radius=2, color=(0, 128, 255), thickness=2
        )
        scene_image = cv.rectangle(
            scene_image,
            contour[2],
            contour[3],
            color=(255, 0, 0),
            thickness=2,
        )
        scene_image = cv.putText(
            scene_image,
            text=str(contour[0, 0]),
            org=contour[2],
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(0, 255, 255),
            thickness=1,
            lineType=cv.LINE_4,
            bottomLeftOrigin=False,
        )

    global scene_boxed_top_down_view_image
    scene_boxed_top_down_view_image = scene_image

    cv.imshow("Scene Top Down View", scene_image)
    cv.setWindowTitle(
        "Scene Top Down View",
        f"Frame {current_frame} - Top Down View:\t{len(scene_objects)} objects detected!",
    )

    global object_result_images

    for object in scene_objects:
        if target_id is None:
            result_image = object.show()
            object_result_images.append([object._id, result_image])
        else:
            if object._id == target_id:
                result_image = object.show()
                object_result_images.append([object._id, result_image])


def save_data(ssr_file: str, frame_list: list[int] = None, fps: np.uint8 = 25):

    file_path = Path(__file__)

    accuracy_directory = Path(f"{file_path.parent}/accuracy results/{ssr_file}")
    accuracy_directory.mkdir(parents=True, exist_ok=True)

    results_file_path = Path(accuracy_directory / f"object_data_{ssr_file}.txt")

    image_directory = Path(f"{file_path.parent}/images/{ssr_file}")
    image_directory.mkdir(parents=True, exist_ok=True)

    if not current_frame % movement_frame_delta:

        time = current_frame / fps

        with results_file_path.open("a", encoding="utf-8") as file:
            for object in scene_objects:
                id = object._id
                distance_traveled = object._distance_traveled_mm
                distance_to_cam = object._distance_to_lens_mm
                displacement_radial = object._displacement_radial_mm
                velocity = object._velocity_mms
                line = (
                    f"{id} {movement_frame_delta} {time} {distance_traveled}"
                    + f" {velocity} {distance_to_cam} {displacement_radial} \n"
                )
                file.write(line)

    global object_result_images
    global object_human_images

    if frame_list is not None:
        if current_frame in frame_list:
            cv.imwrite(
                f"{image_directory}/{ssr_file}_frame_{current_frame}_intensity_map.png",
                intensity_map_image,
            )
            cv.imwrite(
                f"{image_directory}/{ssr_file}_frame_{current_frame}_intensity_filter_mask.png",
                intensity_filter_mask_image,
            )
            cv.imwrite(
                f"{image_directory}/{ssr_file}_frame_{current_frame}_scene_top_down_view.png",
                scene_top_down_view_image,
            )
            cv.imwrite(
                f"{image_directory}/{ssr_file}_frame_{current_frame}_scene_top_down_view_mask.png",
                calibration_top_down_view_image,
            )
            cv.imwrite(
                f"{image_directory}/{ssr_file}_frame_{current_frame}_scene_boxed_top_down_view.png",
                scene_boxed_top_down_view_image,
            )

            for id, image in object_result_images:
                cv.imwrite(
                    f"{image_directory}/{ssr_file}_frame_{current_frame}_object_{id}_pop.png",
                    image,
                )

            for id, image in object_human_images:
                cv.imwrite(
                    f"{image_directory}/{ssr_file}_frame_{current_frame}_object_{id}_hog.png",
                    image,
                )

    object_result_images.clear()
    object_human_images.clear()


# * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ o ////////////////////////////////////////////////// * #
# * CODE ANALYSIS


if __name__ == "__main__":

    # Login "service" pass "CUST_SERV"
    IP = "169.254.214.10"
    PORT = 2114
    SSR_DATA = "ssr_files\H10_high_angle_mtk.ssr"

    PROFILER_ENABLED = True
    GLOBAL_PROFILE = False
    SHOW_3D = False

    LOOP_DATA = False
    START_FRAME = 0
    FRAME_COUNT = 0

    CALIBRATION_FRAMES = 100
    FRAMES_OF_INTEREST = [200, 400]

    profiler = cProfile.Profile()

    # ! MODIFICATION switched to new camera class
    camera = SafeVisionary2()
    data_generator = camera.stream()

    # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #
    # * VISUALIZATION

    def key_function(vis, a, m):
        # press "Shift+S" to save current view
        print(vis, a, m)
        if a and m:
            vis.capture_screen_image("vis_image.png")

    if SHOW_3D:
        visualizer = o3d.visualization.VisualizerWithKeyCallback()
        visualizer.register_key_action_callback(ord("S"), key_function)
        visualizer.create_window(
            window_name="3D Visualization of the Lower Half of the Depth Map:\tred: X\tgreen: Y\tblue: Z",
            width=960,
            height=1000,
        )

        render_control = visualizer.get_render_option()
        view_control = visualizer.get_view_control()

        render_control.background_color = np.array([0, 0, 0])
        render_control.point_size = 0.1

        view_control.camera_local_rotate(x=0.5, y=0)
        view_control.set_constant_z_far(13000)
        view_control.set_constant_z_near(0)
        view_control.rotate(x=5000, y=0)

        # coordinate frame: x, y, z rendered as red, green, blue arrows
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=150)
        visualizer.add_geometry(coord_frame)

        # geometry initiation
        point_cloud = o3d.geometry.PointCloud()
        visualizer.add_geometry(point_cloud)

    # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #

    if PROFILER_ENABLED and GLOBAL_PROFILE:
        profiler.enable()

    # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #
    # * GLOBAL FUNCTIONS TO ANALYSE

    setup(
        calibration_frames=CALIBRATION_FRAMES,
        on_the_run_frames=90,
        image_dimensions_xy=(512, 424),
        height_thresholds_mm=(0, 2000),
        distance_threshold_px=10,
        distance_threshold_mm=8,
        relative_armpit_height=0.7,
        minimum_contour_area=100,
        filter_sensitivity=5,
    )
    calibration_loop(data_generator)

    # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #

    if PROFILER_ENABLED and GLOBAL_PROFILE:
        profiler.disable()

    for cam_data, status in data_generator:

        if PROFILER_ENABLED:
            profiler.enable()

        # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #
        # * PER FRAME FUNCTIONS TO ANALYSE

        detection_loop(cam_data)
        show_results(0)
        save_data("TEST", FRAMES_OF_INTEREST)

        # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #

        if PROFILER_ENABLED:
            profiler.disable()

        # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #
        # * VISUALIZATION

        if SHOW_3D:
            point_cloud.points = o3d.utility.Vector3dVector(0.05 * scene_points_xyz)
            point_cloud.paint_uniform_color([1, 1, 1])

            visualizer.update_geometry(point_cloud)

            if not visualizer.poll_events() or not status:
                data_generator.send(True)

            visualizer.update_renderer()

        k = cv.waitKey(30) & 0xFF
        if k == 27:
            break

    cv.destroyAllWindows()
    if SHOW_3D:
        visualizer.destroy_window()

    # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #

    if PROFILER_ENABLED and GLOBAL_PROFILE:
        profiler.enable()

    # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #
    # * GLOBAL FUNCTIONS TO ANALYSE

    # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #

    if PROFILER_ENABLED and GLOBAL_PROFILE:
        profiler.disable()

    # * \\\\\\\\\\\\\\\\\\\\\\\\\ o ///////////////////////// * #
    # * ANALYTICS

    if PROFILER_ENABLED:
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats("tottime")
        stats.print_stats()

        file_path = Path(__file__)
        file_name = file_path.name

        substrings = [
            "Ordered by",
            "ncalls",
            "fit",
            "detectMultiScale",
            "segment_plane",
            "medianBlur",
            "findContours",
            "morphologyEx",
        ] + [file_name]

        profiler_directory = Path(
            f"{file_path.parent}/profiler results/{file_path.stem}"
        )
        profiler_directory.mkdir(parents=True, exist_ok=True)

        run = len(listdir(profiler_directory))

        profiler_file_path = (
            profiler_directory / f"{file_path.stem}_{date.today()}_{run}.txt"
        )

        with profiler_file_path.open("w", encoding="utf-8") as file:
            lines = s.getvalue().splitlines()
            for line in lines:
                for substring in substrings:
                    if substring in line:
                        line += "\n"
                        file.write(line)

# * \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\ o ////////////////////////////////////////////////// * #
