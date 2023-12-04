import cv2
import numpy as np
from copy import deepcopy
from dataclasses import dataclass
from common.Streaming.Data import Data
from common.Streaming.ParserHelper import CameraParameters
from sick_udp_wrapper.udp_telegram import UDPTelegram


@dataclass
class CamData(object):
    """
    Storage class to resemble the CamData class written by Karsten Flores. The class variables are named to ensure compatibility with the
    pop algorithm. These names are technically wrong, because the variables store maps instead of images in greyscale or color spaces.
    """

    dist_img: np.ndarray = None
    ints_img: np.ndarray = None
    cnfi_img: np.ndarray = None
    cam_params: CameraParameters = None
    depth_to_distance: bool = False

    @classmethod
    def from_udp(cls, telegram: UDPTelegram, roi: np.ndarray = None):
        depth_map = telegram.get_depth_map()
        intensity_map = telegram.get_intensity_map()
        status_map = telegram.get_pixel_status_map()
        camera_parameters = telegram.get_camera_parameters()
        instance = cls(depth_map, intensity_map, status_map, camera_parameters)

        if roi is not None:
            instance.apply_roi(roi)

        return instance

    @classmethod
    def from_ssr(
        cls,
        ssr: tuple[list, list, list, CameraParameters, bool],
        index: np.uint16,
        roi: np.ndarray = None,
    ):
        depth_map = ssr[0][index]
        intensity_map = ssr[1][index]
        status_map = ssr[2][index]
        camera_parameters = deepcopy(ssr[3])

        instance = cls(depth_map, intensity_map, status_map, camera_parameters)

        if roi is not None:
            instance.apply_roi(roi)

        return instance

    @classmethod
    def from_frame(cls, frame: bytearray, roi: np.ndarray = None):
        data = Data()
        data.read(frame, True)
        camera_parameters = deepcopy(data.cameraParams)
        height = camera_parameters.height
        width = camera_parameters.width

        depth_map = np.uint16(np.reshape(list(data.depthmap.distance), (height, width)))
        intensity_map = np.uint16(
            np.reshape(list(data.depthmap.intensity), (height, width))
        )
        status_map = np.uint16(
            np.reshape(list(data.depthmap.confidence), (height, width))
        )

        instance = cls(depth_map, intensity_map, status_map, camera_parameters)

        if roi is not None:
            instance.apply_roi(roi)

        return instance

    def apply_roi(self, roi: np.ndarray):
        """
            Applies the boundaries of an rectangular region of interest on the CamData object.

        Args:
            roi (np.ndarray): Region of interest. The array has the structure [[x0  x1] [y0  y1]], where
                              [x0, y0] is the upper left box corner and [x1, y1] is the bottom right corner.
        """

        self.dist_img = self.dist_img[roi[1, 0] : roi[1, 1], roi[0, 0] : roi[0, 1]]

        self.ints_img = self.ints_img[roi[1, 0] : roi[1, 1], roi[0, 0] : roi[0, 1]]

        self.cnfi_img = self.cnfi_img[roi[1, 0] : roi[1, 1], roi[0, 0] : roi[0, 1]]

        self.cam_params.cx -= roi[0, 0]
        self.cam_params.cy -= roi[1, 0]

        self.cam_params.width = roi[0, 1] - roi[0, 0]
        self.cam_params.height = roi[1, 1] - roi[1, 0]

    def __str__(self):
        return (
            f"\n<< CamData >>"
            + f"\n\tConverted to Distance:\t{self.depth_to_distance}"
            + f"\n\tDepth Map:\t{self.dist_img.shape}\tRange:\t{self.dist_img.min()} - {self.dist_img.max()}"
            + f"\n\tIntensity Map:\t{self.ints_img.shape}\tRange:\t{self.ints_img.min()} - {self.ints_img.max()}"
            + f"\n\tStatus Map:\t{self.cnfi_img.shape}\tRange:\t{self.cnfi_img.min()} - {self.cnfi_img.max()}"
            + "\n"
            + "\n\tCamera Parameters:"
            + f"\n\t\tImage Shape:\t\t({self.cam_params.width}, {self.cam_params.height})"
            + f"\n\t\tFocal Lengths:\t\t({self.cam_params.fx}, {self.cam_params.fy})"
            + f"\n\t\tPrincipal Point:\t({self.cam_params.cx}, {self.cam_params.cy})"
            + f"\n\t\tDistortion Parameters:\t({self.cam_params.k1}, {self.cam_params.k2}, 0)"
            + f"\n\t\tFocal to Ray Cross:\t{self.cam_params.f2rc} mm"
            + "\n"
        )

    def get_principal_point(self):
        cx = int(round(self.cam_params.cx, 0))
        cy = int(round(self.cam_params.cy, 0))
        return (cx, cy)

    def get_distortion_vector(self):
        return np.array([self.cam_params.k1, self.cam_params.k2, 0.0, 0.0])

    def get_camera_matrix(self):
        return np.array(
            [
                [self.cam_params.fx, 0, self.cam_params.cx],
                [0, self.cam_params.fy, self.cam_params.cy],
                [0, 0, 1],
            ]
        )

    def point_cloud(self):

        # code based on the sick_wrapper.cam.dist_to_npy method (c) Karsten Flores

        cx = self.cam_params.cx
        cy = self.cam_params.cy
        fx = self.cam_params.fx
        fy = self.cam_params.fy
        k1 = self.cam_params.k1
        k2 = self.cam_params.k2
        f2rc = self.cam_params.f2rc
        width = self.cam_params.width
        height = self.cam_params.height

        m_c2w = np.asarray(self.cam_params.cam2worldMatrix).reshape(4, 4)
        m_d = self.dist_img

        coordinates_grid_xy = np.array(
            np.meshgrid(
                range(width),
                range(height),
                indexing="xy",
            )
        )

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
                np.ones(width * height),
            ),
            axis=-1,
        )

        # [X_c, Y_c, Z_c] to [X, Y, Z]
        m_w = (m_c2w @ m_c.T)[0:3, :].T
        m_w: np.ndarray = np.zeros(m_w.shape) + m_w

        return m_w

    def depth_to_distance_map(self, transformation: np.ndarray = None):
        point_cloud_points = self.point_cloud()
        if transformation is not None:
            point_cloud_points = np.c_[
                point_cloud_points, np.ones(point_cloud_points.shape[0])
            ]
            point_cloud_points = (transformation @ point_cloud_points.T).T
        distance_map = point_cloud_points[:, 2]
        distance_map[distance_map <= 0] = 0

        self.dist_img = np.reshape(
            distance_map, (self.cam_params.height, self.cam_params.width)
        )
        self.depth_to_distance = True


@dataclass
class CamImage(object):

    __depth_image: np.ndarray = None
    __intensity_image: np.ndarray = None
    __status_image: np.ndarray = None
    __shape: tuple[int, int] = None
    principal_point: tuple[int, int] = None

    @classmethod
    def from_udp(cls, telegram: UDPTelegram):
        instance = cls()

        camera_parameters = telegram.get_camera_parameters()
        instance.__shape = (camera_parameters.height, camera_parameters.width)
        instance.principal_point = telegram.get_principal_point()

        depth_map = telegram.get_depth_map()
        instance.__depth_image = cls.__map_to_grayscale(depth_map)
        intensity_map = telegram.get_intensity_map()
        instance.__intensity_image = cls.__map_to_grayscale(intensity_map)

        status_map = telegram.get_pixel_status_map()
        instance.__status_image = cls.__map_to_grayscale(status_map)

        return instance

    @classmethod
    def from_cam_data(cls, cam_data: CamData):
        instance = cls()

        instance.__shape = (cam_data.cam_params.height, cam_data.cam_params.width)
        instance.principal_point = cam_data.get_principal_point()

        instance.__depth_image = np.reshape(
            cls.__map_to_grayscale(cam_data.dist_img), instance.__shape
        )
        instance.__intensity_image = np.reshape(
            cls.__map_to_grayscale(cam_data.ints_img), instance.__shape
        )
        instance.__status_image = np.reshape(
            cls.__map_to_grayscale(cam_data.cnfi_img), instance.__shape
        )

        return instance

    def __str__(self):
        return (
            f"\n<< ImageData >>"
            + f"\n\tDepth Map:\t{self.__depth_image.shape}\tRange:\t{self.__depth_image.min()} - {self.__depth_image.max()}"
            + f"\n\tIntensity Map:\t{self.__intensity_image.shape}\tRange:\t{self.__intensity_image.min()} - {self.__intensity_image.max()}"
            + f"\n\tStatus Map:\t{self.__status_image.shape}\tRange:\t{self.__status_image.min()} - {self.__status_image.max()}"
            + f"\n"
            + f"\n\tImage Parameters:"
            + f"\n\t\tImage Shape:\t\t({self.__shape[1]}, {self.__shape[0]})"
            + f"\n\t\tPrincipal Point:\t({self.principal_point[0]}, {self.principal_point[1]})"
            + "\n"
        )

    @staticmethod
    def __map_to_grayscale(map: np.ndarray, stretch: bool = False) -> np.ndarray:

        image = map

        if stretch:
            # resolve only the range between smallest and largest array value in grayscale between 0 and 255
            image = image - image.min()

        # scale the array values to grayscale
        return (image / image.max() * 255).astype(np.uint8)

    def __add_cross_hair(self, image: np.ndarray) -> np.ndarray:
        cx, cy = self.principal_point

        line_thickness = 1
        x = (cx, image.shape[1])
        y = (cy, image.shape[0])

        cv2.line(
            image,
            (0, y[0]),
            (x[1], y[0]),
            (255, 255, 255),
            thickness=line_thickness,
        )
        cv2.line(
            image,
            (x[0], 0),
            (x[0], y[1]),
            (255, 255, 255),
            thickness=line_thickness,
        )
        return image

    def __fit_to_size(
        self, image, target_shape: tuple[np.uint16, np.uint16] = (424, 512)
    ) -> cv2.Mat:

        # fit image to target size without distortion
        scaling_factor = np.min(
            np.floor_divide(np.array(target_shape), np.array(self.__shape))
        )

        # calculate new shape, reverse tuple to match openCV input criteria
        target_shape = tuple(scaling_factor * length for length in self.__shape)[::-1]

        # return resized image
        return cv2.resize(image, target_shape, interpolation=cv2.INTER_NEAREST)

    def set_intensity(self, intensity_image: np.ndarray):
        self.__intensity_image = intensity_image

    def set_depth(self, depth_image: np.ndarray):
        self.__depth_image = depth_image

    def set_status(self, status_image: np.ndarray):
        self.__status_image = status_image

    def depth(
        self,
        target_shape: tuple[np.uint16, np.uint16] = (424, 512),
        add_cross_hair: bool = False,
    ):
        image = self.__depth_image

        if add_cross_hair:
            image = self.__add_cross_hair(image)

        if target_shape:
            image = self.__fit_to_size(image, target_shape)

        return image

    def intensity(
        self,
        target_shape: tuple[np.uint16, np.uint16] = (424, 512),
        add_cross_hair: bool = False,
    ):
        image = self.__intensity_image

        if add_cross_hair:
            image = self.__add_cross_hair(image)

        if target_shape:
            image = self.__fit_to_size(image, target_shape)

        return image

    def state(
        self,
        target_shape: tuple[np.uint16, np.uint16] = (424, 512),
        add_cross_hair: bool = False,
    ):
        image = self.__status_image

        if add_cross_hair:
            image = self.__add_cross_hair(image)

        if target_shape:
            image = self.__fit_to_size(image, target_shape)

        return image

    def apply_otsu_threshold(self, target="intensity"):

        if target == "intensity":
            image = self.__intensity_image
        elif target == "depth":
            image = self.__depth_image
        else:
            raise AttributeError(
                "Invalid thresholding target! Please select a valid target for thresholding."
            )

        # Otsu's thresholding after Gaussian filtering
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
        _, threshold_mask = cv2.threshold(
            blurred_image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        self.__depth_image *= threshold_mask
        self.__intensity_image *= threshold_mask

        return threshold_mask
