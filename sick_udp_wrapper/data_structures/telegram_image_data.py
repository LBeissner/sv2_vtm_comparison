import numpy as np
from sick_udp_wrapper.data_structures.data_segment import DataSegment


class TelegramImageData(DataSegment):
    """
    storage class for the telegram image data segment structured as follows:
        * segment start (segment length)
        * time stamp
        * structure version
        * image aquisition number
        * device status
        * flags
        * depth map
        * intensity map
        * pixel status map
        * checksum  (CRC-32)
        * segment end (segment length)

        ? byte order little-endian: <
    """

    __STRUCTURE: str = "<I Q H I B H 217088H 217088H 217088B 2I"
    _BYTES: np.uint32 = 1085469

    __MAX_INTENSITY: np.uint16 = 20000
    __MAX_DEPTH: np.uint16 = 16384
    __IMAGE_SHAPE: tuple = (424, 512)

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            length=TelegramImageData._BYTES,
            structure=TelegramImageData.__STRUCTURE,
        )

    def __str__(self):
        device_status: dict = {
            0: "Configuration",
            1: "Waiting for inputs",
            2: "Application stopped",
            3: "Normal operation",
        }

        depth_map_data: dict = {
            0: "Depth map unfiltered",
            1: "Depth map filtered",
        }

        detection_data: dict = {
            0: "Detection data invalid",
            1: "Detection data valid",
        }

        return (
            f"\n\t[INFO]\t{self.__class__.__name__}"
            + f"\n\t\tLength:\t\t\t{self.content[0]}"
            + f"\n\t\tTime Stamp:\t\t{self.time_stamp()}"
            + f"\n\t\tSegment Version:\t{hex(self.content[2])}"
            + f"\n\t\tCaptured Images:\t{self.content[3]}"
            + f"\n\t\tDevice Status:\t\t{device_status[self.content[4]]}"
            + f"\n\t\tDepth Map Data:\t\t{depth_map_data[bool(self.content[5] & 0b01)]}"
            + f"\n\t\tDetection Data:\t\t{detection_data[bool(self.content[5] & 0b10)]}"
        )

    def get_depth_map(self):
        depth_map = np.asarray(self.content[6:217094], dtype=np.float64)

        # calculate millimeters from quarter millimeters
        depth_map /= 4
        return np.reshape(depth_map, self.__IMAGE_SHAPE)

    def get_intensity_map(self):
        intensity_map = np.asarray(self.content[217094:434182], dtype=np.uint16)
        return np.reshape(intensity_map, self.__IMAGE_SHAPE)

    def get_pixel_status_map(self):
        pixel_status_map = np.asarray(self.content[434182:651270], dtype=np.uint8)
        return np.reshape(pixel_status_map, self.__IMAGE_SHAPE)

    def get_depth_image(self):
        depth_map = np.asarray(self.content[6:217094], dtype=np.uint16)
        depth_image = depth_map / self.__MAX_DEPTH * 255
        return np.reshape(depth_image, self.__IMAGE_SHAPE).astype(np.uint8)

    def get_intensity_image(self, brightness: float = 0.5):
        intensity_map = np.asarray(self.content[217094:434182], dtype=np.uint16)

        gamma = 0.5
        intensity_map_gamma = np.power(intensity_map, gamma)
        intensity_image = (
            intensity_map_gamma / intensity_map_gamma.max() * 255
        ).astype(np.uint8)

        # intensity_image = intensity_map * brightness * 100 / self.__MAX_INTENSITY * 255
        # intensity_image = intensity_map * brightness * 100 / self.__MAX_INTENSITY * 255
        intensity_image[intensity_image > 255] = 255
        return np.reshape(intensity_image, self.__IMAGE_SHAPE).astype(np.uint8)
