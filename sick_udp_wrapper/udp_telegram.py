import numpy as np
import cv2
import math
from pathlib import Path
from sick_udp_wrapper.udp_structure import UDPStructure
from sick_udp_wrapper.udp_message import UDPMessage
from sick_udp_wrapper.data_structures.telegram_header import TelegramHeader
from sick_udp_wrapper.data_structures.telegram_description import TelegramDescription
from sick_udp_wrapper.data_structures.telegram_image_data import TelegramImageData
from sick_udp_wrapper.data_structures.telegram_device_status import TelegramDeviceStatus
from sick_udp_wrapper.data_structures.telegram_roi import TelegramROI
from sick_udp_wrapper.data_structures.telegram_local_in_out import TelegramLocalInOut
from sick_udp_wrapper.data_structures.telegram_fields import TelegramFields
from sick_udp_wrapper.data_structures.telegram_logical_in_out import (
    TelegramLogicalInOut,
)


class UDPTelegram(UDPStructure):

    # number of fragments and bytes in a complete udp telegram
    __FRAGMENTS: int = 766

    # boolean to indicate a complete dataset in the telegram
    __success: bool

    # instances of the data segments contained in a telegram
    __header: TelegramHeader
    __description: TelegramDescription
    __image_data: TelegramImageData
    __device_status: TelegramDeviceStatus
    __rois: TelegramROI
    __local_io: TelegramLocalInOut
    __fields: TelegramFields
    __logical_io: TelegramLogicalInOut

    def __init__(
        self,
        success: bool,
        data: bytes,
        id: np.uint32,
    ):
        self.__success = success
        super().__init__(id)
        self._raw_data = data
        if success:
            self.unpack_data()

    @classmethod
    def from_fragments(cls, fragments: list, id: np.uint32):
        success, data = UDPTelegram.merge(fragments)
        return cls(success, data, id)

    @classmethod
    def from_file(cls, file_path: Path):
        unpacked_data = UDPStructure.load(file_path)

        number_of_fragments = math.ceil(len(unpacked_data) / UDPMessage.get_length())

        id: np.uint8 = None
        data: bytes = b""
        fragments: list = []
        success: bool = False

        if number_of_fragments == cls.__FRAGMENTS:

            for i in range(number_of_fragments):
                message_start = i * UDPMessage.get_length()
                message_end = (i + 1) * UDPMessage.get_length()

                udp_message = UDPMessage(unpacked_data[message_start:message_end])

                fragments.append(udp_message.get_content())

                if i == 0:
                    id = udp_message.get_telegram_id()

            success, data = UDPTelegram.merge(fragments)

        return cls(success, data, id)

    def complete(self) -> bool:
        """
        Returns True, if the telegram merged successfully.

        Returns:
            bool: Boolean stating the success of the merging of all telegram fragments.
        """
        return self.__success

    def __str__(self):
        return (
            f"\n<< UDP Telegram >>"
            + f"\n\tID:\t{self._id}"
            + f"\n\tBytes:\t{len(self._raw_data)}"
            + f"\n\t{self.__header}"
            + f"\n\t{self.__description}"
            + f"\n\t{self.__image_data}"
            + f"\n\t{self.__device_status}"
            + f"\n\t{self.__rois}"
            + f"\n\t{self.__local_io}"
            + f"\n\t{self.__fields}"
            + f"\n\t{self.__logical_io}"
            + "\n"
        )

    @staticmethod
    def merge(fragments: list) -> tuple[bool, bytes]:

        # initialize and fill data bytestring
        data = b""

        # continue only with correct number of fragments
        if len(fragments) != UDPTelegram.__FRAGMENTS:
            return False, data

        # check for missing fragments
        missing_fragments = len(
            list(
                filter(
                    lambda x: fragments[x] == None,
                    range(len(fragments)),
                )
            )
        )

        # continue only, if all fragments are available
        if missing_fragments:
            return False, data

        for fragment in fragments:
            data += fragment

        return True, data

    def unpack_data(self):

        # assign data variable to raw_data
        data = self._raw_data

        # retrieve telegram header to get segment offsets
        self.__header = TelegramHeader(data[:71])

        """
        get data segment offsets in the order:
        * description
        * image data
        * device status
        * regions of interest
        * local inputs and outputs
        * fields
        * logical inputs and outputs

        ? the offsets stored in the header are calculated from the telegram id field (offset 11 bytes)
        """
        offsets = np.asarray(self.__header.content[6::2]) + 11

        # get the data segments in the order stated above
        self.__description = TelegramDescription(data[offsets[0] : offsets[1]])
        self.__image_data = TelegramImageData(data[offsets[1] : offsets[2]])
        self.__device_status = TelegramDeviceStatus(data[offsets[2] : offsets[3]])
        self.__rois = TelegramROI(data[offsets[3] : offsets[4]])
        self.__local_io = TelegramLocalInOut(data[offsets[4] : offsets[5]])
        self.__fields = TelegramFields(data[offsets[5] : offsets[6]])
        self.__logical_io = TelegramLogicalInOut(data[offsets[6] :])

        self._BYTES = (
            self.__header.get_length()
            + self.__description.get_length()
            + self.__image_data.get_length()
            + self.__device_status.get_length()
            + self.__rois.get_length()
            + self.__local_io.get_length()
            + self.__fields.get_length()
            + self.__logical_io.get_length()
        )

        return True

    def get_camera_parameters(self):
        return self.__description.get_camera_parameters()

    def get_depth_map(self):
        return self.__image_data.get_depth_map()

    def get_depth_image(self, display: bool = False):
        depth_image = self.__image_data.get_depth_image()
        if display:
            cv2.imshow("Depth Image", depth_image)
        return depth_image

    def get_principal_point(self) -> tuple[int, int]:
        """_summary_

        Returns:
            tuple[int,int]: The principal point of the image
        """
        camera_parameters = self.get_camera_parameters()

        cx = int(round(camera_parameters.cx, 0))
        cy = int(round(camera_parameters.cy, 0))

        return (cx, cy)

    def get_intensity_map(self):
        return self.__image_data.get_intensity_map()

    def get_intensity_image(
        self, brightness=0.25, cross_hair=True, display: bool = False
    ):
        intensity_image = self.__image_data.get_intensity_image(brightness=brightness)

        if cross_hair:
            cx, cy = self.get_principal_point()

            line_thickness = 1
            x = (cx, intensity_image.shape[1])
            y = (cy, intensity_image.shape[0])
            cv2.line(
                intensity_image,
                (0, y[0]),
                (x[1], y[0]),
                (255, 255, 255),
                thickness=line_thickness,
            )
            cv2.line(
                intensity_image,
                (x[0], 0),
                (x[0], y[1]),
                (255, 255, 255),
                thickness=line_thickness,
            )

        if display:
            cv2.imshow("Intensity Image", intensity_image)
        return intensity_image

    def get_pixel_status_map(self):
        return self.__image_data.get_pixel_status_map()

    def store(self, storage_path: Path):
        file_name = f"{self._id}_{self.__class__.__name__}.msgpack"
        super().store(storage_path, file_name)

    def save_description_xml(self, path: Path):
        self.__description.save_description_xml(path)
