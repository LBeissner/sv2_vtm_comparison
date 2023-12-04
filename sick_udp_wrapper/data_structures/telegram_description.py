import numpy as np
import re
from sick_udp_wrapper.data_structures.data_segment import DataSegment
from pathlib import Path
from common.Streaming.ParserHelper import CameraParameters


class TelegramDescription(DataSegment):
    """
    storage class for the telegram description data segment
    """

    _BYTES: np.uint16
    __xml_string: str
    __file_path: Path = None

    def __init__(self, data: bytes):
        self._BYTES = len(data)
        self.__xml_string = data.decode()

    def __str__(self):
        return (
            f"\n\t[INFO]\t{self.__class__.__name__}"
            + f"\n\t\tFile Path:\t{self.__file_path}"
            + f"\n\t\tthe xml file is stored in the workspace"
        )

    def get_camera_parameters(self):
        # shorten the xml string to the relevant section
        data_stream_xml = re.search(
            "<DataStream>(.*)</DataStream>", self.__xml_string
        ).group(1)

        # extract image dimensions and cast them into ints
        image_width = int(re.search("<Width>(.*)</Width>", data_stream_xml).group(1))
        image_height = int(re.search("<Height>(.*)</Height>", data_stream_xml).group(1))

        # get the focal to ray cross distance and cast it into a float
        focal_to_ray_cross = float(
            re.search("<FocalToRayCross>(.*)</FocalToRayCross>", data_stream_xml).group(
                1
            )
        )

        # extract the extrinsic camera parameters and cast them into floats
        extrinsics_xml = re.search(
            "<CameraToWorldTransform><value>(.*)</value></CameraToWorldTransform>",
            data_stream_xml,
        ).group(1)

        extrinsics_str = re.split(
            "</value><value>",
            extrinsics_xml,
        )

        extrinsics = [float(i) for i in extrinsics_str]

        # extract the intrinsic camera parameters and cast them into floats
        intrinsics_xml = re.search(
            "<CameraMatrix>(.*)</CameraMatrix>", data_stream_xml
        ).group(1)

        intrinsics_fx = float(re.search("<FX>(.*)</FX", intrinsics_xml).group(1))
        intrinsics_fy = float(re.search("<FY>(.*)</FY", intrinsics_xml).group(1))
        intrinsics_cx = float(re.search("<CX>(.*)</CX", intrinsics_xml).group(1))
        intrinsics_cy = float(re.search("<CY>(.*)</CY", intrinsics_xml).group(1))

        # extract two of five distortion coefficients and cast them into floats
        distortion_xml = re.search(
            "<CameraDistortionParams>(.*)</CameraDistortionParams>",
            data_stream_xml,
        ).group(1)

        distortion_k1 = float(re.search("<K1>(.*)</K1", distortion_xml).group(1))
        distortion_k2 = float(re.search("<K2>(.*)</K2", distortion_xml).group(1))

        # cast all camera parameters into a CameraParameter object
        camera_parameters = CameraParameters(
            width=image_width,
            height=image_height,
            cam2worldMatrix=extrinsics,
            fx=intrinsics_fx,
            fy=intrinsics_fy,
            cx=intrinsics_cx,
            cy=intrinsics_cy,
            k1=distortion_k1,
            k2=distortion_k2,
            f2rc=focal_to_ray_cross,
        )

        return camera_parameters

    def save_description_xml(self, path: Path):
        file_path = path / "main.xml"
        camera_xml = re.sub("><", ">\n<", self.__xml_string)

        with file_path.open("w") as file:
            file.write(camera_xml)
            file.close()
