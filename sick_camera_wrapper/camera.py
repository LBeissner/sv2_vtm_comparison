# sick python modules
from common.Streaming.ParserHelper import CameraParameters
from common.data_io import SsrLoader
from common.Stream import Streaming

# custom modules
from sick_udp_wrapper.udp_message import UDPMessage
from sick_udp_wrapper.udp_telegram import UDPTelegram
from sick_camera_wrapper.cam_data import CamData

# python libraries
import socket
from datetime import datetime
import re
import numpy as np

from pathlib import Path
from struct import unpack
import msgpack

from typing import Generator
from time import time


def percent(numerator, denominator):
    return round(numerator / denominator * 100, 2)


class Camera(object):
    # device information
    _NAME: str = None
    _DEVICE_IP: str = None
    _DEVICE_PORT: np.uint16 = None

    # time stamp of instance creation
    _time_stamp: str = None

    # record data
    _PARENT_PATH: Path = Path.cwd() / "camera_data_storage"
    _RECORD_PATH: Path = None

    def __init__(self, name: str, ip: str, port: np.uint16):
        self._NAME = name
        self._DEVICE_IP = ip
        self._DEVICE_PORT = port

        time_stamp = str(datetime.now())
        self._time_stamp = re.sub(":", "-", time_stamp[: time_stamp.rindex(".")])

    def __str__(self):
        return (
            f"\n<< Camera >>"
            + f"\n\tDevice:\t{self._NAME}"
            + f"\n\tDevice IP:\t{self._DEVICE_IP}"
            + f"\n\tDevice Port:\t{self._DEVICE_PORT}"
            + f"\n\tDate/Time:\t{self._time_stamp}"
        )

    @staticmethod
    def _depth_map_to_point_cloud(
        depth_map: np.ndarray,
        camera_parameters: CameraParameters,
    ):

        # code based on the sick_wrapper.cam.dist_to_npy method (c) Karsten Flores
        # * changed to work with numba njit

        cx = camera_parameters.cx
        cy = camera_parameters.cy
        fx = camera_parameters.fx
        fy = camera_parameters.fy
        k1 = camera_parameters.k1
        k2 = camera_parameters.k2
        f2rc = camera_parameters.f2rc
        width = camera_parameters.width
        height = camera_parameters.height

        m_c2w = np.asarray(camera_parameters.cam2worldMatrix).reshape(4, 4)
        m_d = depth_map

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


class SafeVisionary2(Camera):

    # server socket
    __server_socket: socket.socket
    __BUFFER_SIZE: np.uint16 = 1460

    # data output
    __OUTPUT_MODES = ["UDP Telegram", "CamData", "Point Cloud"]

    # data counter for evaluation
    __messages_total: int = 0
    __telegrams_total: int = 0
    __telegrams_clean: int = 0
    __telegrams_error: int = 0

    def __init__(self, ip: str = "192.168.1.1", port: np.uint16 = 5005):
        super().__init__(name="safeVisionary2", ip=ip, port=port)

        # create socket
        self.__server_socket = socket.socket(
            family=socket.AF_INET, type=socket.SOCK_DGRAM
        )

        self._PARENT_PATH = self._PARENT_PATH / "safeVisionary2"

    def evaluate(self):
        print(
            f"\n<< Evaluation >>"
            + f"\n\tMessages Total:\t\t{self.__messages_total}"
            + f"\n\tMessages Clean:\t\tWIP"
            + f"\n\tMessages Error:\t\tWIP"
            + f"\n\tTelegrams Total:\t{self.__telegrams_total}"
            + f"\n\tTelegrams Clean:\t{self.__telegrams_clean}\t{percent(self.__telegrams_clean,self.__telegrams_total)}"
            + f"\n\tTelegrams Error:\t{self.__telegrams_error}\t{percent(self.__telegrams_error,self.__telegrams_total)}"
        )

    def record(self, duration: np.uint16 = 30):

        # initialize socket connection
        self.__server_socket.bind((self._DEVICE_IP, self._DEVICE_PORT))

        # define and create storage directory
        time_stamp = re.sub("\s", "_", re.sub("-", "", self._time_stamp))
        self._RECORD_PATH = self._PARENT_PATH / ("record_" + time_stamp)
        self._RECORD_PATH.mkdir(parents=True, exist_ok=True)

        messages: dict = {}
        timeout = time() + duration

        while time() < timeout:
            data, _ = self.__server_socket.recvfrom(self.__BUFFER_SIZE)
            ids = unpack(">2H", data[0:4])
            message_stamp = ids
            messages[message_stamp] = data

        message_stamps = list(messages.keys())
        message_stamps.sort()
        sorted_messages = {i: messages[i] for i in message_stamps}

        telegram_ids = list({telegram_id for telegram_id, _ in sorted_messages.keys()})
        telegram_ids.sort()

        for telegram_id in telegram_ids:
            unpacked_data = b""
            for message_stamp, data in sorted_messages.items():
                parent_telegram_id, message_id = message_stamp

                if parent_telegram_id == telegram_id:
                    unpacked_data += data

            file_name = self._RECORD_PATH / (f"{telegram_id}_UDPTelegram.msgpack")
            with file_name.open("wb") as file:
                packed_data = msgpack.packb(unpacked_data)
                file.write(packed_data)

        for file_path in self._RECORD_PATH.iterdir():

            telegram = UDPTelegram.from_file(file_path)

            if telegram.complete():
                telegram.save_description_xml(self._RECORD_PATH)
                break

    def play_file(
        self,
        file_name: str,
        output_mode: str = "UDP Telegram",
        roi: np.ndarray = None,
        loop: bool = False,
    ) -> Generator[UDPTelegram, bool, str]:

        if output_mode not in self.__OUTPUT_MODES:
            raise AttributeError(
                f"Invalid output data type selected! Select one of the following:\t{self.__OUTPUT_MODES}"
            )

        # define source directory
        path = self._PARENT_PATH / file_name

        while True:
            file_paths = path.glob(
                "**/*.msgpack",
            )

            for file_path in file_paths:

                telegram = UDPTelegram.from_file(file_path)

                if telegram.complete():
                    if output_mode == "UDP Telegram":
                        yield telegram, None
                    elif output_mode == "CamData" or output_mode == "Point Cloud":
                        camera_data = CamData.from_udp(telegram, roi)
                        if output_mode == "Point Cloud":
                            point_cloud = camera_data.point_cloud()
                            yield point_cloud, None
                            pass
                        else:
                            yield camera_data, None
            if not loop:
                break

    def stream(self) -> Generator[UDPTelegram, bool, str]:
        self.__server_socket.bind((self._DEVICE_IP, self._DEVICE_PORT))
        status = True

        while True:

            telegram_fragments = []
            telegram: UDPTelegram = None
            current_telegram_id: int = None

            while True:
                data, _ = self.__server_socket.recvfrom(self.__BUFFER_SIZE)
                udp_data = UDPMessage(data)
                self.__messages_total += 1

                telegram_id = udp_data.get_telegram_id()

                if current_telegram_id is None:
                    current_telegram_id = telegram_id

                # merge fragments in continuous telegram on id change
                if current_telegram_id != telegram_id:

                    telegram = UDPTelegram.from_fragments(
                        telegram_fragments, current_telegram_id
                    )

                    self.__telegrams_total += 1

                    if telegram.complete():
                        self.__telegrams_clean += 1
                        break
                    else:
                        self.__telegrams_error += 1
                    telegram_fragments = []
                else:
                    telegram = None

                current_telegram_id = telegram_id

                fragment_id = udp_data.get_message_id()
                fragment = udp_data.get_content()

                # sort telegram fragments into a list
                if len(telegram_fragments) > fragment_id:
                    telegram_fragments[fragment_id] = fragment
                else:
                    while len(telegram_fragments) <= fragment_id:
                        telegram_fragments.append(None)
                    telegram_fragments[fragment_id] = fragment

            yield telegram, status


class VisionaryTMini(Camera):
    __OUTPUT_MODES: list[str] = ["CamData", "Point Cloud"]

    def __init__(self, ip: str = "192.168.1.2", port: np.uint16 = 2114):
        super().__init__(name="Visionary-T Mini", ip=ip, port=port)

        self._PARENT_PATH = self._PARENT_PATH / self._NAME

    def play_file(
        self,
        file_name: str,
        output_mode: str = "CamData",
        roi: np.ndarray = None,
        loop: bool = False,
    ) -> Generator[tuple, bool, str]:
        if output_mode not in self.__OUTPUT_MODES:
            raise AttributeError(
                f"Invalid output data type selected! Select one of the following:\t{self.__OUTPUT_MODES}"
            )

        if not file_name.endswith(".ssr"):
            file_name += ".ssr"

        # define source directory
        path = self._PARENT_PATH / file_name

        status = True

        ssr_data = SsrLoader.readSsrData(path, 0, 0)
        ssr_len = len(ssr_data[0])
        frame_no = 0

        while True:
            camera_data = CamData.from_ssr(ssr=ssr_data, index=frame_no, roi=roi)
            frame_no += 1

            if output_mode == "CamData":
                send_stop = yield camera_data, status
            elif output_mode == "Point Cloud":
                point_cloud = camera_data.point_cloud()
                send_stop = yield point_cloud, status

            del camera_data

            if frame_no >= ssr_len:
                status = loop
                frame_no = 0

            if not status or send_stop:
                del ssr_data
                if send_stop:
                    yield
                break

    def stream(
        self, output_mode: str = "CamData", roi: np.ndarray = None
    ) -> Generator[tuple, bool, str]:
        if output_mode not in self.__OUTPUT_MODES:
            raise AttributeError(
                f"Invalid output data type selected! Select one of the following:\t{self.__OUTPUT_MODES}"
            )

        device = Streaming(self._DEVICE_IP, self._DEVICE_PORT)
        device.openStream()
        device.sendBlobRequest()
        camera_data = CamData()
        status = True

        while True:
            try:
                device.getFrame()
                camera_data = CamData.from_frame(device.frame, roi)
            except:
                status = False

            if output_mode == "CamData":
                send_stop = yield camera_data, status
            elif output_mode == "Point Cloud":
                point_cloud = camera_data.point_cloud()
                send_stop = yield point_cloud, status
            if not status or send_stop:
                device.closeStream()
                if send_stop:
                    yield
                break
