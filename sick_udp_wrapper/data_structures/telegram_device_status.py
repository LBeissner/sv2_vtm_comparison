import numpy as np
from sick_udp_wrapper.data_structures.data_segment import DataSegment


class TelegramDeviceStatus(DataSegment):
    """
    storage class for the telegram image data structured as follows:
        * segment start (segment length)
        * time stamp
        * structure version
        * image aquisition number
        * device status
        * cut-off path (safety-oriented)
        * cut-off path (not safety-oriented)
        * active monitoring case
        * front screen degree of contamination
        * checksum
        * segment end (segment length)

        ? byte order little-endian: <
    """

    __STRUCTURE: str = "<I Q 2H 4I B 2I"
    _BYTES: np.uint8 = 41

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            length=TelegramDeviceStatus._BYTES,
            structure=TelegramDeviceStatus.__STRUCTURE,
        )
