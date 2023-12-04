import numpy as np
from sick_udp_wrapper.data_structures.data_segment import DataSegment


class TelegramROI(DataSegment):
    """
    storage class for the telegram regions of interest data segment structured as follows :
        * segment start (segment length)
        * time stamp
        * structure version
        * logical I/O data
        * checksum (CRC-32)
        * segment end (segment length)

        ? byte order little-endian: <
    """

    __STRUCTURE: str = "<I Q H 30B 2I"
    _BYTES: np.uint8 = 52

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            length=TelegramROI._BYTES,
            structure=TelegramROI.__STRUCTURE,
        )
