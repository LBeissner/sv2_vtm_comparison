import numpy as np
from sick_udp_wrapper.data_structures.data_segment import DataSegment


class TelegramFields(DataSegment):
    """
    storage class for the telegram fields data segment structured as follows :
        * segment start (segment length)
        * time stamp
        * structure version
        * data of the active fields
        * checksum (CRC-32)
        * segment end (segment length)

        ? byte order little-endian: <
    """

    __STRUCTURE: str = "< I Q H 80B 2I"
    _BYTES: np.uint8 = 102

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            length=TelegramFields._BYTES,
            structure=TelegramFields.__STRUCTURE,
        )
