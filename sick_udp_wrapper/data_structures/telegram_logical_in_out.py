import numpy as np
from sick_udp_wrapper.data_structures.data_segment import DataSegment


class TelegramLogicalInOut(DataSegment):
    """
    storage class for the telegram logical inputs and outputs data segment structured as follows :
        * segment start (segment length)
        * time stamp
        * structure version
        * logical I/O data
        * checksum (CRC-32)
        * segment end (segment length)

        ? byte order little-endian: <
    """

    __STRUCTURE: str = "<I Q H 120B 2I"
    _BYTES: np.uint8 = 142

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            length=TelegramLogicalInOut._BYTES,
            structure=TelegramLogicalInOut.__STRUCTURE,
        )
