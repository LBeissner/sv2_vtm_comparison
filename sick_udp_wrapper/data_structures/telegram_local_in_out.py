import numpy as np
from sick_udp_wrapper.data_structures.data_segment import DataSegment


class TelegramLocalInOut(DataSegment):
    """
    storage class for the telegram local inputs and outputs data segment structured as follows :
        * segment start (segment length)
        * time stamp
        * structure version
        * configured universal I/Os
        * universal I/O used as input or output
        * logical state of the universal I/Os (inputs)
        * state of the universal I/Os (outputs)
        * Logical state of the OSSDs
        * checksum (CRC-32)
        * segment end (segment length)

        ? byte order little-endian: <
    """

    __STRUCTURE: str = "<I Q 4H 16B 1B 11B 2I"
    _BYTES: np.uint8 = 56

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            length=TelegramLocalInOut._BYTES,
            structure=TelegramLocalInOut.__STRUCTURE,
        )
