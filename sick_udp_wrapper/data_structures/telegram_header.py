import numpy as np
from sick_udp_wrapper.data_structures.data_segment import DataSegment


class TelegramHeader(DataSegment):
    """
    storage class for the telegram header containing information about:
        * telegram start
        * length of the telegram counted starting at the protocol version
        * protocol version
        * package type
        * telegram id
        * number of data segments
        * segment information (segment offset, change counter)

        ? byte order big-endian: >
    """

    __STRUCTURE: str = ">2I H B 2H 14I"
    _BYTES: np.uint8 = 71

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            length=TelegramHeader._BYTES,
            structure=TelegramHeader.__STRUCTURE,
        )

    def __str__(self):
        return (
            f"\n\t[INFO]\t{self.__class__.__name__}"
            + f"\n\t\tTelegram Starter:\t{hex(self.content[0])}"
            + f"\n\t\tTelegram Length:\t{self.content[1]}\tBytes"
            + f"\n\t\tProtocal Version:\t{self.content[2]}"
            + f"\n\t\tPackage Type:\t\t{hex(self.content[3])}"
            + f"\n\t\tTelegram ID:\t\t{self.content[4]}"
            + f"\n\t\tNumber of Segments:\t{self.content[5]}"
        )
