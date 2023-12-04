import numpy as np
from sick_udp_wrapper.data_structures.data_segment import DataSegment


class MessageHeader(DataSegment):
    """
    storage class for the message header containing information about:
        * telegram number
        * fragment number
        * time stamp
        * source ip address
        * source port
        * destination ip address
        * destination port
        * protocol version
        * payload length
        * payload flags
        * package type

        ? byte order big-endian: >
    """

    __STRUCTURE: str = ">2H I 4B H 4B 3H 2B"
    _BYTES: np.uint8 = 26

    def __init__(self, data: bytes):
        super().__init__(
            data=data,
            length=MessageHeader._BYTES,
            structure=MessageHeader.__STRUCTURE,
        )

    def __str__(self):
        return (
            f"\n\t[INFO]\t{self.__class__.__name__}"
            + f"\n\t\tTelegram:\t{self.content[0]}"
            + f"\n\t\tFragment:\t{self.content[1]}"
            + f"\n\t\tSource:\t\t{self.content[3]}.{self.content[4]}.{self.content[5]}.{self.content[6]}:{self.content[7]}"
            + f"\n\t\tDestination:\t{self.content[8]}.{self.content[9]}.{self.content[10]}.{self.content[11]}:{self.content[12]}"
            + f"\n\t\tPayload Length:\t{self.content[14]} of 1430\t\tBytes"
        )

    def payload_length(self):
        return self.content[14]
