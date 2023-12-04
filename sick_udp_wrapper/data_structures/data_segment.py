import numpy as np
from struct import unpack
from crc import Calculator, Configuration, Crc32


class DataSegment(object):
    """
    parent class for data elements
    """

    _BYTES: np.uint32

    content: list

    def __init__(self, data: bytes, length: np.uint32, structure: str):
        self.content = unpack(structure, data[0:length])

    def __str__(self):
        return (
            f"\n\t[INFO]\t{self.__class__.__name__}"
            + f"\n\t\tLength:\t\t\t{self.content[0]}"
            + f"\n\t\tTime Stamp:\t\t{self.time_stamp()}"
            + f"\n\t\tSegment Version:\t{hex(self.content[2])}"
        )

    def time_stamp(self):

        # TODO Da stimmt noch was nicht mit den Bit Shifts!
        bitmask_12b = 0b111111111111  # year
        bitmask_10b = 0b1111111111  # milliseconds
        bitmask_6b = 0b111111  # seconds, minutes
        bitmask_5b = 0b11111  # hours, days
        bitmask_4b = 0b1111  # month

        time_binary = int(self.content[1])

        milliseconds = np.uint16((time_binary >> 0) & bitmask_10b)
        seconds = np.uint8((time_binary >> 10) & bitmask_6b)
        minutes = np.uint8((time_binary >> 16) & bitmask_6b)
        hours = np.uint8((time_binary >> 22) & bitmask_5b)
        day = np.uint8((time_binary >> 38) & bitmask_5b)
        month = np.uint8((time_binary >> 43) & bitmask_4b)
        year = np.uint8((time_binary >> 47) & bitmask_12b)

        return f"{hours}:{minutes}:{seconds}:{milliseconds}\t{year}-{month}-{day}"

    def get_length(self):
        return self._BYTES

    def checksum(self, data: bytes, expected: hex):
        """Calculates the checksum for the bytes given. It is calculated exclusively over the Data field.

        Args:
            data (bytes): Bytes to verify the checksum for.
            expected (hex): The expected value of the checksum.
        """

        configuration = Configuration(
            width=32,
            polynomial=0x04C11DB7,
            init_value=0xFFFFFF,
            final_xor_value=0xFFFFFF,
            reverse_input=False,
            reverse_output=False,
        )

        calculator = Calculator(configuration)

        return calculator.verify(data, expected)
