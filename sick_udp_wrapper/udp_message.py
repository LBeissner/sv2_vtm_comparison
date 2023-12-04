import numpy as np
from struct import unpack
from sick_udp_wrapper.udp_structure import UDPStructure
from sick_udp_wrapper.data_structures.message_header import MessageHeader
from pathlib import Path


class UDPMessage(UDPStructure):
    __telegram_id: np.uint8

    _BYTES: np.uint16 = 1460

    __header: MessageHeader
    __payload: bytes
    __checksum: np.uint8

    def __init__(self, data: bytes):

        # get the message header
        self.__header = MessageHeader(data=data)

        # create a parent and store the fragment index as id
        super().__init__(self.__header.content[1])

        # save fragment index and length of the data fragment in bytes
        self.__telegram_id = self.__header.content[0]

        # store the raw data bytestring
        self._raw_data = data

        # calculate payload and checksum offsets
        payload_offset = self.__header.get_length()
        checksum_offset = payload_offset + self.__header.payload_length()

        # store the data fragment
        self.__payload = data[payload_offset:checksum_offset]

        # store the crc checksum
        (self.__checksum,) = unpack("I", data[checksum_offset:])

    def __str__(self):
        return f"\n<< UDP Message >>\tID: {self._id}" + str(self.__header) + "\n"

    def get_content(self):
        return self.__payload

    def get_message_id(self):
        return self._id

    def get_telegram_id(self):
        return self.__telegram_id

    def store(self, storage_path: Path):
        file_name = f"{self.__telegram_id}_{self._id}_{self.__class__.__name__}.msgpack"
        super().store(storage_path, file_name)

    @classmethod
    def get_length(cls):
        return cls._BYTES
