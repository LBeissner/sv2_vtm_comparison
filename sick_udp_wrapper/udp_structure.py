import numpy as np
import msgpack
from pathlib import Path


class UDPStructure(object):
    _id: np.uint32
    _raw_data: bytes
    _clean: bool
    _BYTES: np.uint32

    def __init__(self, id: np.uint32):
        self._id = id
        pass

    def store(self, storage_path: Path, file_name: str):
        file_path = storage_path / file_name
        with file_path.open("wb") as file:
            packed_data = msgpack.packb(self._raw_data)
            file.write(packed_data)

    @staticmethod
    def load(file_path: Path):
        unpacked_data: bytes = b""
        with file_path.open("rb") as file:
            packed_data = file.read()
            unpacked_data = msgpack.unpackb(packed_data)
        return unpacked_data
