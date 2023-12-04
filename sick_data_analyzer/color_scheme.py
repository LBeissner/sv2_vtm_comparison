import inspect
import numpy as np
from dataclasses import dataclass


@dataclass
class Color(object):
    __r: np.uint8
    __g: np.uint8
    __b: np.uint8

    def __init__(self, r: np.uint8, g: np.uint8, b: np.uint8):
        self.__r = r
        self.__g = g
        self.__b = b

    def __str__(self):
        return f"R: {str(self.__r).zfill(3)} G: {str(self.__g).zfill(3)} B: {str(self.__b).zfill(3)}\t{self.hex()}"

    def o3d_vector(self):
        return np.array([self.__r, self.__g, self.__b], dtype=np.float64) / 255

    def rgb(self):
        return np.array([self.__r, self.__g, self.__b], dtype=np.uint8)

    def bgr(self):
        return np.array([self.__b, self.__g, self.__r], dtype=np.uint8)

    def hex(self):
        return f"#{format(self.__r, '02x')}{format(self.__g, '02x')}{format(self.__b, '02x')}"

    def ocv_tuple(self):
        return (self.__b, self.__g, self.__r)


@dataclass
class ColorScheme(object):
    anthracite = Color(5, 5, 5)
    light_gray = Color(220, 220, 220)
    gray = Color(72, 72, 72)
    orange = Color(243, 117, 27)
    aquamarine = Color(51, 191, 219)
    red = Color(244, 51, 42)
    yellow = Color(244, 216, 37)

    @classmethod
    def table(cls, exclude: list[str] = ["anthracite"]):
        color_table: list = []
        for name, content in inspect.getmembers(cls):

            # remove protected functions and other methods
            if not name.startswith("_") and not inspect.ismethod(content):

                # exclude set colors
                if name not in exclude:
                    color_table.append(content)

        return color_table
