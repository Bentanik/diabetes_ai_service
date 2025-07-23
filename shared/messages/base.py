from enum import Enum


class BaseResultCode(Enum):
    """
    Class cơ sở cho tất cả các result code enum
    Định nghĩa cách thức hoạt động chung cho .code và .message
    """

    def __new__(cls, code: str, message: str):
        obj = object.__new__(cls)
        obj._value_ = code
        obj._code = code
        obj._message = message
        return obj

    @property
    def code(self) -> str:
        return self._code

    @property
    def message(self) -> str:
        return self._message
