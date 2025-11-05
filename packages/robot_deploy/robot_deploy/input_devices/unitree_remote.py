import contextlib
import numpy as np
import struct

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG

from .input_device import Button, InputDevice

REMOTE_BUTTON_ORDER = [
    Button.R1,
    Button.L1,
    Button.start,
    Button.select,
    Button.R2,
    Button.L2,
    Button.F1,
    Button.F2,
    Button.A,
    Button.B,
    Button.X,
    Button.Y,
    Button.up,
    Button.right,
    Button.down,
    Button.left,
]


class UnitreeRemoteDevice(InputDevice):
    def __init__(self, net_interface: str | None) -> None:
        super().__init__()

        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0

        if net_interface is not None:
            with contextlib.suppress(Exception):
                ChannelFactoryInitialize(0, net_interface)
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self._read_command_cb, 10)

    def get_command(self) -> np.ndarray:
        with self.lock:
            return np.clip([self.ly, -self.lx, -self.rx], -1, 1)

    def _read_command_cb(self, msg: LowStateHG) -> None:
        data = msg.wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        with self.lock:
            for i, button in enumerate(REMOTE_BUTTON_ORDER):
                if keys & (1 << i):
                    self._press_button(button)

            self.lx = struct.unpack("f", data[4:8])[0]
            self.rx = struct.unpack("f", data[8:12])[0]
            self.ry = struct.unpack("f", data[12:16])[0]
            self.ly = struct.unpack("f", data[20:24])[0]
