from typing import TYPE_CHECKING

from isaaclab.utils import CircularBuffer

if TYPE_CHECKING:
    import torch


class HStepCircularBuffer(CircularBuffer):
    """Circular buffer with a history step argument
    When sampling from the buffer, it will sample every `history_step`-th element in the buffer, starting from the most recent one.
    If history_step == 1, return directly a CircularBuffer
    """

    def __new__(cls, *args, history_step: int = 1, **kwargs):
        # If the step is 1, bypass this subclass and return the base class
        if history_step == 1:
            return CircularBuffer(*args, **kwargs)
        return super().__new__(cls)

    def __init__(self, *args, history_step: int = 1, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        assert self.max_length > self.history_length * history_step, (
            f"HStepCircularBuffer is too small for the given parameters history_length {self.history_length} and history_step {history_step}"
        )
        self.indices = torch.arange(super().buffer.shape[1] - 1, -1, -self.history_step).flip(0)

    @property
    def buffer(self) -> torch.Tensor:
        buf = super().buffer
        return buf[:, self.indices, :]
