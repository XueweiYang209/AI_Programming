from operators import Tensor
from autodiff import compute_gradient_of_variables
from utils import ones

class TensorFull(Tensor):
    def __init__(
        self,
        array,
        *,
        device=None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        super().__init__(
            array,
            device=device,
            dtype=dtype,
            requires_grad=True,
            **kwargs
        )

    def backward(self, out_grad=None):
        out_grad = (
                    out_grad
                    if out_grad
                    else ones(*self.shape, dtype=self.dtype, device=self.device)
                )
        compute_gradient_of_variables(self, out_grad)