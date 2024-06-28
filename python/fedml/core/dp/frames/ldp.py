from collections import OrderedDict
from fedml.core.dp.frames.base_dp_solution import BaseDPFrame
from fedml.core.dp.mechanisms.dp_mechanism import DPMechanism
from torch.nn.utils import clip_grad_norm_

class LocalDP(BaseDPFrame):
    def __init__(self, args):
        super().__init__(args)
        self.set_ldp(DPMechanism(args.mechanism_type, args.epsilon, args.delta, args.sensitivity))
        self.max_grad_norm = args.max_grad_norm  # Ensure max_grad_norm is set

    def add_local_noise(self, local_grad: OrderedDict):
        clipped_grad = OrderedDict()
        for k, grad in local_grad.items():
            clipped_grad[k] = self._clip_grad(grad)
        return super().add_local_noise(local_grad=clipped_grad)

    def _clip_grad(self, grad):
        clipped_grad = grad.clone()
        clip_grad_norm_(clipped_grad, self.max_grad_norm)
        return clipped_grad
