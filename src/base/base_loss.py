from torch import Tensor, nn


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = None
        self.target = None

    def forward(self, output: Tensor | tuple, target: Tensor | tuple, num_classes: int) -> Tensor:
        self.output = output
        self.target = target
        if type(output) == tuple:
            assert output[0].size()[1:] == target.size()[1:]
            # assert output[0].size()[1] == num_classes
            loss = self._forward(output[0], target)
            loss += self._forward(output[1], target) * 0.4
        else:
            assert output.size()[1:] == target.size()[1:]
            # assert output.size()[1] == num_classes
            loss = self._forward(output, target)
        return loss

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError
