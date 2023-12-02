from torch import Tensor, nn


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = None
        self.target = None

    def forward(self, output: Tensor, target: Tensor, model_type: str, num_classes: int) -> Tensor:
        self.output = output
        self.target = target
        if model_type[:3] == "PSP":
            print(output[0].size(), target.size())
            print(num_classes)
            assert output[0].size()[2:] == target.size()[1:]
            assert output[0].size()[1] == num_classes
            loss = self._forward(output[0], target)
            loss += self._forward(output[1], target) * 0.4
        else:
            assert output.size()[2:] == target.size()[1:]
            assert output.size()[1] == self.num_classes
            loss = self._forward(output, target)
        return loss

    def _forward(self, output: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError
