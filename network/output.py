import torch.nn as nn

# class for output layer
class Output(nn.Module):
    def __init__(self, input_size: int, output_size: int, act_func="sigmoid", device="cpu"):
        super(Output, self).__init__()
        self.input_size  = input_size
        self.output_size = output_size
        self.device      = device

        if act_func == "sigmoid": self.act_func = nn.Sigmoid()
        elif act_func == "tanh": self.act_func = nn.Tanh()
        else: print("Choose the activating function either from sigmoid or tanh")
        if act_func == "no_act":
            self.layer = nn.Linear(input_size, output_size)
        else:
            self.layer = nn.Sequential(nn.Linear(input_size, output_size), self.act_func)

    def forward(self, d):
        x = self.layer(d)
        return x