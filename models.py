import torch.nn as nn

class MLPModel1(nn.Module):
    def __init__(self):
        super(MLPModel1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        return self.model(x)


class MLPModel2(nn.Module):
    def __init__(self):
        super(MLPModel2, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        return self.model(x)


class MLPModel3(nn.Module):
    def __init__(self):
        super(MLPModel3, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        return self.model(x)


class MLPModel4(nn.Module):
    def __init__(self):
        super(MLPModel4, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )

    def forward(self, x):
        return self.model(x)

class MLPModel5(nn.Module):
    def __init__(self):
        super(MLPModel5, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )

    def forward(self, x):
        return self.model(x)
