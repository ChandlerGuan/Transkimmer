from torch import nn
import torch

def init_skim_predictor(module_list, mean_bias=5.0):
    for module in module_list:
        if not isinstance(module, torch.nn.Linear):
            raise ValueError("only support initialization of linear skim predictor")

        # module.bias.data[1].fill_(5.0)
        # module.bias.data[0].fill_(-5.0)
        # module.weight.data.zero_()
        module.bias.data[1].normal_(mean=mean_bias, std=0.02)
        module.bias.data[0].normal_(mean=-mean_bias, std=0.02)
        module.weight.data.normal_(mean=0.0, std=0.02)

        module._skim_initialized = True

class SkimPredictor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=None):
        super().__init__()
        
        self.hidden_size = hidden_size if hidden_size else input_size

        self.predictor = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, self.hidden_size),
            # nn.GELU(),
            # nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_size),
        )

        init_skim_predictor([self.predictor[-1]])

    def forward(self, hidden_states):
        return self.predictor(hidden_states)

def test_init_skim_predictor():
    num_layers = 12

    skim_predictors = torch.nn.ModuleList([torch.nn.Linear(768,2) for _ in range(num_layers)])
    init_skim_predictor(skim_predictors)

    print(skim_predictors[0].weight, skim_predictors[0].bias)

    rand_input = torch.rand((4, 16, 768))
    print(skim_predictors[0](rand_input))

if __name__ == "__main__":
    test_init_skim_predictor()