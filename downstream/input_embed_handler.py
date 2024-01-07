import torch
import torch.nn.functional as F


class InputEmbedHandler:
    def __init__(self, code_size, embed_scale=1.0, normalized=True):
        self.code_size = 8
        learning_weight_init = torch.arange(code_size, 0, step=-1).float().view(1, 1, code_size, 1)
        self.weighted_sum = torch.nn.Parameter(learning_weight_init)
        self.gradient_checkpointing = False

        self.embed_scale = embed_scale
        self.normalized = normalized
        self.layer_norm = torch.nn.LayerNorm(768)

    def __call__(self, input_ids, embed_fun):
        code_size = self.code_size
        batch_dim = input_ids.shape[0]
        code_dim = input_ids.shape[1] // code_size
        input_ids = input_ids.view(batch_dim, code_dim, code_size)

        stacked_inputs = []
        for i in range(code_dim):
            embedded_input = embed_fun(input_ids[:, i, :]) * self.embed_scale
            if self.normalized:
                embedded_input = self.layer_norm(embedded_input)
            stacked_inputs.append(embedded_input)
        stacked_inputs = torch.stack(stacked_inputs, dim=0)
        weighted_input_embed = torch.mul(stacked_inputs,
                                         F.softmax(self.weighted_sum, dim=0))

        weighted_input_embed = torch.sum(weighted_input_embed, dim=2)
        # should handle position embedding based on different kind of models
        return weighted_input_embed.view(batch_dim, -1, 768)
