import torch
from torch import Tensor, nn


class ContextEncoder(nn.Module):
    """GRU encoder to encode observation data into context for the posterior network.

    Uses chunked processing to handle long sequences that exceed cuDNN's ~65k limit.
    """

    # cuDNN GRU has a max sequence length of ~65k
    CUDNN_SEQ_LIMIT = 60000

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(ContextEncoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp: Tensor) -> Tensor:
        """Encode a sequence into a context series."""
        seq_len = inp.size(0)

        # For short sequences, process directly
        if seq_len <= self.CUDNN_SEQ_LIMIT:
            self.gru.flatten_parameters()
            out, _ = self.gru(inp)
        else:
            # Process long sequences in chunks to avoid cuDNN limit
            outputs = []
            hidden = None
            for start in range(0, seq_len, self.CUDNN_SEQ_LIMIT):
                end = min(start + self.CUDNN_SEQ_LIMIT, seq_len)
                chunk = inp[start:end]
                self.gru.flatten_parameters()
                out_chunk, hidden = self.gru(chunk, hidden)
                outputs.append(out_chunk)
            out = torch.cat(outputs, dim=0)

        return self.lin(out)


class LinearScheduler(object):
    """Linear scheduler that ramps a value from 0 to maxval."""

    def __init__(self, iters: int, maxval: float = 1.0) -> None:
        self._iters = max(1, iters)
        self._val = 0
        self._maxval = maxval

    def step(self) -> None:
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self) -> float:
        return self._val


class LipSwish(nn.Module):
    """Lipschitz-constrained Swish activation."""

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x) / 1.1
