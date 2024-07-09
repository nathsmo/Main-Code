import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

# Original implementation of the attention mechanism is through pointer networks
class DecodeStep(nn.Module):
    """
    Base class for decoding (without RNN).
    """
    def __init__(self, attention_instance, hidden_dim, n_glimpses=0, mask_glimpses=True, mask_pointer=True, rnn_layers=1):
        super(DecodeStep, self).__init__()
        self.glimpses = nn.ModuleList([attention_instance for _ in range(n_glimpses)])
        self.pointer = attention_instance
        self.mask_glimpses = mask_glimpses
        self.mask_pointer = mask_pointer
        self.BIGNUMBER = 100000

        # Multi-layer LSTM
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, num_layers=rnn_layers, batch_first=True)

    def forward(self, decoder_inp, context, mask, hidden=None):
        if decoder_inp.dim() == 2:
            decoder_inp = decoder_inp.unsqueeze(1)
        # Verify hidden state dimensions if provided
        if hidden is not None:
            assert hidden[0].dim() == 3 and hidden[1].dim() == 3, "Hidden states should be 3-dimensional"
        output, hidden = self.rnn(decoder_inp, hidden)
        # Use the last hidden state from RNN output
        last_hidden = output[:, -1, :]  # Taking the last output as the next input to attention
        # Process each glimpse
        for glimpse in self.glimpses:
            _, logit = glimpse(last_hidden, context)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER * mask
            prob = F.softmax(logit, dim=1)
            # last_hidden = torch.bmm(prob.unsqueeze(1)).squeeze(1)
            last_hidden = torch.sum(prob.unsqueeze(2) * context, dim=1)

        # Process pointer attention
        _, logit = self.pointer(last_hidden, context)

        if self.mask_pointer:
            logit -= self.BIGNUMBER * mask

        return logit, hidden

