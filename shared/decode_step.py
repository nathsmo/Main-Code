import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class DecodeStep(nn.Module):
    """
    Base class for decoding (without RNN).
    """
    def __init__(self, ClAttention, hidden_dim, use_tanh=False, tanh_exploration=10.,
                 n_glimpses=0, mask_glimpses=True, mask_pointer=True):
        super(DecodeStep, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_pointer = mask_pointer
        self.BIGNUMBER = 100000.0

        # Initialize glimpses and pointer attention mechanisms
        self.glimpses = nn.ModuleList([
            ClAttention(hidden_dim, use_tanh=False, C=None)
            for i in range(n_glimpses)
        ])
        self.pointer = ClAttention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)

    def forward(self, decoder_inp, context, Env, decoder_state=None):
        # I think here's the problem
        # Process glimpses
        print('Decoder input: ', decoder_inp)
        print('Context: ', context)
        mask = Env.mask
        for glimpse in self.glimpses:
            ref, logit = glimpse(decoder_inp, context, Env)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER * mask
            prob = F.softmax(logit, dim=-1)
            decoder_inp = torch.bmm(prob.unsqueeze(1), ref).squeeze(1)

        # Process pointer attention
        # print('Inputs provided: ', decoder_inp, context, Env)
        
        _, logit = self.pointer(decoder_inp, context, Env)

        if self.mask_pointer:
            logit -= self.BIGNUMBER * mask
        
        log_prob = F.log_softmax(logit, dim=-1)
        prob = F.exp(log_prob)

        return logit, decoder_state

class RNNDecodeStep(DecodeStep):
    """
    Decodes the sequence. It keeps the decoding history in an RNN.
    """
    def __init__(self, ClAttention, hidden_dim, use_tanh=False, tanh_exploration=10.,
                 n_glimpses=0, mask_glimpses=True, mask_pointer=True, forget_bias=1.0,
                 rnn_layers=1):
        super(RNNDecodeStep, self).__init__(ClAttention, hidden_dim, use_tanh, tanh_exploration,
                                            n_glimpses, mask_glimpses, mask_pointer)
        self.forget_bias = forget_bias
        self.rnn_layers = rnn_layers
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=rnn_layers,
                            batch_first=True, dropout=0 if rnn_layers == 1 else 0.5)

    def forward(self, decoder_inp, context, Env, decoder_state=None):
        mask = Env.mask
        # print('mask type: ', type(mask))

        if decoder_state is None:
            decoder_state = self._init_hidden(decoder_inp.size(0))

        # lstm_out: [batch_size, 1, hidden_dim]
        lstm_out, decoder_state = self.lstm(decoder_inp, decoder_state)
        # print('lstm_out shape: ', lstm_out.shape)

        # decoder_inp: [batch_size, hidden_dim]
        decoder_inp = lstm_out.squeeze(1)

        # Process glimpses
        # print('Glimpses: ', self.n_glimpses)
        for i in range(self.n_glimpses):
            ref, logit = self.glimpses[i](decoder_inp, context, Env)
            if self.mask_glimpses:
                logit -= self.BIGNUMBER * mask
            prob = F.softmax(logit, dim=-1)
            decoder_inp = torch.bmm(prob.unsqueeze(1), ref).squeeze(1)

        # Process pointer attention
        _, logit = self.pointer(decoder_inp, context, Env)
        if self.mask_pointer:
            logit -= self.BIGNUMBER * mask

        return logit, decoder_state
    
    def _init_hidden(self, batch_size):
        return (torch.zeros(self.rnn_layers, batch_size, self.hidden_dim),
                torch.zeros(self.rnn_layers, batch_size, self.hidden_dim))

# Hasn't been tested yet.