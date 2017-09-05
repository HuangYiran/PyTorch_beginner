import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):
    def __init__(self,
            rnn_type,
            ntoken,
            ninp, # number of input features
            nhid, # number of hidden features
            nlayers,
            dropout = 0.5,
            tie_weights = False):
        self.ntoken = ntoken
        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(self.dropout)
        self.encoder = nn.Embedding(self.ntoken, self.ninp)
        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.rnn_type)(self.ninp, self.nhid, self.nlayers, dropout = self.dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.rnn_type]
            except:
                raise ValueError("""An invalid option for `--model` was supplied, options are ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU']""")
            self.rnn = nn.RNN(self.ninp, self.nhid, self.nlayers, nonlinearity = nonlinearity, dropout = self.dropout)
        self.decoder = nn.Linear(self.nhid, self.ntoken)
        if tie_weights:
            if self.nhid != self.ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
        self.init_weights() # 必须的吗？

    def init_weights(self):
        # 其他的呢？？哪些需要哪些不需要？？
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        # 不知道是干嘛用的
        weight = next(self.parameters()).data # ??
        if self.rnn_type == 'LSTM':
            return(Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
