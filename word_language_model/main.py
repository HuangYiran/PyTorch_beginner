import argparse
import time
import math
import torch
from torch.autograd import Variable

import data_util as du
import model

parser = argparse.ArgumentParser(description = 'PyTorch PTB rnn/lstm model')
parser.add_argument('--data', type = str, default = '../data/PTB',
        help = 'location of the data corpus')
parser.add_argument('--model', type = str, default = 'LSTM',
        help = 'type of the rnn (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type = int, default = 200,
        help = 'size of the embedding')
parser.add_argument('--nhid', type = int, default = 200,
        help = 'number of hidden units per layer')
parser.add_argument('--nlayers', type = int, default = 2,
        help = 'number of layers')
parser.add_argument('--lr', type = float, default = 20,
        help = 'initial learning rate')
parser.add_argument('--clip', type = float, default = 0.25,
        help = 'gradient clipping???')
parser.add_argument('--epochs', type = int, default = 40,
        help = 'upper epoch limit')
parser.add_argument('--batch_size', type = int, default = 20, metavar = 'N', #??
        help = 'batch size')
parser.add_argument('--bptt', type = int, default =35,
        help = 'sequence length ')
parser.add_argument('--dropout', type = float, default = 0.2,
        help = 'dropout applied to layers(0 = no dropout)')
parser.add_argument('--tied', action = 'store_true',
        help = 'tie the word embedding and softmax weights')
parser.add_argument('--seed', type = int, default = 1111,
        help = 'random seed')
#parser.add_argument('--cuda', action = 'store_true')
parser.add_argument('--log-interval', type = int, default = 200, metavar = 'N',
        help = 'report interbal')
parser.add_argumennt('--save', type = str, default = 'model.pt',
        help = 'path to save the final model')
args = parser.parser_args()

# set the random seed manually for reproducibility
torch.manual_seed(args.seed)

# load data
corpus = du.Corpus(args.data)

def batchify(data, bsz): # 这部分可以放到util中
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch*bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data

eval_batch_size = 10
train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

# build the model
ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
criterion = torch.nn.CrossEntropyLoss() #为什么不在model中？

# train code
def repackage_hidden(h): # 目的不明
    """Wraps hidden states in new Variables, to detach them from their history???"""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation = False): # 为什么不在model中？
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile = evaluation)
    target = Variable(source[i+1: i+1+seq_len].view(-1))
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disable dropout
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size) # 因为eval使用不同的batch size
    for i in range(0, data_source.size(0) - 1):
        data, targets = get_batch(data_source, i, evaluation = True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data)*criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0]/len(data_source)

def train():
    # Turn on training mode which enables dropout and bn
    model.turn()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the ay to start of the dataset!!!
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs/ LSTMs
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)
        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('|epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) //args.bptt,
                        lr, elapsed * 1000/ args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Loop over epochs
lr = args.lr
best_val_loss = None

# at any point you can his Ctrl + C to break out of training early
try:
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(val_data)
        print('-'*89)
        print('|end of epoch {:3d} | time {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time()-epoch_start_time), val_loss, math.exp(val_loss)))
        print('-'*89)
        # save the model if the validation loss is the best we've seen so far
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # Anneal the learning rate if no improvement has been seen in the validation dataset
            lr /= 4.0
except KeyboardInterrput:
    print('-'*89)
    print('Exiting from training early')

# Run on test data
test_loss = evaluate(test_data)
print('='*89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, match.exp(test_loss)))
print('='*89)
