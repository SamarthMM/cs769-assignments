import torch
import torch.nn as nn
import zipfile
import numpy as np
import io

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size
        self.vocab_size = len(vocab)

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """

    emb_vocab=dict()
    max_emb_size=0
    emb_f=open(emb_file)
    for line in emb_f:
        line=line.strip().split(' ')
        word=line[0]
        emb_vocab[word]=np.array(line[1:], dtype=np.float64)
        max_emb_size=max(max_emb_size,len(line[1:]))
    if not emb_size:
        emb_size=max_emb_size
    print("max emb size is: ",max_emb_size)
    emb=np.zeros((len(vocab),emb_size)) #emb[word_id]=embedding
    for i in vocab.id2word: #vocab.id2word[integer_id]=word
        word=vocab.id2word[i]
        if word in emb_vocab:
            emb[i]=emb_vocab[word]#[:emb_size]
    return emb       


def init_weights(m):
    v=0.1
    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.uniform_(m.weight,-v,v)
        m.bias.data.fill_(0.01)

class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.embed_dim = 300
        

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.emb_file = args.emb_file
            self.copy_embedding_from_numpy()
        else:
            self.embeddings = torch.nn.Embedding(len(self.vocab.id2word), args.emb_size)

        self.define_model_parameters()
        self.init_model_parameters()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        """
        modules = []
        inp=self.args.emb_size
        outp=self.args.hid_size
        for i in range(self.args.hid_layer):
            modules.append(nn.Linear(inp, outp))
            inp=outp
        modules.append(nn.Linear(outp, self.tag_size))
        self.model = nn.Sequential(*modules)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        """
        self.model.apply(init_weights)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        """
        self.embed_arr = load_embedding(self.vocab, emb_file=self.emb_file, emb_size=self.args.emb_size)
        self.embeddings = torch.nn.Embedding.from_pretrained(torch.from_numpy(self.embed_arr))
        self.embeddings.weight.requires_grad = False

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        '''
        batch_lengths = [0 for i in range(len(x))]
        for i in range(len(x)):
            non_zero = 0
            for num in x[i]:
                if num != 0:
                    non_zero += 1
            batch_lengths[i] = non_zero
        '''
        if (self.args.word_drop!= 0):
            drops=torch.bernoulli(torch.full(x.shape,self.args.word_drop)).bool()
            x=x.masked_fill(drops,0)
        lengths=torch.count_nonzero(x, dim=1).reshape(x.size(0),1)
        
        Z = self.embeddings(x).sum(1)/lengths
        output = self.model(Z.float())
        return output