
import torch
torch.cuda.is_available()

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.legacy.data import Field, BucketIterator

import numpy as np

import random
import math
import time

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



from torchtext.legacy import data

SOS_WORD = "<sos>"
EOS_WORD = "<eos>"

SRC = Field(init_token=SOS_WORD, eos_token=EOS_WORD)
TRG = Field(init_token=SOS_WORD, eos_token=EOS_WORD)

train_data, val_data, test_data = data.TabularDataset.splits(path='/home/prateek.kj/commentary/',
															 train="tokened-test-half.csv", validation="tokened-validate-half.csv",
															 test="tokened-test.csv", format="csv", fields=[("src", SRC),
																									("trg", TRG)])
# SRC.build_vocab(train_data.src)
# TRG.build_vocab(train_data.trg)
SRC.build_vocab(train_data.src, val_data.src, test_data.src)
TRG.build_vocab(train_data.src, train_data.trg, val_data.src, val_data.trg, test_data.src, test_data.trg)

print(SRC.vocab.itos[00:30])
print(TRG.vocab.itos[10:30])

print(f"Unique tokens in source vocabulary: {len(SRC.vocab)}")
print(f"Unique tokens in target vocabulary: {len(TRG.vocab)}")

batch_size = 8
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
# Group similar length text sequences together in batches.
train_dataloader, valid_dataloader,test_dataloader = data.BucketIterator.splits(

                              # Datasets for iterator to draw data from
                              (train_data, val_data, test_data),

                              batch_size = batch_size,
                              sort = False,
                              device = device
                              )

# Print number of batches in each split.
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
print('Created `valid_dataloader` with %d batches!'%len(valid_dataloader))
print('Created `test_dataloader` with %d batches!'%len(test_dataloader))
print(f'batchsize = {batch_size}')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell

class Move_Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Commentary_Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, move_decoder, commentary_decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.move_decoder = move_decoder
        self.commentary_decoder = commentary_decoder
        self.device = device
        
        assert encoder.hid_dim == move_decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.hid_dim == commentary_decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == move_decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        assert encoder.n_layers == commentary_decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.move_decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        t = 1
        #insert input token embedding, previous hidden and previous cell states
        #receive output tensor (predictions) and new hidden and cell states
        output, hidden, cell = self.move_decoder(input, hidden, cell)
         
        #place predictions in a tensor holding predictions for each token
        outputs[t] = output
         
        #decide if we are going to use teacher forcing or not
        teacher_force = random.random() < teacher_forcing_ratio
         
        #get the highest predicted token from our predictions
        top1 = output.argmax(1) 
         
        #if teacher forcing, use actual next token as next input
        #if not, use predicted token
        input = trg[t] if teacher_force else top1
        t += 1

        # till move ending token
        while t < trg_len and trg[t-1] != "]":
          #insert input token embedding, previous hidden and previous cell states
          #receive output tensor (predictions) and new hidden and cell states
          output, hidden, cell = self.move_decoder(input, hidden, cell)
          
          #place predictions in a tensor holding predictions for each token
          outputs[t] = output
          
          #decide if we are going to use teacher forcing or not
          teacher_force = random.random() < teacher_forcing_ratio
          
          #get the highest predicted token from our predictions
          top1 = output.argmax(1) 
          
          #if teacher forcing, use actual next token as next input
          #if not, use predicted token
          input = trg[t] if teacher_force else top1
          t += 1

        # now generate commentary
        while t < trg_len:
          output, hidden, cell = self.commentary_decoder(input, hidden, cell)
          outputs[t] = output
          teacher_force = random.random() < teacher_forcing_ratio
          top1 = output.argmax(1) 
          input = trg[t] if teacher_force else top1
          t += 1

        return outputs

INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENC_EMB_DIM = 128
DEC_EMB_DIM = 128
HID_DIM = 128
N_LAYERS = 1
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec1 = Move_Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
dec2 = Commentary_Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)
# device = 'cpu'
model = Seq2Seq(enc, dec1, dec2, device).to(device)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
        
model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters(),lr=0.0001)

TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output = model(src, trg)
        
        #trg = [trg len, batch size]
        #output = [trg len, batch size, output dim]
        
        output_dim = output.shape[-1]
        
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]
        
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]

            output_dim = output.shape[-1]
            
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            
            #trg = [(trg len - 1) * batch size]
            #output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

N_EPOCHS = 30    
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_dataloader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_dataloader, criterion)
    
    end_time = time.time()
    
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'commentary_move.pt')
    if epoch > N_EPOCHS-2:
      torch.save(model.state_dict(), 'commentary_move_last.pt')    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

model.load_state_dict(torch.load('commentary_move.pt'))

test_loss = evaluate(model, valid_dataloader, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):

    model.eval()
        
    tokens = sentence
        
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    src_len = torch.LongTensor([len(src_indexes)])
    
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

        
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
    
    for i in range(max_len):
    
        with torch.no_grad():
            output, hidden, _ = model.move_decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        # stopping prediction at ']' ---> move ending
        if pred_token == trg_field.vocab.stoi[']']:
            break

    for i in range(max_len):
                
        with torch.no_grad():
            output, hidden, _ = model.commentary_decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        
        trg_indexes.append(pred_token)

        # stopping prediction at ']' ---> move ending
        if pred_token == trg_field.vocab.stoi['<eos>']:
            break
      
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:]


# Predictions

for example in range(10):  

  example_idx = random.randint(0,5000)

  input_ = test_data.examples[example_idx].src
  target = test_data.examples[example_idx].trg

  print(f'src = {input_}')
  print(f'trg = {target}\n')
  
  pred_out = (translate_sentence(input_, SRC, TRG, model, device))
  print(pred_out)
  print("\n\n")
 
