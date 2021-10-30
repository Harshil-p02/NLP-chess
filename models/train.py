import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchtext.legacy import data
from torchtext.legacy.data import Field, BucketIterator

from models import transformer

# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

def get_vocab():
	file = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
	nos = [str(i) for i in range(1, 9)]
	pieces = ['N', 'B', 'R', 'Q', 'K']
	castling = ["O-O", "O-O-O"]
	symbols = ['+', '#', '=', 'x', '[', ']']
	coords = [f+n for f in file for n in nos]

	vocab = file + nos + pieces + castling + symbols + coords

	return vocab


SOS_WORD = "["
EOS_WORD = "]"

SRC = Field(init_token=SOS_WORD, eos_token=EOS_WORD)
TGT = Field(init_token=SOS_WORD, eos_token=EOS_WORD)

train_data, val_data, test_data = data.TabularDataset.splits(path=path,
															 train="train.csv", validation="validate.csv",
															 test="test.csv", format="csv", fields=[("src", SRC),
																									("tgt", TGT)])
SRC.build_vocab(train_data, val_data)
TGT.build_vocab(train_data, val_data)
print(len(SRC.vocab))
print(SRC.vocab.itos[:len(SRC.vocab)])
# SRC.vocab = get_vocab()
# TGT.vocab = get_vocab()

BATCH_SIZE = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataloader, valid_dataloader, test_dataloader = BucketIterator.splits(
	(train_data, val_data, test_data),
	batch_size=BATCH_SIZE,
	device=device,
	sort=False
)

model = transformer.Transformer(num_tokens_inp=len(SRC.vocab), num_tokens_out=len(TGT.vocab), dim_model=200, num_heads=8,
								num_encoder_layers=6, num_decoder_layers=6, max_len=62, dropout_p=0.1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def train(model, opt, loss_fn, dataloader):
	model.train()
	total_loss = 0

	for batch in tqdm(dataloader):
		X = batch.src.T
		y = batch.tgt.T

		y_input = y[:, :-1]
		y_expected = y[:, 1:]

		sequence_length = y_input.size(1)
		tgt_mask = model.get_tgt_mask(sequence_length).to(device)

		pred = model(X, y_input, tgt_mask)

		pred = pred.permute(1, 2, 0)
		loss = loss_fn(pred, y_expected)

		opt.zero_grad()
		loss.backward()
		opt.step()

		# print(output)

		total_loss += loss.detach().item()

	return total_loss/len(dataloader)

train_loss = train(model, optimizer, loss_fn, train_dataloader)
print(train_loss)

# for i in train_dataloader:
# 	print(i)

# print(torch.cuda.is_available())