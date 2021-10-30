import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from torchtext.legacy.data import Field, BucketIterator, TabularDataset

import transformer

def get_vocab():
	file = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
	nos = [str(i) for i in range(1, 9)]
	pieces = ['N', 'B', 'R', 'Q', 'K']
	castling = ["O-O", "O-O-O"]
	symbols = ['+', '#', '=', 'x', '[', ']']
	coords = [f+n for f in file for n in nos]

	vocab = file + nos + pieces + castling + symbols + coords

	return vocab


SOS_WORD = "<sos>"
EOS_WORD = "<eos>"
BLANK_WORD = "<blank>"
MAX_LEN = 15*8

SRC = Field(pad_token=BLANK_WORD, fix_length=MAX_LEN, init_token=SOS_WORD, eos_token=EOS_WORD)
TGT = Field(pad_token=BLANK_WORD, fix_length=MAX_LEN, init_token=SOS_WORD, eos_token=EOS_WORD)
fields = (("src", SRC), ("tgt", TGT))

train_data, val_data, test_data = TabularDataset.splits(path=path,
											train='train.csv',validation='validate.csv', test= 'test.csv',
                                            format='csv', fields=fields)

SRC.build_vocab(train_data.src, train_data.tgt, val_data.src, val_data.tgt, test_data.src, test_data.tgt)
TGT.build_vocab(train_data.src, train_data.tgt, val_data.src, val_data.tgt, test_data.src, test_data.tgt)

print(len(SRC.vocab))
print(SRC.vocab.itos[:len(SRC.vocab)])

BATCH_SIZE = 64
device = "cuda:1" if torch.cuda.is_available() else "cpu"
print(device)

train_dataloader, valid_dataloader, test_dataloader = BucketIterator.splits(
	(train_data, val_data, test_data),
	batch_size=BATCH_SIZE,
	device=device,
	sort=False
)

model = transformer.Transformer(num_tokens_inp=len(SRC.vocab), num_tokens_out=len(TGT.vocab), dim_model=200, num_heads=8,
								num_encoder_layers=3, num_decoder_layers=3, max_len=8*15, dropout_p=0.1).to(device)
opt = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


def train_loop(model, opt, loss_fn, dataloader):
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

		total_loss += loss.detach().item()

	return total_loss/len(dataloader)

def validation_loop(model, loss_fn, dataloader):
	model.eval()
	total_loss = 0

	with torch.no_grad():
		for batch in tqdm(dataloader):
			X, y = batch.src.T, batch.tgt.T

			y_input = y[:, :-1]
			y_expected = y[:, 1:]

			sequence_length = y_input.size(1)
			tgt_mask = model.get_tgt_mask(sequence_length).to(device)

			pred = model(X, y_input, tgt_mask)
			pred = pred.permute(1, 2, 0)
			loss = loss_fn(pred, y_expected)

			total_loss += loss.detach().item()

	return total_loss/len(dataloader)

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
	train_loss_list, validation_loss_list = [], []
	best_valid_loss = float('inf')

	print("Training and validating model")
	for epoch in range(epochs):
		print("-" * 25, f"Epoch {epoch + 1}", "-" * 25)

		train_loss = train_loop(model, opt, loss_fn, train_dataloader)
		train_loss_list += [train_loss]

		validation_loss = validation_loop(model, loss_fn, val_dataloader)
		validation_loss_list += [validation_loss]

		if validation_loss < best_valid_loss:
			torch.save(model.state_dict(), 'transformer-best.pt')
			best_valid_loss = validation_loss

		print(f"Training loss: {train_loss:.4f}")
		print(f"Validation loss: {validation_loss:.4f}")
		print()

	return train_loss_list, validation_loss_list

train_loss_list, validation_loss_list = fit(model, opt, loss_fn, train_dataloader, valid_dataloader, 10)


def predict(model, input_sequence, max_length=8*15, SOS_token=2, EOS_token=3):
	model.eval()

	y_input = torch.tensor([[SOS_token]], dtype=torch.long, device=device)

	for _ in range(max_length):
		# Get source mask
		tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

		pred = model(input_sequence, y_input, tgt_mask)

		next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
		next_item = torch.tensor([[next_item]], device=device)

		# Concatenate previous input with predicted best word
		y_input = torch.cat((y_input, next_item), dim=1)

		# Stop if model predicts end of sentence
		if next_item.view(-1).item() == EOS_token:
			break

	return y_input.view(-1).tolist()

# for i in test_dataloader:
# 	example = i
# 	X = example.src
# 	y = example.tgt
#
# 	pred = predict(model,X[:,0].reshape(1,-1))
# 	print("pred", pred)
#
# 	inp = X[:, 0].tolist()
# 	out = y[:, 0].tolist()
#
# 	print("input")
# 	for i in inp:
# 		print(SRC.vocab.itos[i], end='')
# 	print()
# 	print()
# 	print("output")
# 	for i in out:
# 		print(SRC.vocab.itos[i], end='')
# 	print()
# 	print()
# 	print("pred")
# 	for i in pred:
# 		print(SRC.vocab.itos[i], end='')
# 	print()
# 	print()