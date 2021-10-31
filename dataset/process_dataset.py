import os
import re
from tqdm import tqdm
import time
import pickle
from langdetect import detect

from sklearn.model_selection import train_test_split


# PGN Standard: http://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm#c8.2

def clean_kaggle(read_path, write_path):

	# if starting no is not 1, remove game

	# pgn data from kaggle has different notation
	# only retain move info
	print("reading games...")
	with open(read_path) as f:
		games = f.readlines()

	print("cleaning games...")
	moves = []
	for game in tqdm(games):
		if game[1] == '1':
			move = re.split(" |\.", game)
			moves.append(" ".join(move[1::2]))

	print("cleaned games")

	with open(write_path, "w") as f:
		f.writelines(moves)

	print("finished writing")

def make_dataset(read_path, write_path, ratio=(0.9, 0.05, 0.05), seed=1234):
	"""
	make test, train and validate datasets
	:param read_path: file to read
	:param write_path: folder to write output
	:param ratio: splitting ration
	:param seed:
	"""

	with open(read_path) as f:
		games = f.readlines()

	r_train, r_test, r_validate = ratio

	train, test_validate = train_test_split(games, train_size=r_train, test_size=r_test+r_validate, random_state=seed)
	test, validate = train_test_split(test_validate, train_size=r_test*10, test_size=r_validate*10, random_state=seed)

	with open(write_path + "/" + "train.txt", "w") as f:
		f.writelines(train)

	with open(write_path + "/" + "test.txt", "w") as f:
		f.writelines(test)

	with open(write_path + "/" + "validate.txt", "w") as f:
		f.writelines(validate)

def get_tokens(path):
	"""
	split the moves into tokens separating the chess piece from the move coordinate
	Ex:		fxg1=Q+ -> '[ f x g1 = Q + ]'

	:param path: input file
	:return tokens: tokenized input
	"""

	with open(path) as f:
		games = f.readlines()

	tokens = []

	for game in tqdm(games):
		moves = game.split()

		token = []
		for i in range(len(moves)):
			chars = list(moves[i])

			move = "[ "
			if len(chars) == 2:
				move += ''.join(chars) + " "
			elif chars[0] == "O":
				if len(chars) == 3 or len(chars) == 5:
					move += ''.join(chars) + " "
				else:
					if len(chars) == 4:
						move += "O-O " + chars[3] + " "
					else:
						move += "O-O-O " + chars[5] + " "
			else:
				j = 0
				while j < len(chars):
					if j+1 < len(chars) and chars[j+1].isnumeric():
						move += ''.join(chars[j:j+2]) + " "
						j += 1
					else:
						move += chars[j] + " "
					j += 1

			move += "]"
			token.append(move)
		tokens.append(token)

	return tokens

def make_datasets_nmoves(read_path, write_path, start, end):
	"""
	Create csv files to be used for training the model
	:param read_path: folder to read train, test and validate
	:param write_path: folder to write output
	:param start: index for starting move (0 indexed)
	:param end: index for ending+1 th move
	"""

	t1 = time.time()
	print("getting train tokens...")
	train = get_tokens(rf"{read_path}/train.txt")

	print("getting test tokens...")
	test = get_tokens(rf"{read_path}/test.txt")

	print("getting validate tokens...")
	validate = get_tokens(rf"{read_path}/validate.txt")

	train_src = []
	train_tgt = []
	print(f"grabbing {start}-{end} moves from train tokens...")
	for game in tqdm(train):
		if len(game) >= 2 * end:
			train_src.append(' '.join(game[start:2 * end - 1]))
			train_tgt.append(game[2 * end - 1])

	# n_moves = end - start
	print(f"writing training data for {start}-{end} to file...")
	os.makedirs(os.path.dirname(rf"{write_path}/{start}-{end}/train.csv"), exist_ok=True)
	with open(rf"{write_path}/{start}-{end}/train.csv", "w") as f:
		for i in range(len(train_src)):
			f.write(f"{train_src[i]},{train_tgt[i]}\n")

	test_src = []
	test_tgt = []
	print(f"grabbing {start}-{end} from test tokens...")
	for game in tqdm(test):
		if len(game) >= 2 * end:
			test_src.append(' '.join(game[start:2 * end - 1]))
			test_tgt.append(game[2 * end - 1])

	print(f"writing testing data for {start}-{end} to file...")
	os.makedirs(os.path.dirname(rf"{write_path}/{start}-{end}/test.csv"), exist_ok=True)
	with open(rf"{write_path}/{start}-{end}/test.csv", "w") as f:
		for i in range(len(test_src)):
			f.write(f"{test_src[i]},{test_tgt[i]}\n")

	validate_src = []
	validate_tgt = []
	print(f"grabbing {start}-{end} from validate tokens...")
	for game in tqdm(validate):
		if len(game) >= 2 * end:
			validate_src.append(' '.join(game[start:2 * end - 1]))
			validate_tgt.append(game[2 * end - 1])

	print(f"writing validation data for {start}-{end} to file...")
	os.makedirs(os.path.dirname(rf"{write_path}/{start}-{end}/validate.csv"), exist_ok=True)
	with open(rf"{write_path}/{start}-{end}/validate.csv", "w") as f:
		for i in range(len(validate_src)):
			f.write(f"{validate_src[i]},{validate_tgt[i]}\n")

	print(f"Finished in {time.time()-t1}")

def make_commentary_dataset(input_file, output_folder):
	data = []
	not_en = 0

	print("reading file...")
	with open(f"{input_file}", "rb") as f:
		commentary = pickle.load(f)
		print(len(commentary))

		print("cleaning data...")

		for i in tqdm(range(len(commentary))):
			src = commentary[i][0]
			tgt = commentary[i][1]
			src = src.replace(u'\xa0', u' ')
			tgt = tgt.replace(u'\xa0', u' ')

			try:
				if detect(tgt) != "en":
					not_en += 1
				else:
					src = re.split("\s|\.", src)
					tgt = re.split("\s|,", tgt)

					src = [x for x in src if x and not x.isnumeric()]
					tgt = [x for x in tgt if x]

					data.append((src, tgt))
			except:
				src = re.split("\s|\.", src)
				tgt = re.split("\s|,", tgt)

				src = [x for x in src if x and not x.isnumeric()]
				tgt = [x for x in tgt if x]

				data.append((src, tgt))

	with open(f"{output_folder}/commentary.csv", "w", encoding="utf-8") as f:
		for i in data:
			f.write(f"{' '.join(i[0])},{' '.join(i[1])}\n")

	print(f"Other language games: {not_en}")
