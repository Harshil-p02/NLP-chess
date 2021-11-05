import asyncio
import copy

import chess.engine
import chess.pgn
import chess
import re
from tqdm import tqdm

# e4 c5 2. c3 Nc6 3. d4 cxd4 4. cxd4 e6 5. Nf3 Bb4+ 6. Nc3 d6 7. Bd2 Nge7 8. d5 exd5
'''
# example data:
src = ['e4', 'c5', 'c3', 'Nc6', 'd4', 'cxd4', 'cxd4', 'e6', 'Nf3', 'Bb4+', 'Nc3', 'd6', 'Bd2', 'Nge7', 'd5']
trg = ['Rad8']
res = ['exd5', '<eos>']
'''

def stockfish14_scores(move_list, prediction):

    score = []
    board_in = chess.Board()
    for move in move_list:
      # illegal move in inital game seq
      try:
        board_in.push_san(move)
      except ValueError:
        return score
    # use the input board with initial game state

    engine = chess.engine.SimpleEngine.popen_uci("/content/drive/MyDrive/NLP-Chess_lang-model/stockfish_13_linux_x64_avx2")
    # Scoring ----> Mate(-0) < Mate(-1) < Cp(-50) < Cp(200) < Mate(4) < Mate(1) < MateGiven

    # using copy to have two diff boards for evaluation
    eval_board = copy.deepcopy(board_in)

    info = engine.analyse(eval_board, chess.engine.Limit(time=0.1))
    score.append(info["score"].white().wdl().expectation())
    # win-draw-lose percentage from white's perspective 
    try:
        board_in.push_san(prediction[0])
    except ValueError:
        # print("Illegal move")
        score.append(float('-inf'))
    else:
        info = engine.analyse(board_in, chess.engine.Limit(time=0.1))
        score.append(info['score'].white().wdl().expectation())

    result = engine.play(eval_board, chess.engine.Limit(time=0.1))
    eval_board.push(result.move)
    info = engine.analyse(eval_board, chess.engine.Limit(time=0.1))
    score.append(info['score'].white().wdl().expectation())

    engine.quit()
    return score

# pred = ['err']
# score = stockfish14_scores(src, res)
# +ve score --------> stockfish better
# if score == inf  ------> illelagl move predicted
