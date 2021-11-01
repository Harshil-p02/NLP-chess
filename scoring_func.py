import asyncio
import copy

import chess.engine
import chess.pgn
import chess
import re
from tqdm import tqdm

# e4 c5 2. c3 Nc6 3. d4 cxd4 4. cxd4 e6 5. Nf3 Bb4+ 6. Nc3 d6 7. Bd2 Nge7 8. d5 exd5
src = ['e4', 'c5', 'c3', 'Nc6', 'd4', 'cxd4', 'cxd4', 'e6', 'Nf3', 'Bb4+', 'Nc3', 'd6', 'Bd2', 'Nge7', 'd5']
trg = ['Rad8']
res = ['exd5', '<eos>']


moves = ''
i=0
c = 1
while i < (len(src))-2:
    moves += str(c) + '.'
    for j in range(2):
        moves += " " + src[i+j]
    moves += " "
    c += 1
    i += 2
moves += str(c) + ". " + src[-1] + " "
print(moves)

pgn = open("test.pgn")
game = chess.pgn.read_game(pgn)
board1 = chess.Board()

# src has to be a list of whole moves.
# example: src = ['e4', 'c5', 'c3', 'Nc6', 'd4', 'cxd4', 'cxd4', 'e6', 'Nf3', 'Bb4+', 'Nc3', 'd6', 'Bd2', 'Nge7', 'd5']
for move in src:
    board1.push_san(move)
    # print(board1, end="\n----------------------------------\n")


def stockfish14_scores(move_list, prediction):

    board_in = chess.Board()
    # move_list has to be a list of whole moves.
    # example: move_list = ['e4', 'c5', 'c3', 'Nc6', 'd4', 'cxd4', 'cxd4', 'e6', 'Nf3', 'Bb4+', 'Nc3', 'd6', 'Bd2', 'Nge7', 'd5']
    for move in move_list:
        board_in.push_san(move)
    # use the input board with initial game state
    # board = board_in

    async def main() -> None:
        transport, engine = await chess.engine.popen_uci("stockfish_14_x64_avx2.exe")
        # Scoring ----> Mate(-0) < Mate(-1) < Cp(-50) < Cp(200) < Mate(4) < Mate(1) < MateGiven

        # using copy to have two diff boards for evaluation
        eval_board = copy.deepcopy(board_in)

        info = await engine.analyse(eval_board, chess.engine.Limit(time=0.1, depth=18))
        print("Initial:", info["score"])

        try:
            board_in.push_san(prediction[0])
        except ValueError:
            print("Illegal move")
        else:
            info = await engine.analyse(board_in, chess.engine.Limit(time=0.1, depth=18))
            print('Prediction:', info["score"])

        result = await engine.play(eval_board, chess.engine.Limit(time=0.1, depth=18))
        eval_board.push(result.move)
        info = await engine.analyse(eval_board, chess.engine.Limit(time=0.1, depth=18))
        print('Stockfish:', info["score"])

        await engine.quit()

    asyncio.set_event_loop_policy(chess.engine.EventLoopPolicy())
    asyncio.run(main())


stockfish14_scores(src, res)