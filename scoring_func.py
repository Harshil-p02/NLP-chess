import asyncio
import copy

import chess.engine
import chess.pgn
import chess
import re
from tqdm import tqdm

# Example input
src = ['e4', 'c5', 'c3', 'Nc6', 'd4', 'cxd4', 'cxd4', 'e6', 'Nf3', 'Bb4+', 'Nc3', 'd6', 'Bd2', 'Nge7', 'd5']
trg = ['exd5']
res = ['exd5', '<eos>']


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
