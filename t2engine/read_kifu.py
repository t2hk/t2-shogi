import shogi
import shogi.CSA
import copy
import sys

from t2engine.features import *

# read kifu
def read_kifu(kifu_list_file):
    positions = []
    with open(kifu_list_file, 'r') as f:
        for line in f.readlines():
            try:
                filepath = line.rstrip('\r\n')

                #print("kifu file : " , filepath)

                kifu = shogi.CSA.Parser.parse_file(filepath)[0]
                win_color = shogi.BLACK if kifu['win'] == 'b' else shogi.WHITE
                board = shogi.Board()
                for move in kifu['moves']:
                    if board.turn == shogi.BLACK:
                        piece_bb = copy.deepcopy(board.piece_bb)
                        occupied = copy.deepcopy((board.occupied[shogi.BLACK], board.occupied[shogi.WHITE]))
                        pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.BLACK], board.pieces_in_hand[shogi.WHITE]))
                    else:
                        piece_bb = [bb_rotate_180(bb) for bb in board.piece_bb]
                        occupied = (bb_rotate_180(board.occupied[shogi.WHITE]), bb_rotate_180(board.occupied[shogi.BLACK]))
                        pieces_in_hand = copy.deepcopy((board.pieces_in_hand[shogi.WHITE], board.pieces_in_hand[shogi.BLACK]))

                    # move label
                    move_label = make_output_label(shogi.Move.from_usi(move), board.turn)

                    # result
                    win = 1 if win_color == board.turn else 0

                    positions.append((piece_bb, occupied, pieces_in_hand, move_label, win))
                    #print(len(piece_bb)) # + " " + str(occupied) + " " + str(pieces_in_hand) + " " + str(move_label) + " " + str(win))
                    board.push_usi(move)
            except :
                print("[NG] " + line)
                #print(sys.exc_info())

    return positions

if __name__ == '__main__':
    positions = read_kifu(sys.argv[1])
    board = shogi.Board()
    board.piece_bb = positions[0]
    print(board)
