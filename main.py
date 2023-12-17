import random

import numpy as np
from random import Random


def condensed_print(matrix):
    for i in matrix:
        for j in i:
            print(j, end='')
        print()


def print_all_forms():
    for piece in TetrisEnv.Pieces:
        print(piece + ":")
        print('---')
        condensed_print(TetrisEnv.Pieces[piece])
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], axes=(1, 0)))
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], 2, axes=(1, 0)))
        print('#')
        condensed_print(np.rot90(TetrisEnv.Pieces[piece], 3, axes=(1, 0)))
        print('---')
        print()


class TetrisEnv:
    SCORE_PIXEL = 1
    SCORE_SINGLE = 40 * 10
    SCORE_DOUBLE = 100 * 10
    SCORE_TRIPLE = 300 * 10
    SCORE_TETRIS = 1200 * 10
    MAX_TETRIS_ROWS = 20
    GAMEOVER_ROWS = 4
    TOTAL_ROWS = MAX_TETRIS_ROWS + GAMEOVER_ROWS
    MAX_TETRIS_COLS = 10
    GAMEOVER_PENALTY = -1000
    TETRIS_GRID = (TOTAL_ROWS, MAX_TETRIS_COLS)
    TETRIS_PIECES = ['O', 'I', 'S', 'Z', 'T', 'L', 'J']
    # Note, pieces are rotated clockwise
    Pieces = {'O': np.ones((2, 2), dtype=np.byte),
              'I': np.ones((4, 1), dtype=np.byte),
              'S': np.array([[0, 1, 1], [1, 1, 0]], dtype=np.byte),
              'Z': np.array([[1, 1, 0], [0, 1, 1]], dtype=np.byte),
              'T': np.array([[1, 1, 1], [0, 1, 0]], dtype=np.byte),
              'L': np.array([[1, 0], [1, 0], [1, 1]], dtype=np.byte),
              'J': np.array([[0, 1], [0, 1], [1, 1]], dtype=np.byte),
              }
    '''
    I:   S:      Z:      T:
      1      1 1    1 1     1 1 1
      1    1 1        1 1     1
      1
      1
    L:      J:      O:
      1        1      1 1
      1        1      1 1
      1 1    1 1
     last one is utf
    '''

    def __init__(self):
        self.RNG = Random()  # independent RNG
        self.default_seed = 17  # default seed is IT
        self.__restart()

    def __restart(self):
        self.RNG.seed(self.default_seed)
        self.board = np.zeros(self.TETRIS_GRID, dtype=np.byte)
        self.current_piece = self.RNG.choice(self.TETRIS_PIECES)
        self.next_piece = self.RNG.choice(self.TETRIS_PIECES)
        self.score = 0

    def __gen_next_piece(self):
        self.current_piece = self.next_piece
        self.next_piece = self.RNG.choice(self.TETRIS_PIECES)

    def set_seed(self, seed_value):
        self.default_seed = seed_value

    def get_status(self):
        return self.board.copy(), self.current_piece, self.next_piece

    # while can move down piece, move it down (note to restrict col to rotation max)
    # which is COLS-1 - (piece width in cur rotation -1) or cancel both -1s utf-8 #
    # check if move down, row++, if not, print piece on last row, col
    def __get_score(self, value):
        if value == 1:
            return TetrisEnv.SCORE_SINGLE
        if value == 2:
            return TetrisEnv.SCORE_DOUBLE
        if value == 3:
            return TetrisEnv.SCORE_TRIPLE
        if value == 4:
            return TetrisEnv.SCORE_TETRIS
        return 0

    def __collapse_rows(self, board):
        start_collapse = -1
        for row, i in zip(board, range(TetrisEnv.TOTAL_ROWS)):
            if np.sum(row) == TetrisEnv.MAX_TETRIS_COLS:
                start_collapse = i
                break
        if start_collapse == -1:
            return 0, board
        end_collapse = start_collapse + 1
        while end_collapse < TetrisEnv.TOTAL_ROWS:
            if np.sum(board[end_collapse]) == TetrisEnv.MAX_TETRIS_COLS:
                end_collapse += 1
            else:
                break
        new_board = np.delete(board, slice(start_collapse, end_collapse), axis=0)  # now we need to add them
        new_board = np.insert(new_board, slice(0, end_collapse - start_collapse), 0, axis=0)
        score = self.__get_score(end_collapse - start_collapse)

        return score, new_board

    def __game_over(self, test_board):
        return np.sum(test_board[:TetrisEnv.GAMEOVER_ROWS]) > 0

    def __play(self, col, rot_count):
        falling_piece = self.Pieces[self.current_piece]
        if rot_count > 0:
            falling_piece = np.rot90(falling_piece, rot_count, axes=(1, 0))
        p_dims = falling_piece.shape
        col = min(col, TetrisEnv.MAX_TETRIS_COLS - p_dims[1])
        max_row = TetrisEnv.TOTAL_ROWS - p_dims[0]
        chosen_row = 0
        while chosen_row < max_row:
            next_row = chosen_row + 1
            if np.sum(np.multiply(falling_piece,
                                  self.board[next_row:next_row + p_dims[0], col:col + p_dims[1]])) > 0:
                break
            chosen_row = next_row
        self.board[chosen_row:chosen_row + p_dims[0], col:col + p_dims[1]] |= falling_piece
        collapse_score, new_board = self.__collapse_rows(self.board)
        collapse_score += np.sum(falling_piece) * TetrisEnv.SCORE_PIXEL
        if self.__game_over(new_board):
            return TetrisEnv.GAMEOVER_PENALTY
        self.board = new_board
        return collapse_score

    # does not affect the class, tests a play of the game given a board and a piece b64 #
    def test_play(self, board_copy, piece_type, col, rot_count):
        falling_piece = self.Pieces[piece_type]
        if rot_count > 0:
            falling_piece = np.rot90(falling_piece, rot_count, axes=(1, 0))
        p_dims = falling_piece.shape
        col = min(col, TetrisEnv.MAX_TETRIS_COLS - p_dims[1])
        max_row = TetrisEnv.TOTAL_ROWS - p_dims[0]
        chosen_row = 0
        while chosen_row < max_row:
            next_row = chosen_row + 1
            if np.sum(np.multiply(falling_piece,
                                  board_copy[next_row:next_row + p_dims[0], col:col + p_dims[1]])) > 0:
                break
            chosen_row = next_row
        board_copy[chosen_row:chosen_row + p_dims[0], col:col + p_dims[1]] |= falling_piece
        collapse_score, board_copy = self.__collapse_rows(board_copy)
        collapse_score += np.sum(falling_piece) * TetrisEnv.SCORE_PIXEL
        if self.__game_over(board_copy):
            return TetrisEnv.GAMEOVER_PENALTY, board_copy
        return collapse_score, board_copy

    def __calc_rank_n_rot(self, scoring_function, genetic_params, col):
        # should return rank score and rotation a pair (rank,rot), rot is from 0 to 3
        return scoring_function(self, genetic_params, col)

    def __get_lose_msg(self):
        # if understood, send to owner
        lose_msg = b'TFVMISBfIFlPVSBMT1NFIQrilZbilKTilKTilLzilZHilaLilaLilaLilaLilaLilaLilaPilaLilaLilaPilaLilaLilaLilazilazilazilazilazilazilaPilaPilaLilaLilaLilaLilaLilaLilaPilaLilazilazilazilazilazilaPilaPilaPilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaIK4pSk4pWc4pWc4pSC4pSU4pSU4pWZ4pWZ4pWc4pWc4pWc4pWZ4pWZ4pWZ4pWZ4pWc4pWc4pWi4pWi4pWi4pWi4pWr4pWs4pWj4pWj4pWj4pWj4pWi4pWi4pWi4pWi4pWi4pWc4pWc4pWc4pWc4pWc4pWc4pWc4pWi4pWj4pWj4pWc4pWc4pWc4pWc4pWc4pSk4pSC4pSC4pSC4pSC4pWc4pWc4pWc4pWR4pWc4pWR4pWi4pWiCuKVnOKUguKUlCAgICAgICAgICAgIOKUguKUguKUguKVkeKVouKVouKVouKVouKVouKVouKVo+KVouKVouKVouKVouKVnOKVnOKUguKUguKUguKUguKUlCAgIOKUlOKUlOKUlOKUlOKUlCDilJTilZnilKTilKTilKTilKTilZzilZzilZzilZzilZzilZzilZzilZwK4pSC4pSU4pSM4pSM4pSM4pWT4pWT4pWT4pWT4pWT4pWT4pWT4pWT4pWTICAg4pSU4pWZ4pWi4pWi4pWi4pWR4pWi4pWi4pWj4pWs4pWi4pWR4pWc4pSC4pSC4pSC4pSC4pSUICAgICAgICDilZPilZbilZbilZbilZbilZbilZbilKTilKTilKTilKTilKTilKTilZzilKTilKTilZwK4pSC4pSC4pWT4pWR4pWi4pWi4pWi4pWi4pWj4pWj4pWj4pWi4pWi4pWi4pWj4pWj4pWW4pWW4pSM4pSU4pWZ4pWc4pWc4pWZ4pWi4pWi4pWj4pWi4pWi4pWi4pSk4pSC4pSC4pSC4pWW4pWW4pWW4pWW4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWj4pWj4pWi4pWi4pWi4pWi4pSk4pWc4pSk4pWc4pWc4pWc4pWcCuKUguKUlCAgICAgICAg4pSU4pWZ4pWc4pWc4pSC4pWZ4pWc4pWc4pWi4pWW4pWW4pWW4pWW4pWW4pWi4pWr4pWs4pWj4pWi4pWi4pWi4pWi4pWW4pSk4pSC4pSC4pWc4pWc4pWc4pWc4pWc4pWc4pWc4pWc4pWZ4pWZ4pWZ4pWZ4pWZ4pWc4pWc4pWc4pWc4pWc4pWi4pWi4pWR4pSk4pSk4pSkCuKVluKUkCAgIOKVk+KVk+KVluKVluKVluKVluKVluKVluKVluKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVo+KVo+KVo+KVouKVouKVouKVouKVouKVluKVouKVouKUpOKUguKUguKUlCAgICAg4pSM4pSMICAgICAg4pSM4pSC4pSC4pSC4pWc4pWcCuKVouKVluKVnOKUpOKUguKUguKUguKVnOKVnOKVnOKVnOKVouKVouKVouKVnOKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVo+KVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVluKUpOKUvOKVouKVouKVrOKVrOKVrOKVo+KVo+KVouKVouKVouKVouKVo+KVouKVouKVluKVluKVluKVouKVogrilaLilaLilZbilZbilZbilZbilILilILilIzilZPilZbilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaPilaPilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilZbilILilILilILilZnilZnilZnilZzilZzilaLilaLilaLilaLilaLilaLilaLilaLilaLilaIK4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWj4pWi4pWi4pWi4pWi4pWi4pWc4pWc4pWc4pWc4pWR4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWi4pWi4pWi4pWi4pWi4pWj4pWi4pWi4pWi4pWi4pWi4pWiCuKVnOKVouKVouKVouKVouKVouKVouKVouKVo+KVo+KVouKVouKVouKVouKVouKVouKVnOKVnOKVnOKVnOKVluKVouKVouKVouKVouKVouKVouKVouKVouKVnOKVnOKVnOKVouKVouKVouKVnOKVnOKVq+KVrOKVrOKVrOKVo+KVouKVouKVouKVouKVouKVouKVouKVrOKVrOKVrOKVrOKVo+KVo+KVouKVouKVouKVouKVogrilZHilaLilaLilaLilaLilaLilaLilaLilaPilaPilaLilaLilaLilZzilZzilaLilZHilKTilILilZHilaLilaLilaLilaLilaPilaLilKTilKTilKTilILilILilZbilZHilaLilaLilZbilZbilKTilZzilZzilavilazilazilazilazilazilazilazilazilazilazilazilaPilaPilaPilaLilaPilaLilaLilaIK4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWc4pSC4pSC4pWR4pWc4pWc4pWc4pWc4pWc4pWi4pWi4pWc4pWc4pWc4pWc4pWc4pWc4pSk4pWW4pWR4pWi4pWi4pWi4pWi4pWc4pWc4pWZ4pWi4pWi4pWW4pWZ4pWZ4pWi4pWr4pWs4pWs4pWs4pWs4pWs4pWj4pWj4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWiCuKVnOKVnOKVnOKVnOKVouKVouKVouKVouKVouKVnOKVnOKUguKVluKVouKVnOKVnOKVmeKVmeKVnOKUpOKUpOKUpOKUguKUguKUguKUguKVnOKVnOKVnOKVnOKVnOKVqOKVqOKVnOKVnOKVnOKUguKUguKVkeKVouKVo+KVouKUpOKVnOKVnOKVnOKVnOKVnOKVnOKVouKVouKVouKVouKVouKVouKVouKVouKVouKVouKVogrilILilILilILilILilZzilZzilaLilZzilZzilILilILilZPilZzilJggICAgICAgIOKUlOKUlOKUlCAgICAgICAgIOKUjCAg4pSC4pWc4pSC4pSC4pSC4pWc4pSk4pSk4pWc4pWc4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWc4pWi4pWi4pWR4pWcCuKUguKUguKUguKUguKUguKUguKUguKUguKUguKUguKUguKUmCAgICAgICAgICAg4pSM4pWT4pWT4pSQICDilJTilJTilJTilJTilJQgIOKUlOKUguKUguKUlOKUlOKUlOKUlOKUguKUguKUguKUguKUguKUguKVnOKVnOKVnOKVnOKVnOKVnOKVnOKVnOKVnOKVnOKUggrilILilILilILilILilILilILilILilILilILilILilJQgICAgICAgICAgICAg4pSC4pSCICAgICDilIwgICAgIOKUguKUgiAgICAg4pSU4pSC4pSC4pSC4pSC4pSC4pSC4pWc4pWc4pWc4pWc4pSk4pSk4pSC4pSC4pSCCuKUguKUguKUguKUguKUguKUguKUguKUguKUgiAgICAgICAgICAgICAg4pSM4pWT4pWW4pSQICAgIOKUguKUgiAgICAg4pSUICAgICAgICAg4pSU4pSC4pSC4pSC4pSC4pSC4pSk4pSk4pSC4pSC4pSC4pSkCuKUguKUguKUguKUguKUguKUguKUguKUpOKUmCAgICAgICAgICAgICDilJTilZnilZzilZzilZzilKTilJAg4pSM4pSC4pSM4pSMICAg4pSM4pSM4pSQ4pSMICAgICAgICAg4pWZ4pWc4pSC4pSC4pSC4pSk4pSk4pSk4pSk4pSkCuKUguKUguKUguKUguKUguKUguKUguKUpOKUkCAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICDilJTilJTilJjilJAgICAgIOKUlOKVnOKVnOKVnOKUpOKUpOKUpOKUpArilILilILilILilILilILilILilILilZHilZbilZbilJDilZPilILilIIgICDilJTilavilazilazilaPilIAgICAgICAgICAgICAgICAgICAgICAgICAg4pSU4pWW4pWW4pWW4pSC4pSC4pWW4pSk4pSk4pWc4pSk4pSCCuKUguKUguKUguKUguKUguKUguKUguKVnOKVnOKVouKVo+KUpOKUguKUguKVkeKVouKVliAgICAgICDilZPilZPilZMgICAgICAgIOKVk+KVk+KVk+KVluKVluKVluKVluKVluKVluKVluKUkCAg4pWT4pWW4pSC4pSC4pSC4pSC4pWR4pWc4pWc4pSk4pSC4pSCCuKUguKUguKUguKUguKUguKUguKUguKUguKUguKVmeKVouKUpOKUguKUguKVmeKVouKVouKUpOKVluKVluKVpeKVpeKVo+KVo+KVouKVouKVouKVrOKVrOKVrOKVrOKVrOKVrOKVrOKVo+KVo+KVo+KVouKVouKVouKVnOKVnOKVnOKVnOKUguKUguKVk+KVkeKVouKVnOKUguKUguKUguKUguKVnOKVnOKUpOKUguKUguKUggrilILilILilILilILilILilILilILilILilILilZzilZzilKTilKTilZbilILilZHilaLilKTilILilILilZnilaLilaLilaPilaPilaPilaLilaPilaLilaLilaLilaLilaLilaLilaLilaLilaLilaLilaPilaLilZzilKTilILilILilZbilaLilaPilZzilZzilKTilILilILilILilKTilZzilKTilKTilILilILilIIK4pSC4pSC4pSC4pSC4pSC4pSC4pSC4pWR4pWW4pSC4pSC4pSk4pSk4pSk4pSk4pSk4pWR4pSk4pSC4pSC4pSC4pSC4pWR4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWi4pWj4pWi4pWc4pWc4pWc4pWc4pWc4pSC4pSC4pSC4pWT4pWc4pWi4pWi4pWi4pWi4pWi4pWi4pSk4pSk4pSk4pSk4pWc4pWc4pSk4pSC4pSC4pSC4pSCCuKUguKUguKUguKUguKUguKUguKUguKVkeKVouKVluKUguKVmeKVkeKUpOKUpOKVnOKVnOKVnOKUpOKUguKUguKUguKUpOKVnOKUpOKUpOKVnOKVnOKVnOKVnOKVnOKVnOKUguKVk+KVk+KVq+KVrOKVrOKVmeKVnOKVnOKVnOKVnOKVkeKVouKVouKVouKVouKVouKVnOKUpOKUpOKUpOKVnOKUpOKUpOKUpOKUguKUguKUggrilILilILilILilILilILilILilILilZHilaLilaLilKTilILilZzilZzilKTilILilILilILilKTilZbilILilILilZzilZHilZHilZHilZHilZHilZzilKTilZbilZbilaLilaLilaLilaPilZzilZzilZbilZbilZbilZbilZbilaLilaLilaLilaLilaLilZzilZzilKTilZzilZzilZzilZzilKTilILilILilILilZE='
        return lose_msg

    def run(self, scoring_function, genetic_params, num_of_iters, return_trace):
        self.__restart()
        # no trace
        if not return_trace:
            for it in range(num_of_iters):
                rates = []
                rotations = []
                for c in range(TetrisEnv.MAX_TETRIS_COLS):
                    r1, r2 = self.__calc_rank_n_rot(scoring_function, genetic_params, c)
                    rates.append(r1)
                    rotations.append(r2)
                pos_to_play = rates.index(max(rates))  # plays first max found
                rot_to_play = rotations[pos_to_play]
                play_score = self.__play(pos_to_play, rot_to_play)
                self.score += play_score
                self.__gen_next_piece()
                if play_score < 0:
                    return self.score, self.board, self.__get_lose_msg()
            return self.score, self.board, ""
        else:  # we want to trace
            board_states = []
            ratings_n_rotations = []
            pieces_got = []
            # board_states.append(self.board.copy())
            for it in range(num_of_iters):
                rates = []
                rotations = []
                pieces_got.append(self.current_piece)
                for c in range(TetrisEnv.MAX_TETRIS_COLS):
                    r1, r2 = self.__calc_rank_n_rot(scoring_function, genetic_params, c)
                    rates.append(r1)
                    rotations.append(r2)
                ratings_n_rotations.append(list(zip(rates, rotations)))
                pos_to_play = rates.index(max(rates))  # plays first max found
                rot_to_play = rotations[pos_to_play]
                play_score = self.__play(pos_to_play, rot_to_play)
                self.score += play_score
                self.__gen_next_piece()
                board_states.append(self.board.copy())
                if play_score < 0:
                    return self.score, board_states, ratings_n_rotations, pieces_got, self.__get_lose_msg()
            return self.score, board_states, ratings_n_rotations, pieces_got, ""
        # don't really feel like removing redundancy, cleaning code


# max gain + random

def scoring_function(tetris_env: TetrisEnv, gen_params, col):
    board, piece, next_piece = tetris_env.get_status()  # add type hinting
    scores = []

    def maxheight(tetrisboard):
        height = 23
        for row in range(len(tetrisboard)):
            if 1 not in tetrisboard[row]:
                height = row
        height = abs(height - 23)
        if height > 20:
            return 999
        return height

    def zeroscore(tetrisboard, height):
        board = tetrisboard[::-1]
        score = 0
        for i in range(height):
            for j in range(len(board[i])):
                if board[i][j] == 0:
                    score += 1
        return score

    for i in range(4):
        score4, tmp_board = tetris_env.test_play(board, piece, col, i)
        high = maxheight(tmp_board)
        if high == 999:
            score = -9999
        else:
            score = (-high * gen_params[0]) + (score4 * gen_params[1]) + ((-zeroscore(tmp_board, high)) * gen_params[5])
        tmp_scores = []
        for t in range(tetris_env.MAX_TETRIS_COLS):
            for j in range(4):
                score3, tmp_board2 = tetris_env.test_play(tmp_board, next_piece, t, j)
                high = maxheight(tmp_board2)
                if high == 999:
                    score2 = -9999
                else:
                    score2 = (-high * gen_params[2]) + (score3 * gen_params[3]) + (
                                (-zeroscore(tmp_board2, high)) * gen_params[6])
                tmp_scores.append(score2)
        max_score2 = max(tmp_scores)
        score += max_score2
        score = score * gen_params[4]
        scores.append([score, i])
    val = max(scores, key=lambda item: item[0])  # need to store it first or it iterates
    # print(val)
    return val[0], val[1]


if __name__ == "__main__":
    use_visuals_in_trace = True
    sleep_time = 0.8
    population = []
    for _ in range(15):
        solution = [random.randint(0, 10) for _ in range(5)]
        population.append(solution)

    print_all_forms()
    env = TetrisEnv()
    env.set_seed(5132)
    for generation in range(5):
        populationscores = []
        for chromosome in population:
            total_score, states, rate_rot, pieces, msg = env.run(scoring_function, chromosome, 600, True)
            print(total_score)
            populationscores.append(total_score)
        sorted_indices = np.argsort(populationscores)[::-1]
        max_indices = sorted_indices[:4]
        newchromosome1 = population[max_indices[0]][:3] + population[max_indices[1]][2:]
        newchromosome2 = population[max_indices[2]][:3] + population[max_indices[3]][2:]
        if newchromosome1 in population and newchromosome2 in population:
            break;
        population[sorted_indices[-1]] = newchromosome1
        population[sorted_indices[-2]] = newchromosome2
    bestchromosome = population[sorted_indices[0]]
    print(populationscores)
    print("best chromosome : " ,end = "")
    print(bestchromosome,end = "")
    print("with score of : ", end="")
    print(populationscores[sorted_indices[0]])

    train
    from Visor import BoardVision
    import time


    def print_stats(use_visuals_in_trace_p, states_p, pieces_p, sleep_time_p):
        vision = BoardVision()
        if use_visuals_in_trace_p:

            for state, piece in zip(states_p, pieces_p):
                vision.update_board(state)
                # print("piece")
                # condensed_print(piece)
                # print('-----')
                time.sleep(sleep_time_p)
            time.sleep(2)
            vision.close()
        else:
            for state, piece in zip(states_p, pieces_p):
                print("board")
                condensed_print(state)
                print("piece")
                condensed_print(piece)
                print('-----')


    env = TetrisEnv()
    # total_score, states, rate_rot, pieces, msg = env.run(
    #     random_scoring_function, one_chromo_rando, 100, True)
    total_score, states, rate_rot, pieces, msg = env.run(
        scoring_function, [94, 96, 61, 48, 18, 56, 12], 300, True)
    # after running your iterations (which should be at least 500 for each chromosome)
    # you can evolve your new chromosomes from the best after you test all chromosomes here
    print("Ratings and rotations")
    for rr in rate_rot:
        print(rr)
    print('----')
    print(total_score)
    print(msg)
    print_stats(use_visuals_in_trace, states, pieces, sleep_time)
