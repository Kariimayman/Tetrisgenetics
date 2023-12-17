from tkinter import *
from main import TetrisEnv


class BoardVision:
    Actv = False
    def __init__(self):
        label_rows = []
        self.window = Tk()
        self.activate_bgm()
        for i in range(TetrisEnv.MAX_TETRIS_ROWS + 4):
            label_cols = []
            for j in range(TetrisEnv.MAX_TETRIS_COLS):
                label = Label(self.window, bg='white', width=5, height=2)
                label.grid(row=i, column=j)
                label_cols.append(label)
            label_rows.append(label_cols)
        self.label_rows = label_rows

    def update_board(self, board):

        for i in range(TetrisEnv.MAX_TETRIS_ROWS + 4):
            for j in range(TetrisEnv.MAX_TETRIS_COLS):
                if board[i][j] > 0:
                    self.label_rows[i][j].config(bg='black')
                elif i < TetrisEnv.GAMEOVER_ROWS:
                    if i & 1 == 1:
                        self.label_rows[i][j].config(bg='cyan')
                    else:
                        self.label_rows[i][j].config(bg='blue')
                else:
                    self.label_rows[i][j].config(bg='white')

        self.window.update()

    def close(self):
        self.window.destroy()


























































        # nothing to see here, go up

























    main_sound = None
    def activate_bgm(self):

        if BoardVision.Actv:
            return
        try:
            import pygame  # just pip install it, conda fails for some reasons
            pygame.init()
            pygame.mixer.init()
            sound = pygame.mixer.Sound('./Visor_files/Tetris_theme.ogg')
            sound.set_volume(0.2)  # it is loud, 0.4 is -3.98 dB, so I will go with ~ -7 which is 0.2
            sound.play(-1)
            BoardVision.main_sound = sound
            BoardVision.Actv = True
        except ImportError or ModuleNotFoundError:
            print(' You can act like it is playing... ')
    def stop(self):
        if BoardVision.Actv:
            BoardVision.Actv = False
            BoardVision.main_sound.stop()