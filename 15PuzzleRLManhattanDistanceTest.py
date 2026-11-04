import random, torch, torch.nn as nn
import pygame
import time

def generate_board():
    spielbrett = []
    inversionen = 0
    for y in range(16):
        spielbrett.append(y)
    random.shuffle(spielbrett)
    spielbrett_ohne_null = [tile for tile in spielbrett if tile != 0]
    for y in range(15):
        for j in range(y+1, 15):
            if spielbrett_ohne_null[j] < spielbrett_ohne_null[y]:
                inversionen += 1
    zeile_null = 4 - (spielbrett.index(0) // 4)

    if not (inversionen + zeile_null) % 2 != 0:
        return generate_board()
    else:
        return spielbrett

def random_move(i):
    lösung = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    lösung_lösung = lösung.copy()
    for j in range(i):
        wahrscheinlichkeiten = [1] * 4
        index_null = lösung.index(0)
        if index_null > 11:
            wahrscheinlichkeiten[0] = 0
        if (index_null + 1) % 4 == 0:
            wahrscheinlichkeiten[1] = 0
        if index_null < 4:
            wahrscheinlichkeiten[2] = 0
        if index_null % 4 == 0:
            wahrscheinlichkeiten[3] = 0
        action = random.choices([0, 1, 2, 3], weights=[wahrscheinlichkeiten[0], wahrscheinlichkeiten[1], wahrscheinlichkeiten[2], wahrscheinlichkeiten[3]], k=1)[0]

        if action == 0:
            lösung[index_null], lösung[index_null + 4] = lösung[index_null + 4], lösung[index_null]
        elif action == 1:
            lösung[index_null], lösung[index_null + 1] = lösung[index_null + 1], lösung[index_null]
        elif action == 2:
            lösung[index_null], lösung[index_null - 4] = lösung[index_null - 4], lösung[index_null]
        elif action == 3:
            lösung[index_null], lösung[index_null - 1] = lösung[index_null - 1], lösung[index_null]
    if lösung == lösung_lösung:
        return random_move(i)
    else:
        return lösung

class Agent(nn.Module):
    def __init__(self, input_size=260, hidden_size=256, hidden_size2=128, output_size=4):
        super(Agent, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size2)
        self.layer4 = nn.Linear(hidden_size2, output_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

model = Agent()
model.load_state_dict(torch.load('15Puzzle.pth'))

pygame.init()
screen = pygame.display.set_mode((400, 400))
font = pygame.font.SysFont("Arial", 30)

def show_board(board):
    screen.fill((0, 0, 0))
    for i in range(len(board)):
        tile = font.render(str(board[i]), True, (255,255,255))

        screen.blit(tile, (((i % 4) * 80) + 60, ((i // 4) * 80) + 60))
        pygame.display.flip()

lösung = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]

board = random_move(400)
action_alt_list = [0,0,0,0]
for y in range(200):
    show_board(board)

    time.sleep(0.5)

    input_vektor = []
    for j in range(16):
        teil_vektor = [0] * 16
        teil_vektor[board.index(j)] = 1
        input_vektor.append(teil_vektor)

    try:
        action_alt_list[action] = 1
    except NameError:
        pass

    output = model(torch.cat([torch.flatten(torch.tensor(input_vektor, dtype=torch.float32)),
                              torch.tensor(action_alt_list, dtype=torch.float32)], dim=0))
    action_alt_list = [0, 0, 0, 0]

    index_null = board.index(0)
    if (index_null + 1) % 4 == 0:
        output[1] = -1e10
    if index_null > 11:
        output[0] = -1e10
    if index_null < 4:
        output[2] = -1e10
    if index_null % 4 == 0:
        output[3] = -1e10

    action = list(output).index(max(list(output)))

    if action == 0:
        board[index_null], board[index_null + 4] = board[index_null + 4], board[index_null]
    elif action == 1:
        board[index_null], board[index_null + 1] = board[index_null + 1], board[index_null]
    elif action == 2:
        board[index_null], board[index_null - 4] = board[index_null - 4], board[index_null]
    elif action == 3:
        board[index_null], board[index_null - 1] = board[index_null - 1], board[index_null]
    else:
        print("Fehler bei Auswählen von Action")

    board_list = list(board)
    if board_list == lösung:
        print("Gewonnen!")
        show_board(board)
        time.sleep(5)
        break