import random, torch, torch.nn as nn
from torch import optim
from torch.distributions import Categorical

def generate_board(i, trainingsepochen, state):
    if state == 'initial':
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
            return generate_board(i, trainingsepochen, state)
        else:
            return spielbrett
    else:
        spielbrett = random_move(i)
        return spielbrett

def random_move(i):
    lösung = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]
    lösung_lösung = lösung.copy()
    vermischungen = round(0.0001 * (i ** 1.05) + 1)
    for j in range(vermischungen):
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

def board_zeigen(board):
    print("".join(f":{t}" for t in board[0:4]))
    print("".join(f":{t}" for t in board[4:8]))
    print("".join(f":{t}" for t in board[8:12]))
    print("".join(f":{t}" for t in board[12:16]))
    print(" ")
    print("_____________________________________________")

lösung = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0]
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

if input("State laden? (J/N): ") == "J":
    model.load_state_dict(torch.load('15Puzzle.pth'))

optimizer = optim.Adam(model.parameters(), lr=1e-3)


# ----------------- Main Loop -----------------------

startepoche = 200000
trainingsepochen = 400000
returns_batch = []
log_probs_batch = []
for i in range(startepoche, startepoche + trainingsepochen):
    log_probs = []
    rewards = []
    board = generate_board(i, trainingsepochen, state='initial')
    anzahl_züge = 200
    action_alt_list = [0,0,0,0]

    old_boards = []

    for y in range(anzahl_züge):
        if i % 200 == 0 and y == 0:
            print("Epoche: ", i)
            print("Vorher: ")
            board_zeigen(board)
        if i % 200 == 0 and y == 99:
            print("Nachher")
            board_zeigen(board)

        reward = 0
        input_vektor = []
        for j in range(16):
            teil_vektor = [0] * 16
            teil_vektor[board.index(j)] = 1
            input_vektor.append(teil_vektor)

        try:
            action_alt_list[action] = 1
        except NameError:
            pass

        output = model(torch.cat([torch.flatten(torch.tensor(input_vektor, dtype=torch.float32)),torch.tensor(action_alt_list, dtype=torch.float32)], dim=0))
        action_alt_list = [0,0,0,0]

        index_null = board.index(0)
        if (index_null+1) % 4 == 0:
            output[1] = -1e10
        if index_null > 11:
            output[0] = -1e10
        if index_null < 4:
            output[2] = -1e10
        if index_null % 4 == 0:
            output[3] = -1e10

        m = Categorical(logits=output)
        probs = m.probs
        action = m.sample()
        log_probs.append(m.log_prob(action))

        if action == 0:
            board[index_null], board[index_null+4] = board[index_null+4], board[index_null]
        elif action == 1:
            board[index_null], board[index_null+1] = board[index_null+1], board[index_null]
        elif action == 2:
            board[index_null], board[index_null-4] = board[index_null-4], board[index_null]
        elif action == 3:
            board[index_null], board[index_null - 1] = board[index_null - 1], board[index_null]
        else:
            print("Fehler bei Auswählen von Action")

        # --------- Gewinnprüfung-------------

        board_list = list(board)

        if board_list == lösung:
            reward += 200
            rewards.append(reward)
            if i % 200 == 0:
                print("Gewonnen, Epoche:", y+1)
                board_zeigen(board)
            break

        #----------Rewardfunktion-----------

        manhattan_distance = 0
        for j in range(len(board_list)-1):
            manhattan_distance += abs(board_list.index(j+1) // 4 - j // 4) + abs(board_list.index(j+1) % 4 - j % 4)

        if y > 0:
            reward = (manhattan_distance_alt - manhattan_distance) * 4
            if board_list in old_boards:
                reward -= 5
            else:
                old_boards.append(board_list)

        manhattan_distance_alt = manhattan_distance

        reward -= 2

        rewards.append(reward)
    rewards = torch.tensor(rewards, dtype=torch.float32)

    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + 0.98 * G  # aktueller reward + diskontierte Zukunft
        returns.append(G)
    returns = list(reversed(returns))
    returns = torch.tensor(returns, dtype=torch.float32)
    log_probs = torch.stack(log_probs)

    returns_batch.append(returns)
    log_probs_batch.append(log_probs)
    if i % 50 == 0:
        returns_batch = torch.cat(returns_batch)
        log_probs_batch = torch.cat(log_probs_batch)
        returns_batch = (returns_batch - returns_batch.mean()) / (returns_batch.std(correction=0) + 1e-8)

        if i % 200 == 0:
            print(returns)

        loss = -(log_probs_batch * returns_batch).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        log_probs_batch = []
        returns_batch = []
        if i % 10000 == 0:
            torch.save(model.state_dict(), '15Puzzle.pth')


