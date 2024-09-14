from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.01

LEFT, RIGHT, UP, DOWN = [i for i in range(4)]
DIRS = [i for i in range(4)]

# Rewards
DEATH_REWARD = -1000
SNACK_REWARD = 500
WIN_REWARD = 500
TIE_REWARD = 0

NUM_ACTIONS = 4
STATE_SHAPE = [4, 2, 5, 5, 2, 5]

RADIUS = 2
WALLS_THICKNESS = 1

class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.r = RADIUS
        # self.name = name

        self.rewards, self.num_snacks = [], []
        self.num_border_deaths, self.num_snake_deaths = [], []
            
        # either load a pre-trained Q-table or initialize Q-table as all zeros
        try: 
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros(STATE_SHAPE + [NUM_ACTIONS])

        # initialize lr, df and eps with values defined earlier
        self.lr = LEARNING_RATE
        self.df = DISCOUNT_FACTOR
        self.eps = EPSILON

    def use_epsilon_decay(self):
        if self.eps > 0.01:
            self.eps *= 0.999

    def get_optimal_policy(self, state):
        # The Snake dynamically chooses the best action given the current state.
        possible_actions = self.q_table[*state]
        # sort indices based on values in ascending order
        sorted_indices = np.argsort(possible_actions)
        # reverse the list to convert ascending into descending order
        sorted_indices = sorted_indices[::-1]
        return sorted_indices[0], sorted_indices[1]

    def find_direction(self, dx, dy):
        # this function determines direction of the move
        if dx == 1 and dy == 0:
            return RIGHT
        if dx == -1 and dy == 0:
            return LEFT
        if dx == 0 and dy == -1:
            return UP
        if dx == 0 and dy == 1:
            return DOWN
        # print("nuh uh dir")

    def is_opposite_direction(self, new_dir):
        # this function returns True if the given direction is exactly opposite of the Snake' head direction
        dir = self.find_direction(dx=self.dirnx, dy=self.dirny)
        return (dir == LEFT and new_dir == RIGHT) | \
               (dir == RIGHT and new_dir == LEFT) | \
               (dir == UP and new_dir == DOWN) | \
               (dir == DOWN and new_dir == UP)

    def make_action(self, state):
        # The goal here is to balance exploration with exploitation.
        # exploitation is to greedily choose the best possible action.
        # exploration is to encourage the Snake to try different actions.
        chance = random.random()
        if chance < self.eps:
            while True:
                action = random.randint(0, 3)
                if not self.is_opposite_direction(new_dir=action):
                    break
        else:
            act1, act2 = self.get_optimal_policy(state)
            # choose second best action if the first action is returning backward
            action = act1 if self.is_opposite_direction(new_dir=act1) == False else act2

        return action

    def update_q_table(self, state, action, next_state, reward):
        # Here we update our Q-table using the Q-learning algorithm.
        new_q_value =  reward + self.df * np.max(self.q_table[*next_state]) - self.q_table[*state][action]
        new_q_value *= self.lr
        self.q_table[*state][action] += new_q_value
        # self.use_epsilon_decay()

    def is_threatened(self, r, jiz):
        d = [i for i in range(-r, r+1)]
        for dx in d:
            for dy in d:
                if (dx == 0 and dy == 0) or (abs(dx) + abs(dy) > r):
                    continue
                x, y = self.head.pos
                x += dx
                y += dy
                if (x, y) in jiz:
                    return 1
        return 0
        
    def create_neighbor_cells_state(self, r, jiz):
        d = [i for i in range(-r, r+1)]
        is_in_region = {i: 0 for i in range(1, 5)}
        for dx in d:
            for dy in d:
                if (dx == 0 and dy == 0) or (abs(dx) + abs(dy) > r):
                    continue
                x, y = self.head.pos
                x += dx
                y += dy
                if (x, y) in jiz:
                    if dx >= 0:
                        if abs(dx) >= abs(dy):
                            is_in_region[1] += 1
                    if dx <= 0:
                        if abs(dx) >= abs(dy):
                            is_in_region[3] += 1
                    if dy >= 0:
                        if abs(dy) >= abs(dx):
                            is_in_region[2] += 1
                    if dy <= 0:
                        if abs(dy) >= abs(dx):
                            is_in_region[4] += 1

        if is_in_region[1] == 0 and is_in_region[2] == 0 and \
           is_in_region[3] == 0 and is_in_region[4] == 0:
            return 0
        else:
            return max(is_in_region, key=is_in_region.get)

    def create_snack_state(self, snack):
        dx = snack.pos[0] - self.head.pos[0]
        dy = snack.pos[1] - self.head.pos[1]
        if dx >= 0:
            if abs(dx) >= abs(dy):
                return 0
        else:
            if abs(dx) >= abs(dy):
                return 2
        if dy >= 0:
            if abs(dy) >= abs(dx):
                return 1
        else:
            if abs(dy) >= abs(dx):
                return 3

        # print("nuh uh snack")

    def is_near_borders(self, distance_to_keep_from_borders):
        x, y = self.head.pos
        d = distance_to_keep_from_borders
        if (x < 1 + d)  or \
           (y < 1 + d)  or \
           (x > 18 - d) or \
           (y > 18 - d):
            return 1
        return 0

    def create_border_state(self, distance_to_keep_from_borders):
        x, y = self.head.pos
        d = distance_to_keep_from_borders
        lb, ub = 0 + WALLS_THICKNESS, (ROWS - 1) - WALLS_THICKNESS 
        if x < lb + d:
            return 1
        if y < lb + d:
            return 4
        if x > ub - d:
            return 3
        if y > ub - d:
            return 2
        return 0

    def create_state_space(self, snack, other_snake, r):
        states = []
        states.append(self.create_snack_state(snack=snack))
        states.append(self.is_threatened(r=r, jiz=list(map(lambda z: z.pos, other_snake.body))))
        states.append(self.create_neighbor_cells_state(r=r, jiz=list(map(lambda z: z.pos, other_snake.body))))
        states.append(self.create_neighbor_cells_state(r=r, jiz=list(map(lambda z: z.pos, self.body[1:]))))
        states.append(int(self.is_near_borders(distance_to_keep_from_borders=1)))
        states.append(self.create_border_state(distance_to_keep_from_borders=1))
        return states

    def move(self, snack, other_snake):
        # First we create a state using provided information about our Snake, other Snake and snack
        state = self.create_state_space(snack=snack, other_snake=other_snake, r=self.r)
        
        # Now based on current state, the Snake will choose an action to take
        action = self.make_action(state)

        if action == LEFT:
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == RIGHT:
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == UP:
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == DOWN:
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)

        next_state = self.create_state_space(snack=snack ,other_snake=other_snake, r=self.r)
        return state, next_state, action

    def check_out_of_board(self, point):
        x, y = point
        if x >= ROWS - 1 or x < 1 or y >= ROWS - 1 or y < 1:
            return True
        return False

    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False

        ate_snack, died_border, died_snake = False, False, False
        
        if self.check_out_of_board(point=self.head.pos):
            # Punish the snake for getting out of the board
            # print(self.name, "got out of borders")
            # self.reset((random.randint(3, 18), random.randint(3, 18)))
            reward += DEATH_REWARD
            died_border = True
            win_other = True
            reset(self, other_snake)
        
        if self.head.pos == snack.pos:
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            ate_snack = True
            reward += SNACK_REWARD
            # Reward the snake for eating
            
        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            # Punish the snake for hitting itself
            died_snake = True
            # print(self.name, "hit itself")
            # self.reset((random.randint(3, 18), random.randint(3, 18)))
            reward += DEATH_REWARD
            win_other = True
            reset(self, other_snake)
            
        
        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            # Punish the snake for hitting the other snake' body
            if self.head.pos != other_snake.head.pos:
                # print(self.name, "hit other snake' body")
                # self.reset((random.randint(3, 18), random.randint(3, 18)))
                reward += DEATH_REWARD
                died_snake = True
                win_other = True

            # Two snakes hit each other on the head
            else:
                if len(self.body) > len(other_snake.body):
                    # Reward the snake for hitting the head of the other snake and being longer
                    reward += WIN_REWARD
                    win_self = True
                    # other_snake.reset((random.randint(3, 18), random.randint(3, 18)))
                elif len(self.body) == len(other_snake.body):
                    # No winner
                    reward += TIE_REWARD
                    pass
                else:
                    # Punish the snake for hitting the head of the other snake and being shorter
                    # self.reset((random.randint(3, 18), random.randint(3, 18)))
                    reward += DEATH_REWARD
                    died_snake = True
                    win_other = True

            reset(self, other_snake)

        cur_x, cur_y = self.head.pos
        current_dist_to_snack = abs(cur_x - snack.pos[0]) + abs(cur_y - snack.pos[1])
        # Simulate a move towards the snack to calculate the distance after
        future_dist_to_snack = abs(cur_x + self.dirnx - snack.pos[0]) + abs(cur_y + self.dirny - snack.pos[1])

        if future_dist_to_snack < current_dist_to_snack:
            reward += 200 # Reward for getting closer to the snack
        else :
            reward -= 200 # Punish for getting further from the snack
        

        self.num_snacks.append(int(ate_snack))
        self.num_snake_deaths.append(int(died_snake))
        self.num_border_deaths.append(int(died_border))
        self.rewards.append(reward)

        return snack, reward, win_self, win_other

    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
        
    def break_into_bins(self, l, interval_len):
        output = []
        for i in range(0, len(l), interval_len):
            sublist = l[i:i + interval_len]
            avg = sum(sublist) / len(sublist)
            output.append(avg)
        return output

    def save_results(self, path):
        df_res = pd.DataFrame()
        df_res["snacks"] = self.num_snacks
        df_res["reward"] = self.rewards
        df_res["border death"] = self.num_border_deaths
        df_res["snake death"] = self.num_snake_deaths
        df_res["death"] = df_res["border death"] + df_res["snake death"]
        df_res.to_csv(path)

    def plot_results(self):
        sublist_size=1000
        out_rew = self.break_into_bins(l=self.rewards, interval_len=sublist_size)
        out_det = self.break_into_bins(l=self.num_deaths, interval_len=sublist_size)
        out_sna = self.break_into_bins(l=self.num_snacks, interval_len=sublist_size)
        fig = make_subplots(rows=2, cols=3, column_titles=["Average Reward", "Number of Deaths", "Number of Snacks eaten by the Snake"])
        fig.add_trace(px.line(x=[x for x in range(len(out_rew))], y=out_rew).update_traces(line_color="purple").data[0], row=1, col=1)
        fig.add_trace(px.line(x=[x for x in range(len(out_det))], y=out_det).update_traces(line_color="cyan").data[0], row=1, col=2)
        fig.add_trace(px.line(x=[x for x in range(len(out_sna))], y=out_sna).update_traces(line_color="darkblue").data[0], row=1, col=3)
        fig.add_trace(px.histogram(x=[x for x in range(len(out_rew))], y=out_rew, histfunc="avg", nbins=100).data[0], row=2, col=1)
        fig.add_trace(px.histogram(x=[x for x in range(len(out_det))], y=out_det, histfunc="avg", nbins=100).data[0], row=2, col=2)
        fig.add_trace(px.histogram(x=[x for x in range(len(out_sna))], y=out_sna, histfunc="avg", nbins=100).data[0], row=2, col=3)

        fig.show()