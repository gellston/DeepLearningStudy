# -*- coding: utf-8 -*-
"""
SYS-611: Tic-Tac-Toe Example

@author: Paul T. Grogan, pgrogan@stevens.edu
"""

# import the python3 behavior for importing, division, and printing in python2
from __future__ import absolute_import, division, print_function

# import the pandas library and refer to it as `pd`
import pandas as pd

# define the game state as a list of lists with 3x3 grid cells
# initialize the cells to a blank space character
state = [
    [" "," "," "],
    [" "," "," "],
    [" "," "," "]
]

def reset_game():
    for i in range(3):
        for j in range(3):
            state[i][j] = " "

# define a function to check if a mark is valid
def is_valid(row, col):
    # check if the row/column is empty
    return state[row][col] == " "

# define a function to mark an 'x' at a row and column
def mark_x(row, col):
    # check if this is a valid move
    if is_valid(row, col):
        # if valid, update the state accordingly
        state[row][col] = "x"

# define a function to mark an 'o' at a row and column
def mark_o(row, col):
    # check if this is a valid move
    if is_valid(row, col):
        # if valid, update the state accordingly
        state[row][col] = "o"

# define a function to print out the grid to the console
def show_grid():
    # use the pandas dataframe to help format the matrix
    print(pd.DataFrame(state))
    print("\n")
    
def get_winner():

    for i in range(3):
        count_p1 = 0
        count_p2 = 0
        for j in range(3):
            if state[i][j] == 'o' :
                count_p1 += 1
            if state[i][j] == 'x':
                count_p2 += 1

        if count_p1 == 3 :
            print('p1 win')
            return
        if count_p2 == 3 :
            print('p2 win')
            return

    for i in range(3):
        count_p1 = 0
        count_p2 = 0
        for j in range(3):
            if state[j][i] == 'o' :
                count_p1 += 1
            if state[j][i] == 'x':
                count_p2 += 1

        if count_p1 == 3:
            print('p1 win')
            return
        if count_p2 == 3:
            print('p2 win')
            return

    if (state[0][0] == 'o' and state[1][1] == 'o' and state[2][2] == 'o') or (state[2][0] == 'o' and state[1][1] == 'o' and state[0][2] == 'o'):
        print('p1 win')
        return

    if (state[0][0] == 'x' and state[1][1] == 'x' and state[2][2] == 'x') or (state[2][0] == 'x' and state[1][1] == 'x' and state[0][2] == 'x'):
        print('p2 win')
        return

def is_tie():
  pass # replace this line for HW-01

#%% example game sequence

mark_x(1, 1)
show_grid()
get_winner()

mark_o(0, 2)
show_grid()
get_winner()

mark_x(0, 0)
show_grid()
get_winner()

mark_x(2, 2)
show_grid()
get_winner()


