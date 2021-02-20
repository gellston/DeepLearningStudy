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
state = [[" "," "," "],
         [" "," "," "],
         [" "," "," "]]

def reset_game(row, col):
    for i in range(row):
        for j in range(col):
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

    
def get_winner():

    player1_count = 0
    player2_count = 0

    for row in range(3):
        for col in range(3):
            if state[row][col] == 'x':
                player1_count+=1

            if state[row][col] == 'o':
                player2_count+=1

            if player1_count == 3:
                print('x')
                return

            if player2_count == 3:
                print('o')
                return

    player1_count = 0
    player2_count = 0

    for col in range(3):
        for row in range(3):

            if state[row][col] == 'x':
                player1_count+=1

            if state[row][col] == 'o':
                player2_count+=1

            if player1_count == 3:
                print('x')
                return

            if player2_count == 3:
                print('o')
                return

    player1_count = 0
    player2_count = 0

    for d in range(3):
        if state[d][d] == 'x':
            player1_count += 1

        if state[d][d] == 'o':
            player2_count += 1

        if player1_count == 3:
            print('x')
            return

        if player2_count == 3:
            print('o')
            return

    if state[0][2] == 'x' and state[1][1] == 'x' and state[2][0] == 'x' :
        print('x')
        return

    if state[0][2] == 'o' and state[1][1] == 'o' and state[2][0] == 'o':
        print('o')
        return

    print(' ')



def is_tie():
    count = 0
    for row in range(3):
        for col in range(3):
            if state[row][col] == " ":
                count += 1

    if count == 0:
        return True
    else:
        return False


#%% example game sequence


print('test')












