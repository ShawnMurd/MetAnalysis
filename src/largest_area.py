"""
Functions to Determine the Largest Continguous Area in a 2D Boolean Array

Code is adapted from https://geeksforgeeks.org/find-length-largest-region-boolean-matrix/

Shawn Murdzek
sfm5282@psu.edu
Date Created: April 5, 2020
Environment: local_py (Python 3.6)
"""

#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np


#---------------------------------------------------------------------------------------------------
# Functions
#---------------------------------------------------------------------------------------------------

def DFS(M, row, col, visited, count, iinds, jinds):

    # These arrays are used to get row and column
    # numbers of 8 neighbours of a given cell
    rowNbr = [-1, 0, 0, 1]
    colNbr = [0, 1, -1, 0]

    # Mark this cell as visited
    visited[row][col] = True

    # Recur for all connected neighbours
    for k in range(4):
        if ((row + rowNbr[k] >= 0) and (row + rowNbr[k] < M.shape[0]) and
            (col + colNbr[k] >= 0) and (col + colNbr[k] < M.shape[1])):

            if (M[row + rowNbr[k], col + colNbr[k]] and
                not visited[row + rowNbr[k], col + colNbr[k]]):

                # increment region length by one
                count[0] += 1
                iinds.append(row + rowNbr[k])
                jinds.append(col + colNbr[k])
                DFS(M, row + rowNbr[k], col + colNbr[k], visited, count, iinds, jinds)


def largestArea(x):

    # Create array to keep track of which array elements have been counted

    visited = np.zeros(x.shape)
    iinds_f = []
    jinds_f = []
    result = 0

    # Loop through each value in x

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):

            if x[i, j] and not visited[i, j]:

                # New region

                count = [1]
                iinds = [i]
                jinds = [j]
                DFS(x, i, j, visited, count, iinds, jinds)

                if count[0] > result:
                    result = count[0]
                    iinds_f = iinds
                    jinds_f = jinds

    return result, iinds_f, jinds_f


"""
End largest_area.py
""" 
