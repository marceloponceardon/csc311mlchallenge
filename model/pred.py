#!/bin/python3
'''
Module docstring:

Instructions:
Submit a python3 script that takes as parameter the name of a CSV file containing the test set, 
and produces predictions for that test set.
'''

# The script can use the following libraries
# As well as basic python imports like sys, csv, random
# You will NOT be able to use sklearn, pytorch, tensorflow, etc. in the final submitted code.
# You may reuse any code that you wrote or was provided to you from the labs.
# We should be able to make ~60 predictions within 1 minute (1 prediction per second)

from importlib.metadata import version
import sys
# import csv
# import random
# import numpy as np
import pandas as pd

def main():
    '''
    TODO:
    Main method, reads in the test data, 
    calls relevant functions to make predictions, 
    and writes to a csv file
    '''
    # Print version of python, and import libraries using importlib
    print("Pandas version: ", version('pandas'))
    print("Numpy version: ", version('numpy'))

    # Check if the correct number of arguments are passed
    if len(sys.argv) != 2:
        print("Usage: python3 pred.py <test_file>")
        sys.exit(1)
    # Read in the test data
    test_data = pd.read_csv(sys.argv[1])
    print(test_data)

if __name__ == "__main__":
    main()
