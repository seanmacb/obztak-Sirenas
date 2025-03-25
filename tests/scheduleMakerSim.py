#!/usr/bin/env python

"""
This is a function to simulate dates for testing observing blocks over a semester
"""

import numpy as np

def convert(num,start):
    return start+np.timedelta64(int(num),"D")

def run():

    start = np.datetime64("2025-08-01")
    end = start + np.timedelta64(182,"D")
    
    delta = (end-start).tolist().days
    
    baseNumArr = np.arange(delta)

    
    outArr = []
    n_nights = 0
    while 18 >= (n_nights):
        randomNum = np.random.random()*4
        
        if randomNum<1:
            half = "first"
            inc = 0.5
        elif randomNum <2:
            half = "second"
            inc = 0.5
        else:
            half = "full"
            inc = 1
        n_nights+=inc

        dateSelect = convert(np.floor(np.random.random()*(delta+1)),start)

        while dateSelect in np.array(outArr).flatten():
            dateSelect = convert(np.floor(np.random.random()*(delta+1)),start)
        outArr.append([dateSelect,half])
    
    outArr = np.array(outArr)
    outArr[:,1] = outArr[:,1][np.argsort(outArr[:,0])]
    outArr[:,0] = np.sort(outArr[:,0])
    outArr[:,0] = outArr[:,0].astype(str)

    print("Here are your nights")
    print("========"*15)
    print(outArr)
    return 0

if __name__=="__main__":
    run()
