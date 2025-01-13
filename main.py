# run processs and train py files respectively

import process
import train
import runpy

if __name__ == '__main__':
    runpy.run_path("process.py")
    runpy.run_path("train.py")