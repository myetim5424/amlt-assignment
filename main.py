# run processs and train py files respectively

import process
import train
import runpy

if __name__ == '__main__':
    runpy.run_path("process.py")
    runpy.run_path("train.py")
    runpy.run_path("lstm.py")

    # to run model witn new data
    runpy.run_path("use_lstm_model.py")