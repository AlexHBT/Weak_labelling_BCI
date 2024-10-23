from tkinter import filedialog
from .Participant_test import participant_test

class single_file():
    
    def __init__(self):
        pass
    
    def start(self):
        file = filedialog.askopenfilename()

        print(f'testing file: {file}')        

        pt = participant_test()
        results = pt.start_2([file], name = 'Single_part', session = 0)
        print(f'Without method acc {results[1]}: with method accuracy {results[0]}')




