import pandas as pd
import numpy as np

from tkinter import filedialog
import Offline_test
from weak_labelling_classes.Tests import Dataset_test


class menu():
    
    def __init__(self):
        
        pass
    
    def show_menu(self):
        
        print('Weak labelling BCI testing\n')
        print('1) Test offline method\n')
        print('2) Dataset Test')
        print('4) Quit')
        
        match input('>>'):
            case '1':
                Offline_test.offline_test().start()
            case '2':
                Dataset_test.dataset_test().start()

            case '4':
                pass
            case _:
                print ('invalid input')
                self.show_menu()
        
    
    




