import pandas as pd
import numpy as np

from tkinter import filedialog
import Offline_test
from weak_labelling_classes.Tests import Single_file
from weak_labelling_classes.Tests import Simulated_trainig

class menu():
    
    def __init__(self):
        
        pass
    
    def show_menu(self):
        
        print('Weak labelling BCI testing\n')
        print('1) Test single file')
        #print('2) Dataset Test')
        print('3) Simulated training')
        print('4) Quit')
        
        match input('>>'):
            case '1':
                Single_file.single_file().start()
                print('\n\n\n\n')
            case '2':
               pass
                
            case '3':
                Simulated_trainig.Simulated_Training().start(
                    'D:/BCI_Official_recordings')
                      #'E:/BCI files')

            case '4':
                pass
            case _:
                print ('invalid input')
                self.show_menu()
        
    
    




