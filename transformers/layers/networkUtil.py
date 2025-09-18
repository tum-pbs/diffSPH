    
from typing import Union, Tuple
def verbosePrint(message: str, verbose: bool, separator = False):
    if verbose:
        if separator:
            print(f'===============================================================')
        print(message)
