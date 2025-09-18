    
from typing import Union, Tuple
def verbosePrint(message: str, verbose: bool, separator = False, width = 80):
    if verbose:
        if separator:
            print('=' * width)
        print(message)


        ################################################################################
        #                     Encode Edge Attributes for RPB                           #
        ################################################################################

def verboseBannerPrint(message: str, verbose: bool, width = 80):
    if verbose:
        print('=' * width)
        for line in message.split('\n'):
            print(f'#{line.center(width - 2)}#')
        print('=' * width)