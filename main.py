#-------------------------------------------------------------------------------
# Name:        main.py
# Purpose:
#
# Author:      Killer
#
# Created:     10/10/2024
# Copyright:   (c) Killer 2024
# Licence:     MIT
#-------------------------------------------------------------------------------

from Scanner import *


def main():
    try:
        start()
    except:
        print("Five M n'est pas visible sur l'écran.")
        print("Il faut que vous appuyez 1 secondes sur Entrée une fois que vous souhaitez pêcher en jeu.")
        print("Cliquez sur echap pendant 1 secondes pour arrêter le programme.")

if __name__ == '__main__':
    main()
