#-------------------------------------------------------------------------------
# Name:        LibsAutoInstaller
# Purpose:
#
# Author:      defra
#
# Created:     10/10/2024
# Copyright:   (c) defra 2024
# Licence:     MIT
#-------------------------------------------------------------------------------

import subprocess
import sys

# Fonction pour installer une librairie si elle n'est pas installée
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Liste des packages requis
required_libraries = ["torch", "torchvision", "opencv-python", "numpy", "shapely", "Pillow", "pyautogui", "pygetWindow", "keyboard"]

# Boucle pour installer chaque librairie si elle n'est pas installée
for library in required_libraries:
    try:
        __import__(library)  # Tente d'importer la librairie
    except ImportError:
        print(f"{library} not found. Installing...")
        install(library)
    else:
        print(f"{library} is already installed.")