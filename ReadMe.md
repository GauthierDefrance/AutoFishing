# AutoFishing

### This programms can read skill check that look like this :

_The programs won't work if you don't play in full screen + 1920x1080 pixels_

![SkillCheckExemple](/source/SkillCheckExemple.png)

### It will see them like that after a quick processing

![WhatTheScriptSee](/source/WhatTheScriptSee.png)

Once the programs detect that the red cursor is inside the blue part.
The model will be called and asked to recognize what is the letter in the middle of the screen.
Once the letter has been predicted (it's not 100% probability), it will press in your game
the corresponding touch after 0.1 seconds.

## How to use the program ?

The program need your game to be **ALWAYS** on screen.
For the moment, it's better if you have two screens.
One screen with your game, and another one with 

##### Warning

You need to have Python installed on your computer and in path.
Here is the installer : https://www.python.org/downloads/windows/
_You will have to click on the box add to path_


## How to start the program ?

You can start the program by launching the **launch.bat** file.

You may have a warning message from Windows defender poping on screen.
It's normal, no the programs should normaly not harm your computer.
The warning is because i'm not an official developper.

Once you launched the program, he is waiting for you to press **Enter**.
Then it will auto click when needed.

## How to stop the program ?

If you want to stop the program press **Escape** during 1 secondes.
It should stop.

# What do you need ?

#### I made just for you : InstallMissingLibs.bat, just launch it and it will install all the libs you need.

- Python installed in your machine + In PATH
- Torch library
- torchvision
- opencv-Python
- PIL
- numpy
- shapely
- pillow
- pyautogui
- pygetwindow
- keyboard

## Make sur that you have thoses files :

- main.py 
- Scanner.py
- CaptureEcran.py
