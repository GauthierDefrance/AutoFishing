#-------------------------------------------------------------------------------
# Name:        ImageGetter
# Purpose:     A programs that takes screenshot when needed. It takes a pictures, then crop it to get the skill check circle and finally crop it again, to know which letter he need to press.
#
# Author:      Killer
#
# Created:     09/10/2024
# Copyright:   (c) Killer 2024
# Licence:     MIT
#-------------------------------------------------------------------------------
import pyautogui
import pygetwindow as gw
import keyboard
from PIL import Image

def getImage(window_title:str):
    """Prend en paramètre le nom de la fênétre actuelle. Ou un nom commençant par ça."""
    window = gw.getWindowsWithTitle(window_title)[0]
    if window:
        window.activate()
        screenshot = pyautogui.screenshot(region=(window.left, window.top, window.width, window.height))

        width = 120  # width of the zone to cut
        height = 120  # heigth of the zone to cut

        left = (window.width-width)//2  # Pos X
        top = (window.height-height)//2   # Pos Y

        right = left + width
        bottom = top + height


        cropped_image = screenshot.crop((left, top, right, bottom)) #This image should contain a picture of a skill check circle

        #cropped_image.show()  #Show the current skill check picture


        width2 = 28  # width of the zone to cut again
        height2 = 28  # heigth of the zone to cut again

        left2 = (width-width2)//2  # Pos X
        top2 = (height-height2)//2   # Pos Y

        right2 = left2 + width2
        bottom2 = top2 + height2

        cropped_cropped_image = cropped_image.crop((left2, top2, right2, bottom2)) #This image should contain a picture of a letter
        #cropped_cropped_image.show() #Show the current letter picture

        return cropped_image,cropped_cropped_image
    else:
        print("Windows was not found.")


def presstouch(window_title:str,touch:str) -> None:
    """A function that press a 'touch' on the windows given."""

    windows = gw.getWindowsWithTitle(window_title) #Get the windows

    if windows:
        window = windows[0]
        window.activate()

        pyautogui.press(touch)  # Press the button

    else:
        print("Windows was not found.")

def detect_key_press():
    """A program that detect which touch is pressed."""

    if keyboard.is_pressed('esc'): #Return esc if the esc touch has been pressed
        return "esc"
    elif keyboard.is_pressed('enter'): #Return enter if enter has been pressed
        return "enter"
    else:
        return 0
