"""
Python Program that prevents idle timeouts via
moving the mouse between two points and providing inputs
"""
#import the libraries
import pyautogui as gui
import time

#global pause set in seconds to wait after each gui call
gui.PAUSE = 5

def main():
    
    #take how long this program will run before timeout
    timeout = 60 * int(input("Enter # of mins that this program wil run for"))
    
    while True:
            
        #Input will wait for a user to press enter before taking the positions
        input("Move mouse to position 1 then hit enter to record position")
        x1,y1 = gui.position()
        input("Move mouse to position 2 and press enter to record position")
        x2,y2 = gui.position()

        #confirm user decision
        string = input(f"{x1}, {y1} & {x2}, {y2} were selected. Is this okay? (Y/n)")
        if string.lower()[0] == 'y':
            break

    #get an initial time
    start = int(time.time())
    end = start

    #run the loop that will move mouse but stop after timeout:
    while (end - start < timeout):
        gui.click(x=x1, y=y1)
        time.sleep(1)
        gui.click(x=x2, y=y2)
        end = int(time.time())
        print(f"Time: {end - start} / {timeout}")

if __name__ == '__main__':
    main()