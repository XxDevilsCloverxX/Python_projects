#You may present program / image

###########################################################
# A simple example to demonstrate basic usage of TKinter.
###########################################################

import tkinter as tk
import random as rand
from webbrowser import open_new    #uncomment this for the funny button

# Define a class for our application,
# which inherits from tk.Frame.
class MyApplication(tk.Frame):
    ##########################################
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack()
        # self.handle_list is to remember handles
        # to some of the things we draw,
        # so that we can erase them later.
        self.handle_list = []
        # Create all the widgets we want in
        # our window at the beginning.
        self.create_widgets()
    ##########################################
    def open_button(self):
        open_new("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        pass
    def create_widgets(self):
        # Create the widgets we want our window to have at startup.

        # First, a Canvas widget that we can draw on.
        # It will be 800 pixels wide, and 600 pixels tall.
        self.canvas = tk.Canvas(self.master, width=800, height=600, background="white")
        # This 'pack' method packs it into the top-level window.
        self.canvas.pack()

        # Create a button with label "Draw", which calls the member
        # function self.draw() below when it's activated.
        self.draw_button = tk.Button(text="Draw", command=self.draw)
        # Pack the button into the window.
        self.draw_button.pack()

        # Create another button, with label "Clear" which calls the
        # member function self.clear() when it's activated.
        self.clear_button = tk.Button(text="Clear", command=self.clear)
        self.clear_button.pack()

        #My personal button
        self.open_button = tk.Button(text="The funny button", command=self.open_button)
        self.open_button.pack()
    def draw(self):
        #############################################
        # Add your code to this method. Be sure to  #
        # store the 'handles' in the same way as    #
        # the sample code, so that the objects      #
        # will be removed when the 'clear button'   #
        # is clicked.                               #
        # You can delete all of the existing code   #
        # that's in this method right now - it is   #
        # here just as an example of drawing some   #
        # things and storing the handles.           #
        #############################################
        #use a tuple since the colors are not repeated and I don't need mutable colors
        colors = ('red', 'green', 'blue', 'purple', 'brown', 'orange', 'yellow', 'black', 'grey', 'pink')
        # The canvas methods .create_XXXXX actually return
        # an internal name (integer) corresponding to each
        # object we create, called a 'handle.
        # We will store those handles so that when the 'clear button'
        # is clicked, we can ask the canvas to remove them.

        ## Try clicking the draw button several times- it almost looks animated like stop motion.
        #Create a background:
        for width in range(1,800,8):
            #empty list for a possibly 8 sided shape
            fig = []
            div_fact = rand.randint(1,599)
            #gather 8 coords
            for i in range(16):
                fig.append(rand.randint(1, 799))
            #dont color outside the window
            for i in range(16,2):
                if (fig[i] >600):
                    fig[i] = 600
            #uncomment doodle for lines. shapes looked cooler.
            for ht in range(1,600, 6):
                #doodle=self.canvas.create_line(width%div_fact,ht%div_fact, width, ht,fill=rand.choice(colors))
                #self.handle_list.append(doodle)
                geom = self.canvas.create_polygon(fig, fill=rand.choice(colors))
                self.handle_list.append(geom)
        # Create a self-portrait
            #[x1,y1, ... xn,yn]
        xn =350
        yn =150
        face_coords = [xn,yn, xn+100, yn, xn+100, yn+60, xn+87,yn+75,(xn+(xn+100))//2+12, yn+100,(xn+(xn+100))//2-12, yn+100, xn+13,yn+75, xn, yn+60]
        face = self.canvas.create_polygon(face_coords, outline='black', width = 2, fill = '#ffe5b4')
        self.handle_list.append(face)
            #Create eyes
        for x in range(xn+12,xn+85, 55):
            eye = self.canvas.create_oval(x, yn+32, x+15, yn+45, width=2, fill='white')
            self.handle_list.append(eye)
            eye = self.canvas.create_oval((x+x+15)//2 - 4, yn+35, (x+x+15)//2 +4, yn+45, width=2, fill='yellow')
            self.handle_list.append(eye)
            eye = self.canvas.create_oval((x+x+15)//2 - 2, yn+38, (x+x+15)//2 +2, yn+42, width=1, fill='black')
            self.handle_list.append(eye)
            #create arms
            arm = self.canvas.create_rectangle(x-40, yn+150, x+60, yn+250, fill="#ffe5b4")
            self.handle_list.append(arm)
            #create ears / hair area
        sideburn = self.canvas.create_arc(xn-8,yn+15, xn+3, yn+55, start=60, extent =270, width = 2, fill='#654321')
        self.handle_list.append(sideburn)
        sideburn = self.canvas.create_arc(xn+95,yn+15, xn+105, yn+55, start=240, extent =270, width = 2, fill='#654321')
        self.handle_list.append(sideburn)
            #create hair
        yrange = [i for i in range(yn-8, yn+25)]
        for repeat in range(8): #draws 8 times so that I dont have spot hairs
            for strand in range(xn-6, xn+106,3):
                hair = self.canvas.create_arc(strand-6, rand.choice(yrange), strand+6,rand.choice(yrange), start=rand.randint(0,360), extent=rand.randint(0,360), fill='#654321')
                self.handle_list.append(hair)
        #create a whitespace polygon to "round hair"
        coords = [xn+103, yn, xn+113, yn+10, xn+113, yn-13, xn-13, yn-13, xn-13,yn+10, xn,yn-6,xn+103, yn-3]
        erase  = self.canvas.create_polygon(coords, fill=rand.choice(colors))
        self.handle_list.append(erase)
            #create a neck
        neck = self.canvas.create_rectangle((xn+(xn+100))//2-18, yn+94, (xn+(xn+100))//2+18, yn+120, width=2, fill='#ffe5b4')
        self.handle_list.append(neck)
            #create a shirt/body
        middle_of_neck= (((xn+(xn+100))//2-18) + (xn+(xn+100))//2+18)//2
        vertices = [(xn+(xn+100))//2-18, yn+110, xn-15, yn+110, xn-35, yn+150, xn, yn+165,xn+3, yn+145,xn, yn+285, xn+100,yn+285,xn+103, yn+145,xn+106,yn+165,xn+135, yn+150,xn+115, yn+110,(xn+(xn+100))//2+18, yn+110, middle_of_neck, yn+120]
        body = self.canvas.create_polygon(vertices,outline='black',width=3, fill=rand.choice(colors))
        self.handle_list.append(body)
        pants = self.canvas.create_polygon(xn, yn+285, xn, 485, middle_of_neck-5, 485,middle_of_neck,yn+320, middle_of_neck+5, 485, xn+100,485, xn+100,yn+285, fill=rand.choice(['blue', 'black']), outline='black', width=3)
        self.handle_list.append(pants)
        pants = self.canvas.create_line(xn, yn+295, xn+100, yn+295, width = 12, fill='black')
        self.handle_list.append(pants)
            #create a nose
        for i in range(-6, 6,3):
            nose= self.canvas.create_line(middle_of_neck+i, yn+50, middle_of_neck-i, yn+75)
            self.handle_list.append(nose)
            nose = self.canvas.create_oval(middle_of_neck+i, yn+80, middle_of_neck-i, yn+82, fill='black')
            self.handle_list.append(nose)
        #Create a nameplate of my name: Silas
        #https://www.tutorialspoint.com/python/tk_canvas.htm - parameters reasearch
        x0 = 100 + 85 #declaring this way allows me to move the letters easier
        y0 = 500
        x1 = 180 + 85
        y1 = 550
        border = self.canvas.create_rectangle(x0-10, y0-10, x1+315+10,y1+50, width =5, fill='white')
        self.handle_list.append(border)
        #Create a bunch of lines to design nameplate- animate by range itterations of drawing
        for n in range(x0-8, x1+315+8, 35):
            for j in range(y0-8, y1+48, 20):
                doodle = self.canvas.create_line(n, y1+50, x1+325, j, fill=rand.choice(colors))
                self.handle_list.append(doodle)
                doodle = self.canvas.create_line(n, y0-10, x0-10, j, fill=rand.choice(colors))
                self.handle_list.append(doodle)
        coord = x0,y0,x1,y1 #Coord is in fcormat (top left coord), (bottom right coord)
        #Start = angle in degrees to start arc. extent is how many degrees beyond start to draw
        constcolor = rand.choice(colors)
        s = self.canvas.create_arc(coord, start=0, extent = 280, width=5, fill=constcolor)
        self.handle_list.append(s)
        inv_coord = x0,y0+50,x1, y1+50
        s_invert = self.canvas.create_arc(inv_coord, start=180, extent = 280, width=5, fill= constcolor)
        self.handle_list.append(s_invert)
        #### S 2 will be shorter
        coord = x0+315,y0+25,x1+315, y1+25 #Coord is in format (top left coord), (bottom right coord)
        #Start = angle in degrees to start arc. extent is how many degrees beyond start to draw
        s = self.canvas.create_arc(coord, start=0, extent = 270, width=5, fill='blue')                     # Do not change
        self.handle_list.append(s)                                                                         #
        inv_coord = x0+315,y1,x1+315, y1+50                                                                # These colors
        s_invert = self.canvas.create_arc(inv_coord, start=180, extent = 270, width=5, fill= 'yellow')     #
        self.handle_list.append(s_invert)                                                                  # They make the
            #create eyes to resemble python logo, just 'reflected'                                         #
        coord = x0+360,y0+30,x1+300, y1                                                                    # Python Logo
        s_eye = self.canvas.create_oval(coord, width =2, fill = 'white')                                   #
        self.handle_list.append(s_eye)                                                                     #
        inv_coord = x0+325,y1+25,x1+265, y1+45                                                             #
        s_eye = self.canvas.create_oval(inv_coord, width = 3, fill = 'white')                              #
        self.handle_list.append(s_eye)                                                                     #
        #Create an i and l
            #creating i
        i = self.canvas.create_rectangle(x0+100, y0+45, x1+75, y1+50, width = 3, fill = rand.choice(colors))
        self.handle_list.append(i)
        i = self.canvas.create_oval(x0+100, 500, x1+75, 525, width = 3, fill = rand.choice(colors))
        self.handle_list.append(i)
            #creating l
        x0+=175; x1= x0+50
        l = self.canvas.create_rectangle(x0, y0, x1, y1+50, width = 3, fill = rand.choice(colors))
        self.handle_list.append(l)
        #create the a: use a short l for bump
        x0+=65; x1=x0+55
        constcolor = rand.choice(colors)
        a = self.canvas.create_rectangle(x0+45, y0+25, x1, y1+50, width = 3, fill = constcolor, outline=constcolor)               #leave a colors constant
        self.handle_list.append(a)
        a = self.canvas.create_oval(x0, y0+25, x1, y1+50, width = 3, fill=constcolor, outline= constcolor)
        self.handle_list.append(a)
        #create a white bubble in a
        a = self.canvas.create_oval(x0+10, y0+40, x1-15, y1+35, width = 3, fill='white')
        self.handle_list.append(a)
    ##############################################
    def clear(self):
        # To clear the things we drew in the 'draw'
        # function, we just ask the canvas to delete them,
        # one at a time, by their handles.
        # You should not need to modify anythingin this method.
        while len(self.handle_list)>0:
            h = self.handle_list.pop()
            self.canvas.delete(h)


########################################
# Do not change anything below here!   #
########################################
# Instantiate the Tk class.
# This should only ever be done once in a program.
# Think of it as 'firing up' the library, getting it ready to do stuff.
root = tk.Tk()

# Create an instance of the MyApplication class we defined above.
app = MyApplication(root)

# Pass flow control over to the Tkinter library, so it can do things
# like wait for keyboard and mouse events, redraw the window when needed,...
# One of the things it will do is watch for buttons we created and invoke
# the 'callback functions' we gave them. It will run indefinitely,
# until the operating system sends it a 'quit' command (e.g.,
# if we close the window).
app.mainloop()
