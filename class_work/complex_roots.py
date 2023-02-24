import tkinter as tk
import math

#Newton function that will find convergence, of any, of xn=0
def Newton(Function, Derivative, z0, epsilon):
    iteration=0
    xn = z0 #initial guess
    #itterate up to 35 times.
    while (iteration<35):
        y = Function(xn)
        dy = Derivative(xn)
        #conditions to exit
        if abs(y) < epsilon:
            return xn
        if abs(dy)<epsilon:
            return None
        #compute the next term
        xn = xn - y/dy
        iteration += 1
    if abs(Function(xn))<epsilon:
        return xn
    return None

def colorFromRGB(r, g, b):
    # R, G, B are floating point numbers between 0 and 1 describing
    # the intensity level of the Red, Green and Blue components.
    X = [int(r*255), int(g*255), int(b*255)]
    for i in range(3):
        X[i] = min(max(X[i],0), 255)
    color = "#%02x%02x%02x"%(X[0],X[1],X[2])
    return color

def color_from_comp(z):
    x = z.real*2.7
    y = z.imag*2.7
    r = 0.5*(1.0 + math.tanh(x))
    g = 0.5*(1.0 + math.sin(x*y))
    b = 0.5*(1.0 + math.cos(x+y))
    return colorFromRGB(r, g, b)

#Functions and their derivatives#
def G(z):
    return z**9 +1

def dG(z):
    return 9*z**8

#############################################################
class MyApplication(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master
        self.pack()

        self.canvas_width = 600
        self.canvas_height = 480
        self.set_region(0+0j, 4)
        self.create_widgets()
    ############################################################
    def set_region(self, center, region_width):
        # This sets the region of the complex plane to be displayed on the canvas,
        # by specifying the center and width of that region.
        # The height of the complex region is then determined
        # from the aspect ratio of the canvas.
        # Using those, we set the attributes:
        # self.top_left : complex number at the center of the top-left pixel.
        # self.bot_right: complex number at the center of the bottom-right pixel.

        region_height = self.canvas_height * region_width/self.canvas_width
        # Complex point in the center of the top-left pixel:
        tlx = center.real - region_width/2
        tly = center.imag + region_height/2
        self.top_left = tlx + 1j*tly
        # Complex point in the center of the bottom-right pixel:
        brx = center.real + region_width/2
        bry = center.imag - region_height/2
        self.bot_right = brx + 1j*bry
    ###################################################################
    def update_screen(self):
        self.master.update_idletasks()
        self.master.update()
    ###################################################################
    def create_widgets(self):
        # First, a Canvas widget that we can draw on. It will be self.screen_width pixels wide,
        # and self.screen_height pixels tall.
        self.canvas = tk.Canvas(self.master,
                                width=self.canvas_width,
                                height=self.canvas_height,
                                background="white")
        # This 'pack' method packs it into the top-level window.
        self.canvas.pack()
    ###############################################################################
    def draw_pixel(self, x, y, C):
        self.canvas.create_rectangle(x-0.5, y-0.5, x+0.5, y+0.5, fill=C, outline="")
    ###############################################################################
    def draw_newton_plot(self):
        for i in range(self.canvas_width):
            for j in range(self.canvas_height):
                z = self.center_of_pixel(i, j)
                root = Newton(G, dG, z, 0.00001)
                if type(root) is complex:
                    color = color_from_comp(root)
                else:
                    color="black"
                self.draw_pixel(i, j, color)
            if i%20 == 0:
                self.update_screen()

    ###################################################################
    def center_of_pixel(self, x, y):
        # Return the complex number corresponding to the center of the
        # given pixel.
        real = self.top_left.real+ x*((self.bot_right.real - self.top_left.real)/(self.canvas_width - 1))
        imag = self.top_left.imag+ y*((self.bot_right.imag - self.top_left.imag)/(self.canvas_height - 1))
        return real + (1j)*imag


##############################################################


root = tk.Tk()
app = MyApplication(master=root)
app.draw_newton_plot()
app.mainloop()
