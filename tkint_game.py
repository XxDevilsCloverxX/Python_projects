
from tkinter import *
from math import sin,cos,pi,atan2

################################################################
# Delay between successive screen renderings, in milliseconds.
REFRESH_DELAY = 25

# Speed at which the ship rotates when a Left/Right arrow is pressed, in radians per millisecond.
SHIP_ROT_SPEED=(2.0*pi/1000)
# Maximum speed of the ship, in pixels per millisecond.
MAX_VEL=0.4
# Rate of acceleration of the ship when UP arrow is pressed.
SHIP_DS=0.01

# Missile speed, in pixels per millisecond.
MISSILE_SPEED=.2
# Missile live time, in milliseconds.
MISSILE_LIVE_TIME=1000
INITIAL_MAX_ROCK_SPEED = MISSILE_SPEED
# Max angular velocity of the SpaceRocks, in radians per millisecond.
MAXAV = (2*pi/1000)
# How many points a big rock is worth.
BIGROCK_POINTS=100
# Max number of missiles that can be on the screen simultaneously.
MAX_MISSILES=6
#################################################################


def pip(poly_verts, x, y):
    #initialize starting angle to 0
    theta = 0
    #step through every x in the list poly_verts
    for i in range(0, len(poly_verts), 2):
        #use negative indexing to connect the points in the polygon properly
        coords = (poly_verts[i-2], poly_verts[i-1], x, y, poly_verts[i], poly_verts[i+1])
        theta += signed_angle(*coords)
    #condition where theta needs to be really close to 0 to return false (outside a polygon)
    return(abs(theta) >= (0.5))

def signed_angle(x1,y1,x2,y2,x3,y3):
    #Create Vectors from x2,y2 to other points
    v1 = (x1-x2, y1-y2)
    v2 = (x3-x2, y3-y2)
    #y value goes first in the function to get the angle between the vectors
    theta = atan2(v2[1], v2[0]) - atan2(v1[1], v1[0])
    #normalize the resulting angle to (-pi, pi]
    if(theta > pi):
        theta-=2*pi
    elif(theta <= -1*pi):
        theta +=2*pi
    return theta

########################################################
class LCRand:
    ####################################################
    def __init__(self, seed):
        self.x = seed

    ####################################################
    def step(self):
        self.x = (1664525*self.x + 1013904223) % (2**32)
        return self.x

    ####################################################
    def rand01(self):
        n = self.step()
        return n/(2**32)

    ####################################################
    def rand(self, a, b):
        y = self.rand01()
        return a + y*(b-a)

########################################################
class Polygon:
    ####################################################
    def __init__(self, cx, cy, polarverts):
        # 'polarverts' is a list of vertices in polar coordinates:
        # [ [r1, theta1], [r2, theta2], ..., [r_n, theta_n] ]
        self.polarverts = polarverts
        # The default location of the polygon is the screen coordinate (0,0)
        self.x = cx
        self.y = cy
        self.hasBeenDrawn = False
        self.handle = -1
        self.rotation = 0
        self.velocity = [0,0]
        self.ang_velocity = 0
        self.timeAlive = 0 # How long this Polygon has been around?
        self.hide = False  # See the 'redraw' method below.

    ####################################################
    def setAngVelocity(self, ang_vel):
        self.ang_velocity = ang_vel

    ####################################################
    def setRotation(self, angle):
        self.rotation = angle

    ####################################################
    def getRotation(self):
        return self.rotation

    ####################################################
    def setCoords(self, x, y):
        # Set the location coordinates of this polygon.
        self.x = x
        self.y = y

    ####################################################
    def setVelocity(self, vx, vy):
        self.velocity = [vx, vy]

    ####################################################
    def updatePosition(self, dt, screenW, screenH):
        # Update the position of this Polygon, based on its previous position,
        # velocity, angular velocity, and elapsed time 'dt'.
        self.x = self.x + dt*self.velocity[0]
        self.y = self.y + dt*self.velocity[1]
        self.x = self.x % screenW
        self.y = self.y % screenH
        self.rotation = self.rotation + dt*self.ang_velocity
        self.timeAlive += dt


    ####################################################
    def getCoordList(self):
        # Generate and return a list containing the Cartesian coordinates of the
        # vertices of this polygon, in the format [x0, y0, x1, y1, ..., xk, yk].
        cclist = [] # create an empty list
        for V in self.polarverts:
            r = V[0]
            theta = V[1]
            vx = self.x + r*cos(theta + self.rotation)
            vy = self.y + r*sin(theta + self.rotation)
            cclist.append(vx)
            cclist.append(vy)
        return cclist

    ####################################################
    def remove(self, canvas):
        if self.hasBeenDrawn:
            canvas.delete(self.handle)
        self.hasBeenDrawn = False

    ####################################################
    def redraw(self, canvas):
        # If self.hide is True, then don't actually draw it.
        if self.hide:
            return

        self.remove(canvas) # First remove the previously created widget, if any.
        coords = self.getCoordList()
        self.handle = canvas.create_polygon(coords, outline="white")
        self.hasBeenDrawn = True

    ####################################################
    def pointInPoly(self, x, y):
        verts = self.getCoordList()
        return pip(verts, x, y)

#######################################################################
class SpaceRock(Polygon):
    ####################################################
    def __init__(self, x, y, scale):
        verts = [ [20*scale, 0], [6*scale, 0.8], [12*scale, 1.4],
                  [24*scale, 2.2], [15*scale, 2.7], [18*scale,3.4],
                  [24*scale ,3.9], [10*scale ,4.5], [27*scale,5.3]]
        self.scale = scale
        Polygon.__init__(self, x, y, verts)
        self.value = 0 # How many points is this SpaceRock worth?

    ####################################################
    def getScale(self):
        return self.scale

    ####################################################
    def setValue(self, points):
        self.value = points

    ####################################################
    def getValue(self):
        return self.value


#######################################################################
class SpaceShip(Polygon):
    ####################################################
    def __init__(self, x, y):
        verts = [[14,0], [10, 2*pi/3], [0, 3*pi/2], [10, 4*pi/3]]
        Polygon.__init__(self, x, y, verts)

    ####################################################
    def changeVelocity(self, ds):
        # Increase velocity by amount ds in the direction self.rotation,
        # ensuring that the maximum velocity is not exceeded.
        dx = ds*cos(self.rotation)
        dy = ds*sin(self.rotation)
        self.velocity[0] += dx
        self.velocity[1] += dy
        # Here, we are going to slightly bend the laws of physics,
        # to impose a maximum velocity on the ship. If the max velocity
        # has been exceeded, we will allow a change of direction only.
        v = (self.velocity[0]**2 + self.velocity[1]**2)**(1/2)
        if v > MAX_VEL:
            self.velocity[0] *= MAX_VEL/v
            self.velocity[1] *= MAX_VEL/v

    ####################################################
    def changeAngle(self, dtheta):
        self.rotation += dtheta


#######################################################################
class Missile(Polygon):
    ####################################################
    def __init__(self, x, y, shipvel, angle):
        Polygon.__init__(self, x, y, [[1.5,pi/2.0], [1.5,3*pi/2.0],[1.5,5*pi/2.0],[1.5,7*pi/2.0]])
        self.livetime = MISSILE_LIVE_TIME
        # Determine the velocity; since the ship may be in motion, that velocity
        # is added to the missile's velocity.
        self.velocity[0] = shipvel[0]+MISSILE_SPEED*cos(angle)
        self.velocity[1] = shipvel[1]+MISSILE_SPEED*sin(angle)

    ####################################################
    def isAlive(self):
        if (self.timeAlive <= self.livetime):
            return True
        return False

#######################################################################
class Controller:
    ####################################################
    def __init__(self , screenWidth , screenHeight ):
        master = Tk()
        self.width = screenWidth
        self.height = screenHeight
        self.canvas = Canvas (master, width=self.width, height=self.height, background ="black")
        self.scoreItem = self.canvas.create_text(10,10,anchor="nw",fill="white",text="Score: 0")
        self.hiscoreItem = self.canvas.create_text(self.width-140,10,anchor="nw",fill="white",text="Hi: 0")

        self.canvas.pack()
        # Create a SpaceShip , and store it as an attribute.
        self.Ship = SpaceShip(self.width/2, self. height/2)
        self.Ship.hide = True # Hide it until the region is clear.
        # Create a Psuedo Random Number Generator:
        self.Prng = LCRand(1)
        self.RockList = []

        # List of active missiles; empty at the start.
        self.missiles = []
        # This will be a list of keys that the user is currently pressing:
        self.keysdown = []
        # Tell TKinter that we want to know when a key has been pressed:
        self.canvas.bind_all("<KeyPress>", self.keyPressed)
        # or released:
        self.canvas.bind_all("<KeyRelease>", self.keyReleased)

        self.GOitem = -1 # GameOver?
        self.hiscore = 0 # This will be read from file in the newGame() method.
        self.scoreNeedsRedraw = True
        self.newGame()

    ####################################################
    def newGame(self):
        ###################################
        # Read the high score from file.  #
        ###################################
        try:
            f = open("hiscore.txt", "r")
            self.hiscore = int(f.read())
            f.close()
        except:
            self.hiscore = 0
        ############################################
        # Reset game parameters to initial values. #
        ############################################
        self.gameLevel = 0
        self.score = 0
        self.remainingLives = 2
        ########################################################
        # If there's a `Game Over' label on screen, remove it. #
        ########################################################
        if (self.GOitem > 0):
            self.canvas.delete(self.GOitem)
            self.GOitem = -1

        self.newBoard()
        self.waitingForNewBoard = False

        #######################################################################
        # Redrawing the score on screen is a relatively expensive operation.  #
        # So we will only do it when necessary, and this attribute will       #
        # determine whether or not the score needs to be redrawn.             #
        #######################################################################
        self.scoreNeedsRedraw = True

        ############################################
        # Make sure the ship is initially hidden,  #
        # and try to spawn it after some delay.    #
        ############################################
        self.Ship.hide = True
        self.canvas.after(1000, self.trySpawnShip)

    ####################################################
    def gameOver(self):
        self.GOitem = self.canvas.create_text(self.width/2, self.height/2, fill="white",text="Game Over")
        # Save the current High Score to file.
        f = open("hiscore.txt", "w")
        f.write(str(self.hiscore))
        f.close()
        # A brief pause, start a new game.
        self.canvas.after(3000, self.newGame)


    ####################################################
    def setRockVelocity(self, R):
        # Randomly set the velocity and angular velocity of a SpaceRock,
        # with higher velocities possible for higher self.gameLevel's.
        direction = self.Prng.rand(0, 2*pi)
        speed = self.Prng.rand(0, 1)*INITIAL_MAX_ROCK_SPEED*(1.0 + self.gameLevel/3.0)
        vx = speed*cos(direction)
        vy = speed*sin(direction)
        R.setVelocity(vx, vy)
        dtheta = self.Prng.rand(-MAXAV, MAXAV);
        R.setAngVelocity(dtheta)

    ####################################################
    def newBoard(self):
        # Make sure RockList is empty:
        for R in self.RockList:
            R.remove(self.canvas)
        self.RockList = []

        num_rocks = 4 + 2*self.gameLevel
        for i in range(num_rocks):
            x = self.Prng.rand(0, self.width)
            y = self.Prng.rand(0, self.height)
            R = SpaceRock(x,y,1.0)
            theta = self.Prng.rand(0, 2*pi)
            R.setRotation(theta)
            # Set the initial velocity of this SpaceRock:
            self.setRockVelocity(R)

            R.setValue(BIGROCK_POINTS)
            self.RockList.append(R)

    ####################################################
    def trySpawnShip(self):
        # Attempt to spawn the ship, but don't do it if there are SpaceRocks
        # too close that would immediately obliterate the ship, because that's
        # just not fair. We do this by creating a hidden square in the center
        # and applying the pointInPoly method to check and see if the centers
        # of any SpaceRocks are contained in it.
        d = self.width/6
        # Create a square with diagonal 2d centered on screen (but not drawn)
        Square = Polygon(self.width//2, self.height//2, [ [d, pi/4], [d, 3*pi/4], [d, 5*pi/4],[d, 7*pi/4]])
        squareIsEmpty = True
        for R in self.RockList:
            if Square.pointInPoly(R.x, R.y):
                squareIsEmpty = False
                break
        if squareIsEmpty==True:
            self.Ship.setCoords(self.width//2, self.height//2)
            self.Ship.setVelocity(0,0)
            self.Ship.hide = False
        else:
            # We couldn't spawn the ship; so we will try again after a delay.
            self.canvas.after(REFRESH_DELAY*20, self.trySpawnShip)

    ####################################################
    def keyPressed(self, event):
        self.keysdown.append(event.keysym)
        return "break"

    ####################################################
    def keyReleased(self, event):
        key = event.keysym
        # Check to see if it's in our list before we remove it,
        # so we don't throw an exception (error message).
        while key in self.keysdown:
            self.keysdown.remove(key)
        return "break"

    ####################################################
    def check_keys(self):
        for k in self.keysdown:
            if (k=="Left"):
                self.Ship.changeAngle(-SHIP_ROT_SPEED*REFRESH_DELAY)
            if (k=="Right"):
                self.Ship.changeAngle(SHIP_ROT_SPEED*REFRESH_DELAY)
            if (k=="Up"):
                self.Ship.changeVelocity(SHIP_DS)
            if (k=="space"):
                # Launch a missile; but only do this if we have less than
                # MAX_MISSILES already launched, and the ship is visible.
                if (len(self.missiles)<MAX_MISSILES) and (self.Ship.hide==False):
                    M = Missile(self.Ship.x, self.Ship.y, self.Ship.velocity, self.Ship.rotation)
                    self.missiles.append(M)
                    # Remove this key from the list, so the user needs
                    # to press the spacebar once for each missile fired; e.g.,
                    # no automatic fire.
                    self.keysdown.remove(k)

    ####################################################
    def check_missiles(self):
        # Check for collisions between missles and rocks.
        for M in self.missiles:
            j = 0
            hit = False
            while j<len(self.RockList) and hit==False:
                R = self.RockList[j]
                if R.pointInPoly(M.x, M.y):
                    # Check the size of the rock by looking at its 'scale';
                    # if it's big, break it into several smaller ones.
                    if (R.getScale()>0.5):
                        for i in range(3):
                            newRock = SpaceRock(R.x, R.y,R.getScale()*0.7)
                            newRock.setValue(2*R.getValue())
                            self.setRockVelocity(newRock)
                            self.RockList.append(newRock)
                    self.scoreNeedsRedraw = True
                    self.score += R.getValue()
                    R.remove(self.canvas)
                    self.RockList.remove(R)
                    M.remove(self.canvas)
                    self.missiles.remove(M)
                    hit = True
                j += 1

    ####################################################
    def check_ship_collision(self):
        if self.Ship.hide == False:
            clist = self.Ship.getCoordList()
            N = len(clist)//2
            for i in range(N):
                x = clist[2*i]
                y = clist[2*i+1]
                for R in self.RockList:
                    if R.pointInPoly(x,y):
                        # Collision! Undraw the ship
                        if (self.Ship.hide == False):
                            self.Ship.remove(self.canvas)
                            self.Ship.hide = True
                            if (self.remainingLives > 0):
                                self.remainingLives -= 1
                                self.canvas.after(1000, self.trySpawnShip)
                                break
                            else:
                                # Game over!
                                self.gameOver()

    #######################################################
    def update_missile_positions(self):
        for M in self.missiles:
            if M.isAlive():
                M.updatePosition(REFRESH_DELAY, self.width, self.height)
                M.redraw(self.canvas)
            else:
                M.remove(self.canvas) # Remove it from the screen
                self.missiles.remove(M) # Remove it from the list.

    ####################################################
    def refresh(self):
        ################################################
        # Check to see which keys are currently being  #
        # pressed, and take any necessary action(s).   #
        ################################################
        self.check_keys()

        ###################################################
        # Check for collisions between missiles and rocks #
        ###################################################
        self.check_missiles()

        ########################################
        # Update the high score, if necessary. #
        ########################################
        if self.score > self.hiscore:
            self.hiscore = self.score

        #######################################################
        # Check for collision between the ship and the rocks, #
        # if the ship is not hidden.                          #
        #######################################################
        self.check_ship_collision()

        ##############################################################
        # Update the position of each active missile and redraw.    #
        # If there are any expired ones, remove them from the list.  #
        ##############################################################
        self.update_missile_positions()

        ###############################################################
        # Update the ship's position and redraw it, if there is one.  #
        ###############################################################
        if self.Ship.hide == False:
            self.Ship.updatePosition(REFRESH_DELAY, self.width, self.height)
            self.Ship.redraw(self.canvas)

        #################################################
        # Update the position of each rock and redraw.  #
        #################################################
        for R in self.RockList:
            R.updatePosition(REFRESH_DELAY, self.width, self.height)
            R.redraw(self.canvas)

        #################################################
        # If there are no SpaceRocks, we'll draw a new  #
        # board, after a brief delay.                   #
        #################################################
        if (len(self.RockList)==0):
            if self.waitingForNewBoard == False:
                self.waitingForNewBoard = True
                self.gameLevel += 1
                self.canvas.after(1500, self.newBoard)
        else:
            self.waitingForNewBoard = False

        ########################################################################
        # Update the score on screen; only do it if the score changed, though, #
        # because it's a relatively expensive operation.                       #
        ########################################################################
        if (self.scoreNeedsRedraw):
            self.canvas.itemconfigure(self.scoreItem, text="Score:"+str(self.score))
            self.canvas.itemconfigure(self.hiscoreItem, text="Hi:"+str(self.hiscore))
            self.scoreNeedsRedraw = False

        ####################################################################
        # Tell TKinter to call this function again, after a brief delay.   #
        ####################################################################
        self.canvas.after(REFRESH_DELAY, self.refresh)



"""
  Here is the starting execution point (the 'main').
  There isn't much to do except
  (i) instantiate a Controller,
  (ii) start off the refreshing process,
  (iii) hand off execution to TKinter.
"""
C = Controller(800, 600)
C.refresh()
mainloop()
