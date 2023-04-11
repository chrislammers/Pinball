"""
author: Chris Lammers
"""

import pygame
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import ode
import random
from datetime import datetime


# colors:
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# constants:
disk1_mass = 1
disk2_mass = 1
dim = [20, 20]
window = [640, 640]
# Distance = dim[0]
field = [-(dim[0]/2),(dim[1]/2)]
# field = [-10,10]
scale = window[0]/dim[0]

# field = [-10,10]
print(scale)

# clock
clock = pygame.time.Clock()

class Brick(pygame.sprite.Sprite):
    # Bricks are the makeup of the shape of the arena
    # They only have a shape (4 x,y points) and a rotation orientation.
    # 0 velocity, so 0 momentum, 0, angular vel, 0 angular momentum.
    # effectively inf mass, so that collisions feel like hitting a wall

    def __init__(self, name, w, h, x=0, y=0, color=GREEN):
        pygame.sprite.Sprite.__init__(self)

        self.w = w*scale
        self.h = h*scale
        self.image = pygame.Surface([self.w,self.h])
        # self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.color = color
        pygame.draw.rect(self.image, color, self.rect)
        self.name = name
        self.state = np.array([x,y])  #this will be expanded when rotation is involved
        self.image_rot = self.image

    # def to_screen(self):
    #     # return [int(pos[0]), int(pos[1])]
    #     return [int((self.state[0] - self.radius + dim[0]//2)*scale), int((-self.state[1] - self.radius +  dim[1]//2)*scale)]
    def rotate(self, angle):
        self.image_rot = pygame.transform.rotate(self.image, 90)
        self.rect_rot = self.image_rot.get_rect()
        pygame.draw.rect(self.image_rot, WHITE, self.rect_rot)

    def to_screen(self):
        return [int((self.state[0]  + dim[0]//2)*scale), int((-self.state[1]  + dim[1]//2)*scale)]

    def get_center(self, p):
        return p[0]-self.w//2, p[1]-self.h//2

    def get_mass(self):
        return 1000000

    def get_vel(self):
        return np.zeros(2)

class Disk(pygame.sprite.Sprite):

    def __init__(self, name, mass, color=BLACK, radius=1):
        pygame.sprite.Sprite.__init__(self)

        # no image file stuff (yet)...

        scaled = radius*scale
        self.image = pygame.Surface([scaled*2, scaled*2])
        self.image.set_colorkey(BLACK)
        # self.image.fill(BLACK)

        pygame.draw.circle(self.image, color, (scaled, scaled), scaled, 3)

        self.rect = self.image.get_rect()
        self.state = np.zeros(4)
        self.mass = mass
        self.radius = radius
        self.name = name
        self.distances = []
        self.tol_distance = 0.001
        self.grav = [0, -4]

        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')

    def set_pos(self, pos):
        # self.pos = np.array(pos)
        self.state[0:2] = pos

    def set_vel(self, vel):
        self.state[2:4] = vel

    def to_screen(self):
        # return [int(pos[0]), int(pos[1])]
        return [int((self.state[0] - self.radius + dim[0]//2)*scale), int((-self.state[1] - self.radius +  dim[1]//2)*scale)]

    def get_center(self, p):
        return p[0]-self.radius, p[1]-self.radius

    def is_brick_coll(self, other):

        pass

    def is_ball_coll(self, state, other):
        # its a disk. The collision point will always be 1 radius away from the center
        # no inertia, to the velocity of the ball is uniform across the surface
        #  no orientation here.
        col = False

        rA = self.radius
        rB = other.radius
        xA = self.state[:2]
        xB = other.state[:2]


        # pA and pB are the points on the ball closest to each other.
        # at any time, the line from xA to xB intersects pA and pB
        # the angle is the same
        # angle = np.dot(xA,xB)
        # print("dot of (xA,xB): ",angle)

        # n = (xA-xB)/abs(xA-xB)
        # print("xA:",xA,"xB:",xB, "(xA-xB)/abs(xA-xB): ",n)



        # distance from center to center:
        d = (xA-xB)
        # print(d)
        # Distance from edge to edge
        # r is the length of the normal between the objects.
        # If r > 0, they aren't touching
        r_c = np.linalg.norm(d)
        r = r_c-rA-rB

        u = d/r_c
        # print(u)

        # return r<0

        if r<0:
            # print("contact")
            vA = self.state[-2:]
            vB = other.state[-2:]
            vAB = vA-vB

            che = np.dot(vAB, u)
            # print(vAB, "dot", n, " = ",che)
            if che > 0:
                pass
                # print("moving away")
            elif che < 0:
                # print("Collision!")
                col = True
            else:
                pass
                # print("resting contact")



        return col

    def ball_coll_resp(self, state, other, t, dt):
        # return 0
        # change is momentum for self is the opposite of other
        rA = self.radius
        rB = other.radius
        xA = self.state[:2]
        xB = other.state[:2]

        d = xA-xB
        # n = (xA-xB)/abs(xA-xB)
        # n = np.linalg.norm(xA-xB)

        r_c = np.linalg.norm(d)
        r = r_c-rA-rB

        n = d/r_c

        vA = self.state[-2:]
        vB = other.state[-2:]
        vAB = vA-vB

        # -2 here becasue inelastic
        j = (-2*np.dot(vAB, n))/(1/self.mass + 1/other.mass)
        # print("j: ",j)

        # print(j)
        v2A = vA + (j*n)/self.mass
        v2B = vB + (-j*n)/other.mass
        # print("vA1:",vA, "vA2:", v2A)
        # print("vB1:",vB, "vB2:", v2B)

        self.state[-2:] = v2A
        other.state[-2:] = v2B
        # new_state = np.array([self.state[0], self.state[1], v2A[0], v2A[1]])
        self.solver.set_initial_value(self.state, t+dt)
        other.solver.set_initial_value(other.state, t+dt)
        pass


    def is_collision(self, state):
        return state[0]<=field[0]+self.radius or state[0]>=field[1]-self.radius or state[1]<=field[0]+self.radius or state[1]>=field[1]-self.radius

    def respond_to_collision(self, state, t, dt):
        # return [0,0], t

        # ball to wall collisions:
        #  ball can contact 4 walls:
        #  state[0]=10, state[0]=-10, state[1]=10, state[1]=-10
        #  abs(state[0])=10, or abs(state[1])=10

        side = False
        if abs(state[0]) > self.tol_distance+10-self.radius:
            # print("sidewall")
            side = True

        beg = t-dt
        end = t
        mid = (beg+end)/2
        # print("between",beg,"and", end)
        # print(state[0],self.tol_distance)
        # print("--while loop--")
        iterations = 0
        while abs(state[0])-10+self.radius > self.tol_distance and abs(state[1])-10+self.radius > self.tol_distance:
            # print("between",beg,"and", end)
            mid = (beg+end)/2
            state = self.solver.integrate(mid)
            if self.is_collision(state):
                end = mid
                # t = mid
            else:
                beg = mid

            iterations+=1
            if iterations > 150:
                break

        # print("collision at", mid)
        # find out if the collision is on the walls or roof.
        if not side:
            # print("Ceiling or floor bounce. New state:")
            # print([state[0], state[1], state[2], -1*state[3]])
            return np.array([state[0], state[1], state[2], -1*state[3]]), mid
        else:
            # print("Wall bounce. New state:")
            # print([state[0], state[1], -1*state[2], state[3]])
            return np.array([state[0], state[1], -1*state[2], state[3]]), mid

    def f(self, t, state, arg1):
        dx = state[2]
        dy = state[3]
        dvx = self.grav[0]
        dvy = self.grav[1]
        return [dx, dy, dvx, dvy]

    def update1(self, objects, dt):
        # force = np.array([0,0])

        for o in objects:
            if o != self.name:
                other = objects[o]
                self.solver.set_f_params(other)

                # print("time:",self.solver.t)
                # print(self.name)
                new_state = self.solver.integrate(self.solver.t+dt)
                # print("state: [x,y,dx,dy]",new_state)
                # self.solver.integrate(self.solver.t+dt)
                # Collision detection
                if not self.is_collision(new_state):
                    self.state = new_state

                    self.solver.t += dt
                    # print(self.pos)
                else:
                    # print("\nCollision!\n\n")
                    state_after_collision, collision_time = self.respond_to_collision(new_state, self.solver.t, dt)
                    self.state = state_after_collision

                    self.solver.t = collision_time


                if self.is_ball_coll(self.state, other):
                    # print("ball collision between", self.name, "and", other.name)
                    self.ball_coll_resp(self.state, other, self.solver.t, dt)
                # self.solver.y - [self.pos[0], self.pos[1], self.vel[0], self.vel[1]]
                self.solver.set_initial_value(self.state, self.solver.t)

                # d = (other.pos - self.pos)
                # r = np.linalg.norm(d) - 2*self.radius

                # print(r)

class Universe:
    def __init__(self):
        self.w, self.h = dim[0], dim[1]
        self.objects_dict = {}
        self.walls_dict = {}
        self.objects = pygame.sprite.Group()
        self.dt = 0.033

    # when something is added, we need to specify if it's part of the walls
    # collides = True means the object is a ball or spring.
    #       these objects interact with each other, and the environment.
    # collides = False means that it's a wall/environment
    #       these object only get interacted with. They don't collide with each
    def add_body(self, body, collides):
        if collides:
            self.objects_dict[body.name] = body
            self.objects.add(body)
        else:
            p = body.to_screen()
            body.rect.x, body.rect.y = body.get_center(p)
            # pygame.draw.rect(self.image, color, self.rect)
            self.walls_dict[body.name] = body
            self.objects.add(body)

    def to_screen(self, pos):
        # return [int(pos[0]), int(pos[1])]
        return [int((pos[0] + dim[0]//2)*window[0]//self.w), int((-pos[1] + dim[1]//2)*window[1]//self.h)]

    def update(self):
        for o in self.objects_dict:
            # Compute positions for screen
            # print(self, self.objects_dict)
            obj = self.objects_dict[o]
            obj.update1(self.objects_dict, self.dt)
            p = obj.to_screen()

            if False: # Set this to True to print the following values
                print ('Name', obj.name)
                print ('Position in simulation space', obj.state[:2])
                print ('Position on screen', p)
                print ('Velocity in space', obj.state[-2:])

            # Update sprite locations
            obj.rect.x, obj.rect.y = obj.get_center(p)
        self.objects.update()
        # print("")
        # print(self.objects)

    def draw(self, screen):
        self.objects.draw(screen)


def main():

    print ('Press q to quit')

    # Initializing pygame
    pygame.init()
    win_width = window[0]
    win_height = window[0]
    screen = pygame.display.set_mode((win_width, win_height))  # Top left corner is (0,0)
    pygame.display.set_caption('Disks')

    # Create a Universe object, which will hold our heavenly bodies (planets, stars, disk2s, etc.)
    universe = Universe()

    disk1 = Disk('disk1', disk1_mass, RED, radius=1)
    disk1.set_pos([-5, 0.1])
    # disk2 = Disk('disk2', disk2_mass, WHITE, radius=1)
    # disk2.set_pos([0, 0])
    disk3 = Disk('disk3', 4, WHITE, radius=2)
    disk3.set_pos([0, 8])
    # disk4 = Disk('disk4', disk2_mass, WHITE, radius=1)
    # disk4.set_pos([0, 1])


    disk1.set_vel([4, 5])
    # disk2.set_vel([-2, 0])
    disk3.set_vel([3, 3])
    # disk4.set_vel([-2, 0])
    # rect1 = Brick('b1', 6, 4, 0, 0)
    # rect1.rotate(50)
    # universe.add_body(rect1, False)

    # cur_time = clock.get_time()

    # disk2.solver.set_initial_value(disk2.state, 0.0)

    # print([disk2.pos[0], disk2.pos[1], disk2.vel[0], disk2.vel[1]])
    # print(disk2.solver.y)
    disk1.solver.set_initial_value(disk1.state, 0.0)
    disk3.solver.set_initial_value(disk3.state, 0.0)
    # disk4.solver.set_initial_value(disk4.state, 0.0)



    universe.add_body(disk1, True)
    # universe.add_body(disk2)
    universe.add_body(disk3, True)
    # universe.add_body(disk4)

    total_frames = 30000
    iter_per_frame = 1

    frame = 0

    # clock.tick(60)
    while frame < total_frames:
        clock.tick(60)
        if False:
            print ('Frame number', frame)

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit(0)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            pygame.quit()
            sys.exit(0)
        else:
            pass

        universe.update()
        if frame % iter_per_frame == 0:
            screen.fill(BLACK) # clear the background
            universe.draw(screen)
            pygame.display.flip()
        frame += 1


    pygame.quit()



if __name__ == '__main__':
    main()
