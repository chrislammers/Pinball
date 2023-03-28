"""
author: Chris Lammers
"""

# even more object-oriented.

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

# Class Structure:
# everything is a Body
#

clock = pygame.time.Clock()

window = [640, 640]

class Poly(pygame.sprite.Sprite):
    # Poly's are the makeup of the shape of the arena
    # They only have a shape (4 x,y points) and a rotation orientation.
    # 0 velocity, so 0 momentum, 0, angular vel, 0 angular momentum.
    # effectively inf mass, so that collisions feel like hitting a wall

    def __init__(self, name, points, color=GREEN):
        pygame.sprite.Sprite.__init__(self)


        self.image = pygame.Surface(window)
        # self.image.fill(BLUE)
        self.image.set_colorkey(BLACK)
        self.color = color
        # pygame.draw.rect(self.image, color, self.rect)
        self.rect = pygame.draw.polygon(self.image, color, points)
        self.rect.x = 0
        self.rect.y = 0
        self.name = name
        self.points = points

    def get_mass(self):
        return 1000000

    def get_vel(self):
        return np.zeros(2)




class Ball(pygame.sprite.Sprite):

    def __init__(self, name, mass, color=BLACK, radius=1):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([radius*2, radius*2])
        self.image.set_colorkey(BLACK)
        # self.image.fill(BLACK)

        pygame.draw.circle(self.image, color, (radius, radius), radius, 3)
        self.rect = self.image.get_rect()
        self.state = np.zeros(4)
        self.state[:2] = window
        self.mass = mass
        self.radius = radius
        self.name = name
        self.distances = []
        self.tol_distance = 0.001
        self.grav = [0, 0]

        self.solver = ode(self.f)
        self.solver.set_integrator('dop853')
        self.solver.set_initial_value(self.state)

    def set_pos(self, pos):
        # self.pos = np.array(pos)
        self.state[0:2] = pos
        self.solver.set_initial_value(self.state)

    def set_vel(self, vel):
        self.state[2:4] = vel
        self.solver.set_initial_value(self.state)

    # def to_screen(self):
    #     # return [int(pos[0]), int(pos[1])]
    #     return [int((self.state[0] - self.radius + dim[0]//2)*scale), int((-self.state[1] - self.radius +  dim[1]//2)*scale)]

    # def get_center(self, p):
    #     return p[0]-self.radius, p[1]-self.radius

    def is_coll(self, other):
        # ball can collide with a walls' vertex or edge.
        # the vertex is easy enough. it's stored in other.points
        # print(other.points)


        # vertex collision detection:
        for point in other.points:
            # print(other.name, self.state[:2], point)
            d = point - self.state[:2]
            # print(d, d[0]**2 + d[1]**2 <= self.radius**2)
            if (d[0]**2 + d[1]**2 <= self.radius**2):
                return True


        pass


    def f(self, t, state):
        dx = state[2]
        dy = state[3]
        dvx = self.grav[0]
        dvy = self.grav[1]
        return [dx, dy, dvx, dvy]

    def update1(self, objects, walls, dt):
        # force = np.array([0,0])
        # print("updating,", self.name)

        pass

        for b in objects:
            new_state = self.solver.integrate(self.solver.t+dt)

            # self.state = new_state
            #
            # self.solver.t += dt

            for w in walls:
                other = walls[w]
                # self.solver.set_f_params(other)

                # print("time:",self.solver.t)
                # print(self.name)

                # print("state: [x,y,dx,dy]",new_state)
                # self.solver.integrate(self.solver.t+dt)
                # Collision detection

                if not self.is_coll(other):
                    # self.state = new_state
                    #
                    # self.solver.t += dt
                    # print(self.pos)
                    pass
                else:
                    # print("\nCollision!\n\n")



class Universe:
    def __init__(self):
        self.w, self.h = window[0], window[1]
        self.objects_dict = {}
        self.walls_dict = {}
        self.objects = pygame.sprite.Group()
        self.dt = 0.033

    def add_wall(self, wall):
        self.walls_dict[wall.name] = wall
        self.objects.add(wall)

    def add_body(self, body):
        self.objects_dict[body.name] = body
        self.objects.add(body)

    def update(self):
        for o in self.objects_dict:
            # Compute positions for screen
            # print(self, self.objects_dict)
            obj = self.objects_dict[o]
            obj.update1(self.objects_dict, self.walls_dict, self.dt)
            # p = obj.to_screen()

            if False: # Set this to True to print the following values
                print ('Name', obj.name)
                print ('Position in simulation space', obj.state[:2])
                # print ('Position on screen', p)
                print ('Velocity in space', obj.state[-2:])

            # Update sprite locations
            obj.rect.x, obj.rect.y = obj.state[:2] - np.array([obj.radius, obj.radius])
        self.objects.update()
        # print("")
        # print(self.objects)

    def draw(self, screen):
        self.objects.draw(screen)

def main():

    # Initializing pygame
    pygame.init()
    win_width = window[0]
    win_height = window[0]
    screen = pygame.display.set_mode((win_width, win_height))  # Top left corner is (0,0)
    pygame.display.set_caption('Disks')

    universe = Universe()

    # Ball(name, mass, color=BLACK, radius=1)
    ball1 = Ball('ball1', 1, WHITE, 32)
    ball1.set_pos([80,80])
    ball1.set_vel([2,2])
    shape1 = Poly('shape1', [(100, 120), (110, 150), (160, 130)])
    shape2 = Poly('shape2', [(100, 200), (100, 500), (600, 300)])

    universe.add_wall(shape1)
    # universe.add_wall(shape2)
    universe.add_body(ball1)

    total_frames = 30000
    iter_per_frame = 1

    frame = 0
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
