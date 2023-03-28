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
        self.image.set_colorkey(BLACK)
        self.color = color
        # pygame.draw.rect(self.image, color, self.rect)
        self.rect = pygame.draw.polygon(self.image, color, points)
        self.name = name
        self.points = points

    def get_mass(self):
        return 1000000

    def get_vel(self):
        return np.zeros(2)

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

    def add_body(self, body, collides):
        self.objects_dict[body.name] = body
        self.objects.add(body)

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

    # Initializing pygame
    pygame.init()
    win_width = window[0]
    win_height = window[0]
    screen = pygame.display.set_mode((win_width, win_height))  # Top left corner is (0,0)
    pygame.display.set_caption('Disks')

    universe = Universe()

    shape1 = Poly('shape1', [(10, 20), (10, 50), (60, 30)])
    shape2 = Poly('shape2', [(100, 200), (100, 500), (600, 300)])
    universe.add_wall(shape1)
    universe.add_wall(shape2)

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
