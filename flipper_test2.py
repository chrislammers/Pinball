import pygame
import random

# Initialize Pygame
pygame.init()

# Set the window size and title
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
WINDOW_TITLE = "Pinball Game"
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption(WINDOW_TITLE)

# Set the flipper size and position
FLIPPER_WIDTH = 100
FLIPPER_HEIGHT = 20
FLIPPER_Y = 500

# Set the flipper initial angle and rotation speed
LEFT_FLIPPER_ANGLE = 0
RIGHT_FLIPPER_ANGLE = 0
FLIPPER_ROTATION_SPEED = 5

# Create the left flipper
left_flipper = pygame.Rect(50, FLIPPER_Y, FLIPPER_WIDTH, FLIPPER_HEIGHT)
left_flipper_surface = pygame.Surface((FLIPPER_WIDTH, FLIPPER_HEIGHT))
left_flipper_surface.fill((255, 255, 255))

# Create the right flipper
right_flipper = pygame.Rect(WINDOW_WIDTH - 50 - FLIPPER_WIDTH, FLIPPER_Y, FLIPPER_WIDTH, FLIPPER_HEIGHT)
right_flipper_surface = pygame.Surface((FLIPPER_WIDTH, FLIPPER_HEIGHT))
right_flipper_surface.fill((255, 255, 255))

# Create the ball
BALL_RADIUS = 10
ball = pygame.draw.circle(screen, (255, 255, 255), (WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2), BALL_RADIUS)

# Set the initial ball speed and direction
ball_speed_x = random.randint(-5, 5)
ball_speed_y = random.randint(-5, 5)

# Draw the flippers on the screen
pygame.draw.rect(screen, (255, 255, 255), left_flipper)
pygame.draw.rect(screen, (255, 255, 255), right_flipper)

# Start the game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                LEFT_FLIPPER_ANGLE = -45
            elif event.key == pygame.K_RIGHT:
                RIGHT_FLIPPER_ANGLE = 45

        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                LEFT_FLIPPER_ANGLE = 0
            elif event.key == pygame.K_RIGHT:
                RIGHT_FLIPPER_ANGLE = 0

    # Draw the background
    screen.fill((0, 0, 0))

# Update the flipper rotation
    left_flipper.center = (50 + FLIPPER_WIDTH // 2, FLIPPER_Y + FLIPPER_HEIGHT // 2)
    right_flipper.center = (WINDOW_WIDTH - 50 - FLIPPER_WIDTH // 2, FLIPPER_Y + FLIPPER_HEIGHT // 2)

# Rotate the left half of the right flipper
    right_flipper_surface_rotated = pygame.Surface((FLIPPER_WIDTH, FLIPPER_HEIGHT))
    right_flipper_surface_rotated.blit(right_flipper_surface, (FLIPPER_WIDTH // 2, 0))
    right_flipper_surface_rotated = pygame.transform.rotate(right_flipper_surface_rotated, -RIGHT_FLIPPER_ANGLE)

# Rotate the right half of the left flipper
    left_flipper_surface_rotated = pygame.Surface((FLIPPER_WIDTH, FLIPPER_HEIGHT))
    left_flipper_surface_rotated.blit(left_flipper_surface, (0, 0, FLIPPER_WIDTH // 2, FLIPPER_HEIGHT))
    left_flipper_rotated = pygame.transform.rotate(left_flipper_surface_rotated, LEFT_FLIPPER_ANGLE)

# Draw the flippers on the screen
    screen.blit(left_flipper_rotated, left_flipper)
    screen.blit(right_flipper_surface_rotated, right_flipper)

# Update the ball position
    ball.x += ball_speed_x
    ball.y += ball_speed_y

# Check for ball collision with the left flipper
    if ball.colliderect(left_flipper):
        ball_speed_y *= -1

# Check for ball collision with the right flipper
    if ball.colliderect(right_flipper):
        ball_speed_y *= -1

# Check for ball collision with the walls
    if ball.left < 0 or ball.right > WINDOW_WIDTH:
        ball_speed_x *= -1
    if ball.top < 0 or ball.bottom > WINDOW_HEIGHT:
        ball_speed_y *= -1

# Draw the ball on the screen
    pygame.draw.circle(screen, (255, 255, 255), (ball.x, ball.y), BALL_RADIUS)

# Update the display
    pygame.display.flip()


pygame.quit()
