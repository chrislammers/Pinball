# Pinball
CSCI3010U Simulation and Modelling - Final Project


Object Oriented:

entities:
- Ball (circle): can move, hit things (circle) Only one exists at a time
- Flipper (rectangle): Ideally rotates, hits the ball, movement isn't affected by collisions. Only left and right exist
- Bumper (circles): No moving or rotating. gets hit by ball. Might have a bounce multiplier. Many can exist.
- Spring (spring): launches the ball, 1 dimensional sprint movement, one exists
