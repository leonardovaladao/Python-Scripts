import pygame, sys, random, time

# Look for errors
if pygame.init()[1]>0:
    print("(!) Found {0} errors, exiting... ".format(pygame.init()[1]))
    sys.exit(-1)
else:
    print("Pygame Loaded.")
    
# Interface
Interface = pygame.display.set_mode((720,460))
pygame.display.set_caption('SnakePy')

# Colors
red = pygame.Color(255, 117, 109)
green = pygame.Color(133, 222, 119)
blue = pygame.Color(127, 147, 145)
white = pygame.Color(240, 240, 240)
brown = pygame.Color(225, 184, 148)

# FPS
fpsController = pygame.time.Clock()

# Variables
snk_position = [100, 50]
snk_body = [[100, 50], [90, 50], [80, 50]]
food_position = [random.randrange(1, 72)*10, random.randrange(1, 46)*10]
food_on = True
score = 0

# Direction
direct = 'RIGHT'
change = direct

# Game Over
def game_over():
    myFont = pygame.font.SysFont('Roboto', 72)
    GOsurf = myFont.render('Game Over', True, red)
    GOrect = GOsurf.get_rect()
    GOrect.midtop = (360, 100)
    Interface.blit(GOsurf,GOrect)
    show_score(0)
    pygame.display.flip()
    time.sleep(2)
    pygame.quit()
    sys.exit()
    
def show_score(choice=1):
    sFont = pygame.font.SysFont('Roboto', 24)
    Ssurf = sFont.render("Score: {0}".format(score), True, blue)
    Srect = Ssurf.get_rect()
    if choice == 1:    
        Srect.midtop = (60, 10)
    else:
        Srect.midtop = (360, 180)
    Interface.blit(Ssurf, Srect)
    
    
# Main Game
while True:
    # Keyboard
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT or event.key == ord('d'):
                change = 'RIGHT'
            if event.key == pygame.K_LEFT or event.key == ord('a'):
                change = 'LEFT'
            if event.key == pygame.K_UP or event.key == ord('w'):
                change = 'UP'
            if event.key == pygame.K_DOWN or event.key == ord('s'):
                change = 'DOWN'
            if event.key == pygame.K_ESCAPE:
                pygame.event.post(pygame.event.Event(pygame.QUIT))
    
    
    # Direction 
    if change == 'RIGHT' and not direct == 'LEFT':
        direct = 'RIGHT'
    if change == 'LEFT' and not direct == 'RIGHT':
        direct = 'LEFT'
    if change == 'UP' and not direct == 'DOWN':
        direct = 'UP'
    if change == 'DOWN' and not direct == 'UP':
        direct = 'DOWN'
        
    if direct == 'RIGHT':
        snk_position[0] += 10
    if direct == 'LEFT':
        snk_position[0] -= 10
    if direct == 'UP':
        snk_position[1] -= 10
    if direct == 'DOWN':
        snk_position[1] += 10
        
    # Body
    snk_body.insert(0, list(snk_position))
    if snk_position[0] == food_position[0] and snk_position[1] == food_position[1]:
        score += 1
        food_on = False
    else:
        snk_body.pop()
        
    # Food
    if food_on == False:
        food_position = [random.randrange(1, 72)*10, random.randrange(1, 46)*10]
    food_on = True
        
    # Graphics
    Interface.fill(white)
    
    # Draw Snake
    for pos in snk_body:
        pygame.draw.rect(Interface, green, pygame.Rect(pos[0],pos[1],10,10))
   
    # Draw Food
    pygame.draw.rect(Interface, brown, pygame.Rect(food_position[0],food_position[1],10,10))
    
    # Boundaries
    if snk_position[0] > 710 or snk_position[0] < 0:
        game_over()
    if snk_position[1] > 450 or snk_position[1] < 0:
        game_over()
    for block in snk_body[1:]:
        if snk_position[0] == block[0] and snk_position[1] == block[1]:
            game_over()
    
        
    # Update Screen
    show_score()
    pygame.display.flip()
    fpsController.tick(24)
            
