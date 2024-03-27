import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Set up the window
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dodge the Obstacles")

# Set up colors
WHITE = (0, 20, 0)
RED = (255, 0, 0)

# Set up the player
player_width, player_height = 50, 50
player_x = WIDTH // 2 - player_width // 2

player_y = HEIGHT - player_height - 20
player_speed = 5

# Set up obstacles
obstacle_width, obstacle_height = 100, 20
obstacle_speed = 3
obstacle_color = RED
obstacles = []
spawn_timer = 0
spawn_delay = 60  # Spawn a new obstacle every 60 frames (1 second)

# Set up fonts
font = pygame.font.Font(None, 36)

# Main loop
running = True
score = 0
clock = pygame.time.Clock()
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Move the player
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        player_x -= player_speed
    if keys[pygame.K_RIGHT]:
        player_x += player_speed
    
    # Wrap the player around the screen
    if player_x < 0:
        player_x = WIDTH - player_width
    if player_x > WIDTH - player_width:
        player_x = 0
    
    # Spawn obstacles
    spawn_timer += 1
    if spawn_timer >= spawn_delay:
        spawn_timer = 0
        obstacle_x = random.randint(0, WIDTH - obstacle_width)
        obstacle_y = 0 - obstacle_height
        obstacles.append(pygame.Rect(obstacle_x, obstacle_y, obstacle_width, obstacle_height))
    
    # Move obstacles
    for obstacle in obstacles:
        obstacle.y += obstacle_speed
        if obstacle.y > HEIGHT:
            obstacles.remove(obstacle)
            score += 1
    
    # Check for collision with obstacles
    player_rect = pygame.Rect(player_x, player_y, player_width, player_height)
    for obstacle in obstacles:
        if player_rect.colliderect(obstacle):
            running = False
    
    # Clear the screen
    window.fill(WHITE)
    
    # Draw the player
    pygame.draw.rect(window, RED, pygame.Rect(player_x, player_y, player_width, player_height))
    
    # Draw obstacles
    for obstacle in obstacles:
        pygame.draw.rect(window, obstacle_color, obstacle)
    
    # Display the score
    score_text = font.render(f"Score: {score}", True, RED)
    window.blit(score_text, (10, 10))
    
    # Update the display
    pygame.display.flip()
    
    # Cap the frame rate
    clock.tick(60)

# Game over
game_over_text = font.render("Game Over", True, RED)
game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
window.blit(game_over_text, game_over_rect)
pygame.display.flip()

# Wait for a moment before quitting
pygame.time.wait(2000)

# Quit Pygame
pygame.quit()
sys.exit()
