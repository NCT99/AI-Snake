from pygame.locals import *
import pygame
import time
import random
from collections import deque
import agent

EPISODES = 1000000

class Player:
    def __init__(self):
        self.x = 20
        self.y = 20
        self.step = 20
        self.direction = 0
        self.body = deque([])
        self.body.append((self.x,self.y))
        self.eaten = False
        
    def move(self):
        if self.direction == 0:
            self.x += self.step
        if self.direction == 1:
            self.y -= self.step
        if self.direction == 2:
            self.x -= self.step
        if self.direction == 3:
            self.y += self.step

    def update(self, has_eaten):
        self.body.append((self.x,self.y))
        if not has_eaten:
            self.body.popleft()

class Apple:
    def __init__(self, player):
        self.x = 80
        self.y = 20
        self.player = player

    def new_apple(self):
        #Here we are carefull to guarantee that the new apple location is not
        #on the body of the snake
        grid = {(20*x,20*y) for x in range(20) for y in range(30)}
        new_point = random.sample(grid.difference(self.player.body),1)
        self.x = new_point[0][0]
        self.y = new_point[0][1]

class Display:
    def __init__(self, player, apple):
        pygame.init()
        self.width = 400
        self.height = 600
        self.caption = "snake"
        self.bodyImage = "images/body.png"
        self.appleImage = "images/apple.png"
        self.body = player.body
        self.apple = apple
        self._display_surf = pygame.display.set_mode((self.width,self.height)
                                                     , pygame.HWSURFACE)
        pygame.display.set_caption(self.caption)
        self._image_surf = pygame.image.load(self.bodyImage).convert()
        self._image_surf2 = pygame.image.load(self.appleImage).convert()
        
    def render(self):
        pygame.event.pump()
        self._display_surf.fill((0,0,0))
        for (x,y) in self.body:
            self._display_surf.blit(self._image_surf,(x,y))
        self._display_surf.blit(self._image_surf2,(self.apple.x,self.apple.y))
        pygame.display.flip()
        
class Trainer:
    def __init__(self):
        self.player = Player()
        self.apple = Apple(self.player)
        self.agent = agent.Agent()
        self.display = Display(self.player, self.apple)
        self.running = True
        self.counter = 0
        self.reward = 0

    def is_crash(self):
        if ( self.player.x > 380 or  self.player.x < 0 or
        self.player.y > 580 or self.player.y < 0 or
        self.player.body.count((self.player.x,self.player.y))>1 ):
            return True
        return False
    
    def has_eaten(self):
        if self.player.x == self.apple.x and self.player.y == self.apple.y:
            return True
        return False
    
    def loop_iteration(self):
        self.counter += 1
        self.player.move()
        self.player.update(self.has_eaten())
        if self.has_eaten():
            self.counter = 0
            self.reward += 10
            self.apple.new_apple()
        self.display.render()
        if self.is_crash():
            self.running = False
        if self.counter > 400:
            self.running = False

    def get_state(self):
        state = deque([])
        state.extend([i == self.player.direction for i in range(4)])
        state.append(((self.player.x + 20, self.player.y) in self.player.body)
                     or  (self.player.x + 20 > 380))
        state.append(((self.player.x, self.player.y + 20) in self.player.body)
                     or  (self.player.y + 20 > 580))  
        state.append(((self.player.x - 20, self.player.y) in self.player.body)
                     or  (self.player.x - 20 < 0))
        state.append(((self.player.x, self.player.y - 20) in self.player.body)
                     or  (self.player.y - 20 < 0))
        state.append(self.player.x < self.apple.x)
        state.append(self.player.y < self.apple.y)
        return state

    def reset(self):
        self.running = True
        self.player.x = 20
        self.player.y = 20
        self.player.step = 20
        self.player.direction = 0
        self.player.body.clear()
        self.player.body.append((self.player.x,self.player.y))
        self.player.eaten = False
        self.apple.x = 80
        self.apple.y = 20
        self.counter = 0
        
    def quit(self):
        pygame.quit()
        
    def run(self):
        self.agent.load_weights()
        for e in range(EPISODES):
            while( self.running ):
                state = self.get_state()
                self.reward = 0
                over = False
                
                action = self.agent.action(state)
                if action == 0:
                    if len(self.player.body) == 1 or not self.player.direction == 2:
                        self.player.direction = 0
                elif action == 1:
                    if len(self.player.body) == 1 or not self.player.direction == 3:
                        self.player.direction = 1
                elif action == 2:
                    if len(self.player.body) == 1 or not self.player.direction == 0:
                        self.player.direction = 2
                elif action == 3:
                    if len(self.player.body) == 1 or not self.player.direction == 1:
                        self.player.direction = 3
                self.loop_iteration()
                
                next_state = self.get_state()
                if not self.running:
                    over = True
                    self.reward -= 1
                    #print(len(self.player.body))
                self.agent.memorize(state, action, self.reward, next_state, over)
                time.sleep (50.0 / 1000.0);
            if len(self.agent.memory) > self.agent.batch_size:
                minibatch = random.sample(self.agent.memory, self.agent.batch_size)
            else:
                minibatch = self.agent.memory
            self.agent.learn(minibatch)
            self.agent.epsilon *= self.agent.epsilon_decay
            self.reset()
            if e % 100 == 0:
                pass
                #self.agent.save_weights()
        self.quit()
        
if __name__ == "__main__" :
    trainer = Trainer()
    trainer.run()
                
        
        
        
