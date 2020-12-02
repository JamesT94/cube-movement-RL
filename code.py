import os
import neat
import pickle
import pygame
pygame.init()

WIN_WIDTH = 600
WIN_HEIGHT = 600
STAT_FONT = pygame.font.SysFont("calibri", 50)

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Cube Chaser")


class Green(object):
    def __init__(self, x, y, aX, aY, width, height, minvel, maxvel):
        self.x = x
        self.y = y
        self.aX = aX
        self.aY = aY
        self.width = width
        self.height = height
        self.minvel = minvel
        self.maxvel = maxvel
        self.vel = 4

    def move(self):
        self.x += self.aX * self.vel
        self.y += self.aY * self.vel

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, win):
        pygame.draw.rect(WIN, (0, 255, 0), (self.x, self.y, self.width, self.height))


class Red(object):
    def __init__(self, x, y, aX, aY, width, height, minvel, maxvel):
        self.x = x
        self.y = y
        self.aX = aX
        self.aY = aY
        self.width = width
        self.height = height
        self.minvel = minvel
        self.maxvel = maxvel
        self.vel = 1.5

    def move(self):
        self.x += self.aX * self.vel
        self.y += self.aY * self.vel

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)

    def draw(self, win):
        pygame.draw.rect(WIN, (255, 0, 0), (self.x, self.y, self.width, self.height))


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def draw_window(win, greens, reds1, reds2, score):
    win.fill((0, 0, 0))

    for green in greens:
        green.draw(win)

    for red in reds1:
        red.draw(win)

    for red in reds2:
        red.draw(win)

    text = STAT_FONT.render(str(score), 1, (255, 255, 255))
    textRect = text.get_rect()
    win.blit(text, textRect)

    pygame.display.update()


def train_green(genomes, config):

    global WIN

    nets = []
    ge = []
    greens = []
    reds1 = []
    reds2 = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        greens.append(Green(150, 150, 0, 0, 30, 30, -1, 1))
        reds1.append(Red(250, 250, 0, 0, 30, 30, -1, 1))
        reds2.append(Red(200, 350, 0, 0, 30, 30, -1, 1))
        g.fitness = 0
        ge.append(g)

    clock = pygame.time.Clock()
    score = 0

    run = True
    while run and len(greens) > 0:
        clock.tick(50)
        score += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        if len(greens) < 1:
            run = False
            break

        # Increase speed of red AI
        for red in reds1:
            red.vel += 0.0001

        for red in reds2:
            red.vel += 0.0001

        # Check for collisions
        for x, green in enumerate(greens):
            if reds1[x].get_rect().colliderect(green.get_rect()) or reds2[x].get_rect().colliderect(green.get_rect()):
                ge[greens.index(green)].fitness -= 1
                nets.pop(greens.index(green))
                ge.pop(greens.index(green))
                greens.pop(greens.index(green))
                reds1.pop(x)
                reds2.pop(x)

        for x, green in enumerate(greens):
            ge[x].fitness += 1

            output = nets[x].activate((green.x, green.y, reds1[x].x, reds1[x].y, reds2[x].x, reds2[x].y))

            if output[0] > 0.5 and green.x > green.vel:
                green.aX = clamp(green.aX - 0.1, green.minvel, green.maxvel)
            elif output[0] < 0.5 and green.x < WIN_WIDTH - green.width - green.vel:
                green.aX = clamp(green.aX + 0.1, green.minvel, green.maxvel)
            else:
                green.aX = clamp(green.aX * 0.3, green.minvel, green.maxvel)

            if output[1] > 0.5 and green.y > green.vel:
                green.aY = clamp(green.aY - 0.1, green.minvel, green.maxvel)
            elif output[1] < 0.5 and green.y < WIN_HEIGHT - green.height - green.vel:
                green.aY = clamp(green.aY + 0.1, green.minvel, green.maxvel)
            else:
                green.aY = clamp(green.aY * 0.3, green.minvel, green.maxvel)

            green.move()

            movered1x = green.x - reds1[x].x
            movered1y = green.y - reds1[x].y

            if movered1x > 0:
                reds1[x].x += reds1[x].vel
            if movered1x < 0:
                reds1[x].x -= reds1[x].vel
            if movered1y > 0:
                reds1[x].y += reds1[x].vel
            if movered1y < 0:
                reds1[x].y -= reds1[x].vel

            movered2x = green.x - reds2[x].x
            movered2y = green.y - reds2[x].y

            if movered2x > 0:
                reds2[x].x += reds2[x].vel
            if movered2x < 0:
                reds2[x].x -= reds2[x].vel
            if movered2y > 0:
                reds2[x].y += reds2[x].vel
            if movered2y < 0:
                reds2[x].y -= reds2[x].vel

        if score > 1250:
            break

        draw_window(WIN, greens, reds1, reds2, score)


def read_model():
    pickle_in = open("dict.pickle", "rb")
    winner = pickle.load(pickle_in)


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(train_green, 50)
    print('\nBest genome:\n{!s}'.format(winner))

    pickle_out = open("dict.pickle", "wb")
    pickle.dump(winner, pickle_out)
    pickle_out.close()


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
