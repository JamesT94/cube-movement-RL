import os
import neat
import pickle
import pygame
pygame.init()

# Initial setup and global variables
WIN_WIDTH = 600
WIN_HEIGHT = 600
STAT_FONT = pygame.font.SysFont("calibri", 50)
WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Cube Movement AI")


def clamp(n, minn, maxn):  # Helpful function for clamping values between constraints
    return max(min(maxn, n), minn)


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


def load_model(filename):
    pickle_in = open(filename, "rb")
    return pickle.load(pickle_in)


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

            # Green movement calculations
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

            # Red1 movement calculations
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

            # Red2 movement calculations
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
            pickle_out = open("green-winner.pickle", "wb")
            pickle.dump(nets[0], pickle_out)
            pickle_out.close()
            break

        draw_window(WIN, greens, reds1, reds2, score)


def train_red(genomes, config):

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
    score = 1250

    green_nn = load_model('green-winner.pickle')

    run = True
    while run and len(reds1) > 0:
        clock.tick(1000)
        score -= 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        if len(reds1) < 1:
            run = False
            break

        # Check for collisions
        for x, green in enumerate(greens):
            if reds1[x].get_rect().colliderect(green.get_rect()) or reds2[x].get_rect().colliderect(green.get_rect()):
                ge[greens.index(green)].fitness += 500
                nets.pop(greens.index(green))
                ge.pop(greens.index(green))
                greens.pop(greens.index(green))
                reds1.pop(x)
                reds2.pop(x)

        for x, green in enumerate(greens):
            ge[x].fitness -= 1

            green_output = green_nn.activate((green.x, green.y, reds1[x].x, reds1[x].y, reds2[x].x, reds2[x].y))

            # Green movement calculations
            if green_output[0] > 0.5 and green.x > green.vel:
                green.aX = clamp(green.aX - 0.1, green.minvel, green.maxvel)
            elif green_output[0] < 0.5 and green.x < WIN_WIDTH - green.width - green.vel:
                green.aX = clamp(green.aX + 0.1, green.minvel, green.maxvel)
            else:
                green.aX = clamp(green.aX * 0.3, green.minvel, green.maxvel)

            if green_output[1] > 0.5 and green.y > green.vel:
                green.aY = clamp(green.aY - 0.1, green.minvel, green.maxvel)
            elif green_output[1] < 0.5 and green.y < WIN_HEIGHT - green.height - green.vel:
                green.aY = clamp(green.aY + 0.1, green.minvel, green.maxvel)
            else:
                green.aY = clamp(green.aY * 0.3, green.minvel, green.maxvel)

            green.move()

            output = nets[x].activate((green.x, green.y, reds1[x].x, reds1[x].y, reds2[x].x, reds2[x].y))

            # Red1 movement calculations
            if output[0] > 0.5:
                reds1[x].x += reds1[x].vel
            if output[0] < 0.5:
                reds1[x].x -= reds1[x].vel
            if output[1] > 0.5:
                reds1[x].y += reds1[x].vel
            if output[1] < 0.5:
                reds1[x].y -= reds1[x].vel

            # Red2 movement calculations
            if output[2] > 0.5:
                reds2[x].x += reds2[x].vel
            if output[2] < 0.5:
                reds2[x].x -= reds2[x].vel
            if output[3] > 0.5:
                reds2[x].y += reds2[x].vel
            if output[3] < 0.5:
                reds2[x].y -= reds2[x].vel

        if score < 0:
            break

        draw_window(WIN, greens, reds1, reds2, score)


def run_green(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(train_green, 50)
    print('\nBest genome:\n{!s}'.format(winner))


def run_red(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(train_red, 50)
    print('\nBest genome:\n{!s}'.format(winner))

    pickle_out = open("red-winner.pickle", "wb")
    pickle.dump(winner, pickle_out)
    pickle_out.close()


config_file_name = "config-red-1.txt"

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file_name)
    run_red(config_path)
