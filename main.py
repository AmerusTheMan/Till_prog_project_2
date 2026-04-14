
import pygame
import numpy as np
from network import Network
import pong_objects
from random import randint, random
import os



#screen variables
screen_width, screen_height = 800, 450
screen_bg_color = (0, 0, 0)


#pygame setup
pygame.init()
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("PONG GAME")
screen.fill(screen_bg_color)
clock = pygame.time.Clock()
delta_t = 0

#pad variables
pad_width = 7
pad_height = 50
pad_screen_distance = 30
pad_color = (255, 255, 255)
pad_speed = 7

pad1_up_key = pygame.K_w
pad1_down_key = pygame.K_s
pad2_up_key = pygame.K_o
pad2_down_key = pygame.K_l



#ball variables
ball_width = 10
ball_height = 10
ball_speed = 15
ball_color = (255, 255, 255)


#create objects
pad1 = pong_objects.Pad(
    width=pad_width,
    height=pad_height,
    speed=pad_speed
)

pad2 = pong_objects.Pad(
    width=pad_width,
    height=pad_height,
    speed=pad_speed
)

ball = pong_objects.Ball(
    direction_vector=(1, 1),
    speed=ball_speed,
    width=ball_width,
    height=ball_height,
    startx=screen_width/2,
    starty=screen_height/2
)

ai = Network(
    layers = [5, 20, 20, 20, 1],
    model_path="./model_temp.pkl"
)
ai_certanty_threshold = 0.8 # how certain the ai has to be to make an action

#move pads to correct positions
pad1.rect.center = (pad_screen_distance, screen_height/2)
pad2.rect.center = (screen_width-pad_screen_distance, screen_height/2)


#lines that sit on the screen's edges. Used for collision checking in Ball.move_and_bounce
screen_edge_lines = [
    ((0, 0), (screen_width, 0)),
    ((screen_width, 0), (screen_width, screen_height)),
    ((screen_width, screen_height), (0, screen_height)),
    ((0, screen_height), (0, 0))
]


def format_ai_input(pad_ball_distance: float, ball_y: float, ball_dir: float, pad_y: float) -> np.array():
    """ Takes a list of normalized values and formats them into a 5x1 matrix that the ai can use as input

    :param pad_ball_distance: distance between ball and pad
    :param ball_y: ball y coord
    :param ball_dir: ball move direciton
    :param pad_y: pad y coord
    :return: input for ai
    """
    temp_input = [
        [pad_ball_distance],
        [ball_y],
        [ball_dir.x],
        [ball_dir.y],
        [pad_y]
    ]

    input = np.array(temp_input)

    return input


def clear_collected_data():
    """removes all sets from the folder 'collected data'"""
    for name in os.listdir("./collected_data"):
        os.remove(os.path.join("./collected_data", name))


def random_dir():
    """Generate a random direction where neither x or y direciton is 0"""
    x = 0
    y = 0
    while x == 0 or y == 0:
        x = randint(-10, 10)
        y = randint(-10, 10)

    return pygame.Vector2(x, y).normalize()



# Functions for normalizing coordinates and directions
# Necessary for the ai to work on all screen dimensions
def normalize_x_coord(coord):
    """Map x coordinate to 0-1 based on screen width"""
    return coord/screen_width

def normalize_y_coord(coord):
    """Map y coordinate to 0-1 based on screen height"""
    return coord/screen_height

def normalize_move_dir(x, y):
    """Map a direction vector based on screen width and height"""
    return pygame.Vector2(x/screen_width, y/screen_height).normalize()


def draw_screen():
    """Updates the screen to display all pong objects"""
    screen.fill(screen_bg_color)

    pygame.draw.rect(screen, pad_color, pad1.rect)
    pygame.draw.rect(screen, pad_color, pad2.rect)
    pygame.draw.rect(screen, ball_color, ball.rect)

    #for line_i in screen_edge_lines:
    #    pygame.draw.line(screen, "white", *line_i, width=10)

    pygame.display.flip()


def run_game(collect_data=False, max_set_size_GB=1, clear_data=True):
    """Run the pong game
    
    :param colect_data: if the game should collect data or play normally (default: False)
    :param max_set_size_GB: max gigabytes per set saved (default: 1)
    :param clear_data: if all previously collected data should be erased before collecting new data (default: True)
    """

    if collect_data == True:
        if clear_data: clear_collected_data()
        max_bytes = round((1024**3) * max_set_size_GB)
        data = {"collections": [], "results": []}
        data_size = 0 # Total byte size of current set

        # What ball.centerx will be when it bounces with pad2. Used to decide when the result for current collection will be collected
        pad2_bounce_x = screen_width - pad_screen_distance - (pad_width / 2) - (ball_width / 2)

        # A 2D array with format [...[pad_ball_distance, ball_y, ball_dir_x, ball_dir_y]...].
        # mapped to a result when the ball goes past pad2_bounce_x
        current_collection = []

        collected_result = False # If a result have been collected for the current collection
        n_sets_saved = 0 # How many sets that already have been saved
        ball.speed=2*ball_speed # A higher multiplier will lower the length of current_collection for each result but will result in more results per set

        #move pads out of the way
        pad1.move(100)
        pad2.move(100)


    ball.set_direction(random_dir())
    #ball.set_direction(pygame.Vector2(1, 1))

    running = True
    while running:
        for event in pygame.event.get():
            # Listen for window close
            if event.type == pygame.QUIT:
                running = False


        keys = pygame.key.get_pressed()

        ball.move_and_bounce([pad1.rect, pad2.rect], screen_edge_lines)


        if not collect_data:
            pad1.move(keys[pad1_down_key] - keys[pad1_up_key])

            ai_input = format_ai_input(
                pad_ball_distance=normalize_x_coord(pad2.rect.left-ball.rect.right),
                ball_y=normalize_y_coord(ball.rect.centery),
                ball_dir=normalize_move_dir(ball.move_dir.x, ball.move_dir.y),
                pad_y=normalize_y_coord(pad2.rect.centery)
            )

            ai_action = (ai.calc_output(ai_input)[0])

            if ai_action < 1-ai_certanty_threshold:
                pad2.move(-1)
            elif ai_action > ai_certanty_threshold:
                pad2.move(1)
            #print(round(ai_action, 4))

            # Point the ball in a random direction if it foes past pad1. Used to test whether the ai can handle multiple directions
            if ball.rect.centerx < pad_screen_distance:
                ball.set_direction(random_dir())
                while ball.rect.centerx < pad_screen_distance:
                    ball.move_and_bounce([pad1.rect, pad2.rect], screen_edge_lines)


            draw_screen()
            clock.tick(60) #limit framerate to 60fps

        elif collect_data:
            normalized_pad_ball_distance = normalize_x_coord(pad2.rect.left-ball.rect.right)
            normalized_y_coord = normalize_y_coord(ball.rect.centery)

            normalized_move_dir = normalize_move_dir(ball.move_dir.x, ball.move_dir.y)

            current_collection.append([normalized_pad_ball_distance, normalized_y_coord, normalized_move_dir.x, normalized_move_dir.y, random()])


            if ball.rect.centerx >= pad2_bounce_x and not collected_result:
                arr = np.array(current_collection, dtype=np.float32)
                current_collection = []
                #print(arr)
                collection_size = arr.nbytes
                if data_size + collection_size > max_bytes: # If adding the current collection to data will exceed max set size
                    print(f"set_{n_sets_saved}: {len(data['results'])} results")
                    data["results"] = np.array(data["results"], dtype=np.float32)

                    # Dump the current set
                    with open(f"./collected_data/set_{n_sets_saved}.npy", "wb+") as file:
                        np.save(file, data["results"])
                        for collection in data["collections"]:
                            np.save(file, collection)

                    # Reset Set values
                    data = {"collections": [], "results": []}
                    data_size = 0
                    n_sets_saved += 1

                # Add current_collection and it's result to current set
                data["collections"].append(arr)
                data["results"].append(normalize_y_coord(ball.rect.centery))
                data_size += collection_size
                collected_result = True
                ball.move_dir = random_dir()

                del arr

            elif collected_result and ball.rect.centerx < pad2_bounce_x:
                collected_result = False

            #Uncomment to debug data collection
            #draw_screen()
            #clock.tick(60)

    # Save the remaining collections to one last set before the program closes
    if collect_data:
        data["results"] = np.array(data["results"])
        with open(f"./collected_data/set_{n_sets_saved}.npy", "wb+") as file:
            np.save(file, data["results"])
            for collection in data["collections"]:
                np.save(file, collection)



if __name__=="__main__":

    run_game(
        collect_data=False,
        max_set_size_GB=0.01
    )


pygame.quit()
