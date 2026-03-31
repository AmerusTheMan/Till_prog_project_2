import pygame
from time import sleep


class Pad:
    def __init__(self, width, height, speed):
        self.x = 0
        self.y = 0
        self.speed = speed
        self.rect = pygame.Rect(self.x, self.y, width, height)



    def move(self, direction):
        """Move tha pad speed*direction pixels"""
        self.rect.centery += direction*self.speed


class Ball:
    def __init__(self, direction_vector, speed, width, height, startx, starty):
        self.x = startx
        self.y = starty
        self.rect = pygame.Rect(startx, starty, width, height)
        self.move_dir = pygame.Vector2(0, 0)
        self.set_direction((direction_vector))
        self.speed = speed
        self.times_colided_x = 0
        self.times_colided_y = 0

    def set_direction(self, dir: pygame.Vector2):
        """ Set the ball's direction to a normailzed vector of the input

        :param dir: Desired direction
        :return: None
        """
        self.move_dir = pygame.Vector2(dir).normalize()

    def __move(self, x: float, y: float):
        """Update absolute position and move self.rect.center to the closest pixel
        :param x: X distance to move
        :param y: Y distance to move
        :return: None
        """

        self.x += x
        self.y += y
        self.rect.center = (self.x, self.y)


    def __is_coliding(self, rects: List[pygame.Rect], lines: List[tuple]):
        """Check if self.rect is coliding with any of the passed rects or lines

        :param rects: List of rects the ball can colide with
        :param lines: List of lines the ball can colide with
        :return: None
        """
        is_coliding = False
        for pad in rects:
            if self.rect.colliderect(pad):
                #print("coliding rect")
                is_coliding = True
                break
        else:
            for line in lines:
                if self.rect.clipline(*line):
                    #print("coliding line")
                    is_coliding = True
                    break

        return is_coliding


    def move_and_bounce(self, pad_rects: List[pygame.Rect], screen_edge_lines: List[tuple]):
        """Moves the ball in self.move_dir self.speed times. Bounces if coliding with either of the rects or lines

        :param pad_rects: The pad rects the ball can bounce on
        :param screen_edge_lines: List of lines for where the screen edges are
        :return: None
        """
        for i in range(self.speed):
            self.__move(self.move_dir.x, 0)
            if self.__is_coliding(pad_rects, screen_edge_lines):
                while self.__is_coliding(pad_rects, screen_edge_lines):
                    self.__move(self.move_dir.x*-1, 0)
                self.move_dir.x *= -1

            self.__move(0, self.move_dir.y)
            if self.__is_coliding(pad_rects, screen_edge_lines):
                while self.__is_coliding(pad_rects, screen_edge_lines):
                    self.__move(0, self.move_dir.y*-1)
                self.move_dir.y *= -1





















