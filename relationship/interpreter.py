from typing import NamedTuple, List, Callable
import numpy as np
import torch
from itertools import product, groupby
from PIL import Image


# Do two line segments intersect? Copied from
# https://stackoverflow.com/questions/3838329/how-can-i-check-if-two-segments-intersect


class Box(NamedTuple):
    obj_name: str
    x: int
    y: int
    w: int = 0
    h: int = 0

    @property
    def object(self):
        return self.obj_name

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.w

    @property
    def top(self):
        return self.y

    @property
    def bottom(self):
        return self.y + self.h

    @property
    def center(self):
        return Box(self.x + self.w // 2, self.y + self.h // 2)

    def corners(self):
        yield Box(self.x, self.y)
        yield Box(self.x + self.w, self.y)
        yield Box(self.x + self.w, self.y + self.h)
        yield Box(self.x, self.y + self.h)

    @property
    def area(self):
        return self.w * self.h

    def intersect(self, other: "Box") -> "Box":
        x1 = max(self.x, other.x)
        x2 = max(x1, min(self.x+self.w, other.x+other.w))
        y1 = max(self.y, other.y)
        y2 = max(y1, min(self.y+self.h, other.y+other.h))
        return Box(x=x1, y=y1, w=x2-x1, h=y2-y1)

    def min_bounding(self, other: "Box") -> "Box":
        corners = list(self.corners())
        corners.extend(other.corners())
        min_x = min_y = float("inf")
        max_x = max_y = -float("inf")

        for item in corners:
            min_x = min(min_x, item.x)
            min_y = min(min_y, item.y)
            max_x = max(max_x, item.x)
            max_y = max(max_y, item.y)

        return Box(min_x, min_y, max_x - min_x, max_y - min_y)


def iou(box1, box2):
    x1 = max(box1.x, box2.x)
    x2 = max(x1, min(box1.x+box1.w, box2.x+box2.w))
    y1 = max(box1.y, box2.y)
    y2 = max(y1, min(box1.y+box1.h, box2.y+box2.h))
    intersection = Box(x=x1, y=y1, w=x2-x1, h=y2-y1)
    intersection_area = intersection.area
    union_area = box1.area+box2.area-intersection_area
    return intersection_area / union_area


def all_equal(iterable):
    """Are all elements the same?"""
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class Environment:
    def __init__(self, image: Image, box_sub: List[Box], box_obj: List[Box]):
        self.image = image
        self.sub = box_sub[0]
        self.obj = box_obj[0]

    def left_of(self):
        return (self.sub.right+self.sub.left) / 2 < (self.obj.right+self.obj.left) / 2

    def right_of(self):
        return (self.sub.right+self.sub.left) / 2 > (self.obj.right+self.obj.left) / 2

    def above(self):
        return (self.sub.bottom+self.sub.top) < (self.obj.bottom+self.obj.top)

    def below(self):
        return (self.sub.bottom+self.sub.top) > (self.obj.bottom+self.obj.top)

    def bigger_than(self):
        return self.sub.area > self.obj.area

    def smaller_than(self):
        return self.sub.area < self.obj.area

    def within(self):
        """Return percent of box1 inside box2."""
        intersection = self.sub.intersect(self.obj)
        return intersection.area / self.sub.area

    # @spatial()
    # def between(self, box1, box2, box3):
    #     """How much of box1 lies in min bounding box over box2 and box3?"""
    #     min_bounding = box2.min_bounding(box3)
    #     intersect = box1.intersect(min_bounding)
    #     return intersect.area / box1.area
