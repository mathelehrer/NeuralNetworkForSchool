import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from matplotlib import pyplot as plt

pi = np.pi

def fluc():
    return 1+(-0.1 + random.random()*0.2)

def generate_convex_polygon_image(n):
    """
    function for an 28x28 gray scale image of a convex polygon from n random vertex positions
    :param n: number of vertices
    """
    # Generate n points:
    # compute n directions which slightly vary from the regular spacing
    # vary the length



    points = [(int(random.randrange(10, 14) * fluc() * np.cos(2 * pi / n * i)) , int(
        random.randrange(4, 12) * fluc() * np.sin(2 * pi / n * i)) ) for i in range(n)]

    # rotate points around random angle
    angle = random.random()*2*pi
    points =[(x * np.cos(angle) - y * np.sin(angle), x * np.sin(angle) + y * np.cos(angle)) for x,y in points]

    # shift centers

    points = [(x+14,y+14) for x,y in points]
    # Compute Convex Hull using Monotone Chain algorithm

    if len(points) <= 1:
        hull = points
    else:
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and cross_product(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross_product(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull
        hull = lower[:-1] + upper[:-1]

    # Create a blank 28x28 image (L mode for grayscale)

    img = Image.new('L', (28, 28), 0)
    draw = ImageDraw.Draw(img)

    # Draw the polygon
    if len(hull) > 2:
        draw.polygon(hull, fill=255)

    # blur edges with a Gaussian filter
    img = img.filter(ImageFilter.GaussianBlur(radius=1))
    return np.array(img)


def cross_product(o, a, b):
    """Cross product of vectors OA and OB. A positive cross product indicates a counter-clockwise turn."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


if __name__ == '__main__':
    img = generate_convex_polygon_image(10)
    for x in range(len(img)):
        for y in range(len(img[0])):
            if img[x][y]<10:
                print("  ",end="")
            elif img[x][y]<50:
                print("..",end="")
            elif img[x][y]<50:
                print("..",end="")
            elif img[x][y]<100:
                print("oo",end="")
            elif img[x][y]<150:
                print("xx",end="")
            elif img[x][y]<200:
                print("AO",end="")
            else:
                print( "XX",end="")
        print()


    imgplot = plt.imshow(img)
    plt.show()
