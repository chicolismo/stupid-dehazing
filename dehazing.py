import cv
import numpy as np

class Image:
    patch_size = 15

    def __init__(self, filename):
        self.matrix = cv.LoadImage(filename)
        self.width = self.matrix.width
        self.height = self.matrix.height

    def dark_channel(self):
        dark = cv.iplimage
        print(dark)

if __name__ == '__main__':
    # print(cv.Mat)
    image = Image('image2.jpeg')
    image.dark_channel()
    # print(image.width)
    # print(image.height)
    # image.dark_channel()

