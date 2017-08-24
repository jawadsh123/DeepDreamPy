from PIL import Image

CANVAS_SIZE = (1280, 720)
COLOUR_MODE = "RGB"
START_COLOUR = (20, 20, 20)
CANVAS_NAME = "black_canvas_720.jpg"

im = Image.new(COLOUR_MODE, CANVAS_SIZE, START_COLOUR)
im.save("./images/"+CANVAS_NAME)