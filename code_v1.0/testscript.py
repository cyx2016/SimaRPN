# Importing Image and ImageOps module from PIL package
from PIL import Image, ImageOps, ImageStat, ImageDraw

# creating a image1 object
#tmp\test\Bird1\1_check_detection_target_in_padding\0000.jpg
im1 = Image.open("../tmp/test/Bird1/1_check_detection_target_in_padding/0000.jpg")
#im1.show()
#mean_detection = tuple(map(round, ImageStat.Stat(im1).mean))
print(im1.size)

# applying expand method
# using border value = 20
# using fill = 50 which is brown type color
padding = tuple(map(int, [80,20,60,40]))
print(padding)
#im2 = ImageOps.expand(im1, border=padding, fill=50)
im2 = ImageOps.crop(im1, (50,40,30,20))
print(im2.size)

#im2.show()