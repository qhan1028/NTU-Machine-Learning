from PIL import Image
import sys

im1 = Image.open(sys.argv[1])
im2 = Image.open(sys.argv[2])
im1_pixels = im1.load()
im2_pixels = im2.load()

result = Image.new('RGBA', im1.size)
result_pixels = result.load()

for i in range(im1.size[0]):
	for j in range(im1.size[1]):
		if im1_pixels[i, j] == im2_pixels[i, j]:
			result_pixels[i, j] = (0, 0, 0, 0) 
		else:
			result_pixels[i, j] = im2_pixels[i, j]

result.save('ans_two.png')
