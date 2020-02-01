import os
import numpy as np
from PIL import Image

# image into flat array
def into_pixels(path):
    data = []
    for folder in '0123456789':
        cur_path = path + "/" + folder
        for filename in os.listdir(cur_path):
            img = Image.open(cur_path + '/' + filename).convert('L')
            vect = list(img.getdata())
            data.append(vect + [int(folder)])
        save = '{}'.format(path + '/' + folder + '_data')
        np.savetxt(save, np.array(data), delimiter=',')
        data.clear()


path = '~/training'
into_pixels(path)

path = '~/testing'
into_pixels(path)
