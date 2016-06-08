import reconstruction
import utils

def plot_images(images, shape, path, filename):
     # finally save to file
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    images = utils.reshape_and_tile_images(images, shape)
    plt.imshow(images, cmap='Greys')
    plt.axis('off')
    plt.savefig(path + filename + ".svg", format="svg")
    plt.close()
