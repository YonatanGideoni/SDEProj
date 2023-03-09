from matplotlib import pyplot as plt, animation as animation
from mpl_toolkits.axes_grid1 import ImageGrid


def plot_trajectory(ms, ts, solver_name: str = ''):
    plt.figure()

    for m in ms:
        plt.plot(ts, m)

    plt.xlabel("Time [s]")
    plt.ylabel("x")
    plt.title("Pixel's values over time for " + solver_name)


# TODO - plot also time
def plot_reverse_process(ms, title: str = ''):
    fig = plt.figure()
    plt.title(title)

    # make axesimage object
    # the vmin and vmax here are very important to get the color map correct
    im = plt.imshow(ms[:-1, 0].reshape(28, 28), cmap=plt.get_cmap('jet'), vmin=0, vmax=1.0)

    # function to update figure
    def updatefig(j):
        # set the data in the axesimage object
        im.set_array(ms[:-1, j].reshape(28, 28))
        # return the artists set
        return [im]

    # kick off the animation
    ani = animation.FuncAnimation(fig, updatefig, frames=range(len(ms[0])),
                                  interval=75, blit=True, repeat_delay=2000)
    plt.show()


def plot_final_results(diffusions: list, nrows_ncols: tuple = None):
    if nrows_ncols is None:
        n_images = len(diffusions)
        if n_images % 2 == 0:
            nrows_ncols = (len(diffusions) // 2, 2)
        else:
            nrows_ncols = (len(diffusions), 1)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=nrows_ncols,
                     axes_pad=0.1,  # pad between axes in inch.
                     )

    for ax, [diffusion, title] in zip(grid, diffusions):
        # Iterating over the grid returns the Axes.
        ax.imshow(diffusion[:, -1])
        ax.set_title(title)

    plt.imshow()
