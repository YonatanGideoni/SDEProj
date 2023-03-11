if __name__ == "__main__":

    import pickle
    from matplotlib import pyplot as plt, animation as animation
    from mpl_toolkits.axes_grid1 import ImageGrid

    ms_list = []
    ts_list = []
    small_nums = [1e-2, 1e-3, 1e-4]
    for SMALL_NUMBER in small_nums:
        rtol, atol = 1e-8, 1e-8
        fname = f'gt_{1e-7}_{rtol}_{atol}_{42}_{SMALL_NUMBER}.pkl'
        with open(fname, 'rb') as f:
                loaded_dict = pickle.load(f)
                ms, ts = loaded_dict["ms"], loaded_dict["ts"]
                ms_list.append(ms)
                ts_list.append(ts)
    
    fig = plt.figure(figsize=(6., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(1, 3),
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    
    # Iterating over the grid returns the Axes.
    for ax, ms, title in zip(grid, ms_list, small_nums):
        ax.imshow(ms[:-1, -1].reshape(28, 28))
        ax.set_title(title)
    
    plt.show()
        