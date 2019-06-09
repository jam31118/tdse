from matplotlib.colors import LinearSegmentedColormap

colors = [
    (1, 1, 1), (0, 0, 1), (0, 0.3, 1), (0, 0.5, 1), (0, 0.7, 1), (0, 1, 1), (0, 1, 0.5), 
    (0, 1, 0), (0.5, 1, 0), (1, 1, 0), (1, 0.7, 0), (1, 0.5, 0), (1, 0.3, 0), (1, 0, 0)
]
num_of_color_bins = 256
cmap_wjet = LinearSegmentedColormap.from_list("wjet", colors, N=num_of_color_bins)

