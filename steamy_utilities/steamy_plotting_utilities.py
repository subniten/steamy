import matplotlib.dates as mdates


def rotate_xtick_labels(ax, rotation=0, horizontal_alignment='center'):
    """Rotate xlabels so the are slanted in the desired angle

    This can be used to rectify slanted xlabels if you prefer non-slanted xlabels for instance.

    Args:
        ax (figure aces): axes whose x axis to correct
        rotation (int, optional): rotation in degrees. Defaults to 0.
        horizontal_alignment (str, optional): How should the label be justified relative the x-tick. Defaults to 'center'.
    """
    for xlabels in ax.get_xticklabels():
        xlabels.set_rotation(rotation)
        xlabels.set_ha(horizontal_alignment)


def time_axis_formatter(ax, interval=None):
    if interval is not None:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval))
    else:
        ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
