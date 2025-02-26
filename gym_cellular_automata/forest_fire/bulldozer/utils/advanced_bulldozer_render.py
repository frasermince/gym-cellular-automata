"""
The render of the bulldozer consists of four subplots:
1. Local Grid
    + Grid centered at current position, visualizes agent's micromanagment
2. Global Grid
    + Whole grid view, visualizes agent's strategy
3. Gauge
    + Shows time until next CA update
4. Counts
    + Shows Forest vs No Forest cell counts. Translates on how well the agent is doing.
"""

from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np

from gym_cellular_automata.forest_fire.utils.neighbors import moore_n
from gym_cellular_automata.forest_fire.utils.render import (
    EMOJIFONT,
    TITLEFONT,
    align_marker,
    clear_ax,
    get_norm_cmap,
    parse_svg_into_mpl,
    plot_grid,
)

from . import svg_paths

# Figure Globals
FIGSIZE = (15, 12)
FIGSTYLE = "seaborn-v0_8-whitegrid"

TITLE_SIZE = 42
TITLE_POS = {"x": 0.121, "y": 0.96}
TITLE_ALIGN = "left"

# Day colors
COLOR_EMPTY_DAY = "#DDD1D3"  # Gray
COLOR_TREE_DAY = "#A9C499"  # Green
COLOR_FIRE_DAY = "#E68181"  # Salmon-Red
COLOR_BURNED_DAY = "#8B0000"  # Dark Red
COLOR_GAUGE_DAY = "#D4CCDB"  # Gray-Purple

# Night colors
COLOR_EMPTY_NIGHT = "#696969"  # Darker Gray
COLOR_TREE_NIGHT = "#2F4F4F"  # Dark Green
COLOR_FIRE_NIGHT = "#8B0000"  # Dark Red
COLOR_BURNED_NIGHT = "#8B0000"  # Dark Red
COLOR_GAUGE_NIGHT = "#483D8B"  # Dark Slate Blue

# Local Grid
N_LOCAL = 3  # n x n local grid size
MARKBULL_SIZE = 52

# Global Grid
MARKFSEED_SIZE = 62
MARKLOCATION_SIZE = 62

# Gauge
CYCLE_SYMBOL = "\U0001f504"
CYCLE_SIZE = 32

# Counts
TREE_SYMBOL = "\U0001f332"
BURNED_SYMBOL = "\ue08a"


# Ignore warnings trigger by Bulldozer Render
# EmojiFont raises RuntimeWarning
filterwarnings("ignore", message="Glyph 108")
filterwarnings("ignore", message="Glyph 112")


def plot_grid_attribute(grid, attribute_name):
    """Renders a visualization of the given grid attribute (altitude, density, etc).

    Args:
        grid: MxN numpy array containing attribute values
        attribute_name: String name of the attribute being plotted (e.g. "Altitude", "Density")
    """
    # Get min and max values
    min_val = np.min(grid)
    max_val = np.max(grid)

    # Create 5 evenly spaced ranges
    num_ranges = 5
    range_size = (max_val - min_val) / num_ranges
    values = [min_val + i * range_size for i in range(num_ranges + 1)]

    # Define colors for different ranges - from light to dark
    colors = [
        "#FFF5F0",  # Lightest - almost white
        "#FEE0D2",
        "#FCBBA1",
        "#FC9272",
        "#FB6A4A",
        "#CB181D",  # Darkest red
    ]

    NORM, CMAP = get_norm_cmap(values, colors)

    def main():
        plt.style.use(FIGSTYLE)
        # Create figure and axis
        fig, ax = plt.subplots()

        # Plot grid
        ax.imshow(grid, interpolation="none", cmap=CMAP, norm=NORM)

        # Add colorbar with actual values
        cbar = plt.colorbar(
            ax.images[0], ax=ax, label=attribute_name, orientation="horizontal"
        )

        # Format colorbar ticks to show actual values
        cbar.set_ticks(values)
        cbar.set_ticklabels([f"{val:.1f}" for val in values])

        # Add title
        ax.set_title(f"Terrain {attribute_name} Map", pad=10, fontsize=12)

        # Clear axes
        clear_ax(ax)

        return plt

    return main()


def render(
    empty,
    tree,
    fire,
    title,
    grid,
    time,
    pos,
    cell_count,
    pos_fire,
    dousing_counts,
    wind_index=None,
    is_night=0,
):
    EMPTY = empty
    TREE = tree
    FIRE = fire
    # Select color scheme based on is_night
    if is_night:
        COLOR_EMPTY = COLOR_EMPTY_NIGHT
        COLOR_TREE = COLOR_TREE_NIGHT
        COLOR_FIRE = COLOR_FIRE_NIGHT
        COLOR_GAUGE = COLOR_GAUGE_NIGHT
    else:
        COLOR_EMPTY = COLOR_EMPTY_DAY
        COLOR_TREE = COLOR_TREE_DAY
        COLOR_FIRE = COLOR_FIRE_DAY
        COLOR_GAUGE = COLOR_GAUGE_DAY

    # Assumes that cells values are in ascending order and paired with its colors
    COLORS = [COLOR_EMPTY, COLOR_TREE, COLOR_FIRE]
    CELLS = [EMPTY, TREE, FIRE]
    NORM, CMAP = get_norm_cmap(CELLS, COLORS)

    # local_grid = moore_n(N_LOCAL, pos, grid, EMPTY)
    pos_fseed = pos_fire[0]

    # Why two titles?
    # The env was registered (benchmark) or
    # The env was directly created (prototype)
    TITLE = title

    def main():
        plt.style.use(FIGSTYLE)
        fig_shape = (12, 16)
        fig = plt.figure(figsize=FIGSIZE)
        fig.suptitle(
            TITLE,
            font=TITLEFONT,
            fontsize=TITLE_SIZE,
            **TITLE_POS,
            color="0.6",
            ha=TITLE_ALIGN,
        )

        # ax_lgrid = plt.subplot2grid(fig_shape, (0, 0), colspan=8, rowspan=10)
        ax_ggrid = plt.subplot2grid(fig_shape, (0, 0), colspan=10, rowspan=10)
        ax_gauge = plt.subplot2grid(fig_shape, (10, 0), colspan=8, rowspan=2)
        ax_counts = plt.subplot2grid(fig_shape, (5, 10), colspan=6, rowspan=6)

        # plot_local(ax_lgrid, local_grid)

        plot_global(ax_ggrid, grid, pos, pos_fseed)

        plot_gauge(ax_gauge, time)

        d = cell_count
        counts = d[EMPTY], d[TREE], d[FIRE]
        plot_counts(ax_counts, *counts)

        return fig

    def plot_local(ax, grid):
        nrows, ncols = grid.shape
        mid_row, mid_col = nrows // 2, nrows // 2

        plot_grid(ax, grid, interpolation="none", cmap=CMAP, norm=NORM)

        markbull = parse_svg_into_mpl(svg_paths.BULLDOZER)
        ax.plot(
            mid_col, mid_row, marker=markbull, markersize=MARKBULL_SIZE, color="1.0"
        )

    def plot_global(ax, grid, pos, pos_fseed):
        # Calculate dousing effect
        dousing_strength = np.minimum(dousing_counts, 3) / 3.0
        water_tint = np.array([0.7, 0.7, 1.0])  # Slight blue tint

        # Create a custom colormap for each cell type with water tint
        colors_with_tint = []
        for color in COLORS:
            # Convert hex to RGB if needed
            if isinstance(color, str):
                color = np.array(plt.matplotlib.colors.to_rgb(color))

            # Apply water tint where there is dousing
            tinted_color = np.where(
                dousing_counts[..., np.newaxis] > 0,
                color * (1 - dousing_strength[..., np.newaxis])
                + water_tint * dousing_strength[..., np.newaxis],
                color,
            )
            colors_with_tint.append(tinted_color)

        # Create new colormap with tinted colors
        NORM, CMAP = get_norm_cmap(CELLS, colors_with_tint)

        # Plot with tinted colormap
        ax.imshow(grid, interpolation="none", cmap=CMAP, norm=NORM)

        # Fire Seed
        markfire = align_marker(parse_svg_into_mpl(svg_paths.FIRE), valign="bottom")
        ax.plot(
            pos_fseed[1],
            pos_fseed[0],
            marker=markfire,
            markersize=MARKFSEED_SIZE,
            color=COLOR_FIRE,
        )

        # Bulldozer Location
        marklocation = align_marker(
            parse_svg_into_mpl(svg_paths.LOCATION), valign="bottom"
        )
        ax.plot(
            pos[1],
            pos[0],
            marker=marklocation,
            markersize=MARKLOCATION_SIZE,
            color="1.0",
        )
        clear_ax(ax)

    def plot_gauge(ax, time):
        HEIGHT_GAUGE = 0.1
        ax.barh(0.0, time, height=HEIGHT_GAUGE, color=COLOR_GAUGE, edgecolor="None")

        ax.barh(
            0.0,
            1.0,
            height=0.15,
            color="None",
            edgecolor="0.86",
        )

        # Mess with x,y limits for aethetics reasons
        INCREASE_LIMS = True

        if INCREASE_LIMS:
            ax.set_xlim(0 - 0.03, 1 + 0.1)  # Breathing room
            ax.set_ylim(-0.4, 0.4)  # Center the bar

        ax.set_xticks([0.0, 1.0])  # Start Time and End Time x ticks

        # Set the CA update symbol
        ax.set_yticks([0])  # Set symbol position
        ax.set_yticklabels(CYCLE_SYMBOL, font=EMOJIFONT, size=CYCLE_SIZE)
        ax.get_yticklabels()[0].set_color("0.74")  # Light gray

        clear_ax(ax, yticks=False)

    def plot_counts(ax, counts_empty, counts_tree, counts_fire):
        counts_total = sum((counts_empty, counts_tree, counts_fire))

        commons = {"x": [0, 1], "width": 0.1}
        pc = "1.0"  # placeholder color

        lv1y = [counts_tree, 0]
        lv1c = [COLOR_TREE, pc]

        lv2y = [0, counts_empty]  # level 2 y axis
        lv2c = [pc, COLOR_EMPTY]  # level 2 colors
        lv2b = lv1y  # level 2 bottom

        lv3y = [0, counts_fire]
        lv3c = [pc, COLOR_FIRE]
        lv3b = [lv1y[i] + lv2y[i] for i in range(len(lv1y))]

        # First Level Bars
        ax.bar(height=lv1y, color=lv1c, **commons)

        # Second Level Bars
        ax.bar(height=lv2y, color=lv2c, bottom=lv2b, **commons)

        # Third Level Bars
        ax.bar(height=lv3y, color=lv3c, bottom=lv3b, **commons)

        # Add wind direction arrow and symbol
        if wind_index is not None:
            arrow_x = 1.8
            arrow_y = counts_total * 0.2

            # Convert wind index to angle (0=North, clockwise)
            angle = (-45 * wind_index) + 90

            ax.set_xlim([0, 3])  # Adjust based on your desired range
            ax.set_ylim([0, counts_total])

            ax.quiver(
                arrow_x,
                arrow_y,
                np.cos(np.radians(angle)),
                np.sin(np.radians(angle)),
                scale=2,  # Increase this value to make arrow shorter
                scale_units="x",
                color="gray",
                alpha=0.5,
            )

            # Add wind symbol
            WIND_SYMBOL = "\U0001F32C"  # Unicode wind face symbol
            ax.text(
                arrow_x - 0.15,  # Slightly left of arrow
                arrow_y - counts_total * 0.43,
                WIND_SYMBOL,
                font=EMOJIFONT,
                size=28,
                color="gray",
                alpha=0.5,
            )

        # Bar Symbols Settings
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels([TREE_SYMBOL, BURNED_SYMBOL], font=EMOJIFONT, size=34)
        # Same colors as bars
        for label, color in zip(ax.get_xticklabels(), [COLOR_TREE, COLOR_FIRE]):
            label.set_color(color)

        # Mess with x,y limits for aethetics reasons
        INCREASE_LIMS = True
        INCREASE_FACTORS = [0.1, 0.3]  # Y axis down, up

        if INCREASE_LIMS:
            # Makes the bars look long & tall, also centers them
            offdown, offup = (
                counts_total * INCREASE_FACTORS[i] for i in range(len(INCREASE_FACTORS))
            )
            ax.set_ylim(
                0 - offdown, counts_total + offup
            )  # It gives breathing room for bars
            ax.set_xlim(-1, 2)  # It centers the bars

        # Grid Settings and Tick settings
        # Show marks each quarter
        ax.set_yticks(np.linspace(0, counts_total, 3, dtype=int))
        # Remove clutter
        clear_ax(ax, xticks=False)
        # Add back y marks each quarter
        ax.grid(axis="y", color="0.94")  # Dim gray

    return main()
