"""Deterministic PNG rendering for AirSense V2 figures.

Protocol Phase 1.

Figures are drawn with Pillow rather than a plotting framework for two
reasons: the machine is at 99% disk and no package may be installed, and PNG
output must be byte-identical across runs. Pillow writes no creation-time
metadata, so repeated runs of the same code on the same data produce the same
bytes.

Every public helper takes already-computed numbers. Nothing here reads data,
and nothing here computes a statistic.
"""

from PIL import Image, ImageDraw, ImageFont

WHITE = (255, 255, 255)
BLACK = (17, 17, 17)
GREY = (120, 120, 120)
LIGHT = (222, 222, 222)
BLUE = (31, 90, 148)
TEAL = (17, 128, 128)
ORANGE = (196, 106, 22)
RED = (168, 42, 42)

MARGIN_LEFT = 92
MARGIN_RIGHT = 34
MARGIN_TOP = 74
MARGIN_BOTTOM = 96


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                                     # older Pillow
        return ImageFont.load_default()


FONT_TITLE = _font(19)
FONT_SUB = _font(14)
FONT_AXIS = _font(13)
FONT_TICK = _font(12)


class Figure(object):
    """A single plot area with linear axes."""

    def __init__(self, width=1000, height=600, title="", subtitle="",
                 xlabel="", ylabel=""):
        self.image = Image.new("RGB", (width, height), WHITE)
        self.draw = ImageDraw.Draw(self.image)
        self.width = width
        self.height = height
        self.left = MARGIN_LEFT
        self.right = width - MARGIN_RIGHT
        self.top = MARGIN_TOP
        self.bottom = height - MARGIN_BOTTOM
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)
        self.draw.text((self.left, 20), title, font=FONT_TITLE, fill=BLACK)
        if subtitle:
            self.draw.text((self.left, 46), subtitle, font=FONT_SUB,
                           fill=GREY)
        self.xlabel = xlabel
        self.ylabel = ylabel

    # -- coordinate mapping -------------------------------------------------

    def set_xlim(self, low, high):
        self.xlim = (float(low), float(high) if high > low else float(low) + 1)

    def set_ylim(self, low, high):
        self.ylim = (float(low), float(high) if high > low else float(low) + 1)

    def px(self, value):
        low, high = self.xlim
        return self.left + (value - low) / (high - low) * (self.right
                                                           - self.left)

    def py(self, value):
        low, high = self.ylim
        return self.bottom - (value - low) / (high - low) * (self.bottom
                                                             - self.top)

    # -- decoration ---------------------------------------------------------

    def frame(self):
        self.draw.rectangle([self.left, self.top, self.right, self.bottom],
                            outline=GREY)
        if self.ylabel:
            self.draw.text((14, self.top - 22), self.ylabel, font=FONT_AXIS,
                           fill=BLACK)
        if self.xlabel:
            self.draw.text((self.left, self.bottom + 52), self.xlabel,
                           font=FONT_AXIS, fill=BLACK)

    def yticks(self, values, fmt="%g"):
        for value in values:
            y = self.py(value)
            self.draw.line([self.left, y, self.right, y], fill=LIGHT)
            label = fmt % value
            self.draw.text((self.left - 10 - 7 * len(label), y - 7), label,
                           font=FONT_TICK, fill=BLACK)

    def xticks(self, positions, labels, rotate=False):
        for position, label in zip(positions, labels):
            x = self.px(position)
            self.draw.line([x, self.bottom, x, self.bottom + 4], fill=GREY)
            if rotate:
                self._vertical_text(x, self.bottom + 8, label)
            else:
                self.draw.text((x - 3.4 * len(label), self.bottom + 8), label,
                               font=FONT_TICK, fill=BLACK)

    def _vertical_text(self, x, y, label):
        width = 9 * len(label) + 8
        strip = Image.new("RGB", (width, 20), WHITE)
        ImageDraw.Draw(strip).text((2, 2), label, font=FONT_TICK, fill=BLACK)
        strip = strip.rotate(90, expand=True)
        self.image.paste(strip, (int(x - 10), int(y)))

    # -- marks --------------------------------------------------------------

    def bars(self, centres, heights, width=0.7, colour=BLUE, baseline=0.0):
        for centre, height in zip(centres, heights):
            x0 = self.px(centre - width / 2.0)
            x1 = self.px(centre + width / 2.0)
            y0 = self.py(baseline)
            y1 = self.py(height)
            if y1 > y0:
                y0, y1 = y1, y0
            self.draw.rectangle([x0, y1, x1, y0], fill=colour)

    def line(self, xs, ys, colour=BLUE, width=2):
        points = [(self.px(x), self.py(y)) for x, y in zip(xs, ys)]
        if len(points) > 1:
            self.draw.line(points, fill=colour, width=width, joint="curve")

    def points(self, xs, ys, colour=RED, radius=4):
        for x, y in zip(xs, ys):
            cx, cy = self.px(x), self.py(y)
            self.draw.ellipse([cx - radius, cy - radius, cx + radius,
                               cy + radius], fill=colour)

    def hline(self, value, colour=RED, dash=6):
        y = self.py(value)
        x = self.left
        while x < self.right:
            self.draw.line([x, y, min(x + dash, self.right), y], fill=colour,
                           width=2)
            x += 2 * dash

    def legend(self, entries, x=None, y=None):
        x = self.left + 12 if x is None else x
        y = self.top + 10 if y is None else y
        for label, colour in entries:
            self.draw.rectangle([x, y + 3, x + 12, y + 13], fill=colour)
            self.draw.text((x + 18, y), label, font=FONT_TICK, fill=BLACK)
            y += 18

    def note(self, text, y_offset=26):
        self.draw.text((self.left, self.bottom + y_offset), text,
                       font=FONT_TICK, fill=GREY)

    def save(self, path):
        self.image.save(str(path), "PNG", optimize=True)


def heatmap(path, matrix, row_labels, column_labels, title, subtitle,
            vmin=0.0, vmax=1.0, cell=46, low=(247, 251, 255),
            high=(8, 48, 107), fmt="%.2f", missing_colour=(235, 235, 235),
            legend_label=""):
    """Render a labelled matrix. `matrix` may contain None for absent cells."""
    n_rows = len(row_labels)
    n_columns = len(column_labels)
    left = 168
    top = 118
    width = left + n_columns * cell + 40
    height = top + n_rows * cell + 116
    image = Image.new("RGB", (width, height), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((24, 22), title, font=FONT_TITLE, fill=BLACK)
    draw.text((24, 48), subtitle, font=FONT_SUB, fill=GREY)

    for j, label in enumerate(column_labels):
        strip = Image.new("RGB", (9 * len(label) + 10, 20), WHITE)
        ImageDraw.Draw(strip).text((2, 2), label, font=FONT_TICK, fill=BLACK)
        strip = strip.rotate(90, expand=True)
        image.paste(strip, (left + j * cell + cell // 2 - 10,
                            top - strip.height - 4))

    for i, label in enumerate(row_labels):
        draw.text((left - 12 - 7 * len(label), top + i * cell + cell // 2 - 7),
                  label, font=FONT_TICK, fill=BLACK)

    span = (vmax - vmin) if vmax > vmin else 1.0
    for i in range(n_rows):
        for j in range(n_columns):
            value = matrix[i][j]
            x0 = left + j * cell
            y0 = top + i * cell
            if value is None:
                colour = missing_colour
            else:
                ratio = min(1.0, max(0.0, (float(value) - vmin) / span))
                colour = tuple(int(low[k] + (high[k] - low[k]) * ratio)
                               for k in range(3))
            draw.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=colour,
                           outline=WHITE)
            if value is not None and cell >= 40:
                text = fmt % value
                luminance = (0.299 * colour[0] + 0.587 * colour[1]
                             + 0.114 * colour[2])
                draw.text((x0 + cell // 2 - 3.2 * len(text),
                           y0 + cell // 2 - 7), text, font=FONT_TICK,
                          fill=BLACK if luminance > 140 else WHITE)

    bar_top = top + n_rows * cell + 34
    for k in range(240):
        ratio = k / 239.0
        colour = tuple(int(low[c] + (high[c] - low[c]) * ratio)
                       for c in range(3))
        draw.line([left + k, bar_top, left + k, bar_top + 14], fill=colour)
    draw.rectangle([left, bar_top, left + 239, bar_top + 14], outline=GREY)
    draw.text((left - 4, bar_top + 20), fmt % vmin, font=FONT_TICK, fill=BLACK)
    draw.text((left + 214, bar_top + 20), fmt % vmax, font=FONT_TICK,
              fill=BLACK)
    if legend_label:
        draw.text((left + 258, bar_top), legend_label, font=FONT_TICK,
                  fill=BLACK)
    image.save(str(path), "PNG", optimize=True)
