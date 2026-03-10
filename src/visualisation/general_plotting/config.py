LINEWIDTH = 426.79 / 72.27


class FigSize(tuple):
    def __new__(cls, width_factor: float, height_factor: float):
        width = LINEWIDTH * width_factor
        height = LINEWIDTH * height_factor
        return super().__new__(cls, (width, height))

    def __mul__(self, other: float):
        if not isinstance(other, (float, int)):
            return NotImplemented
        return FigSize(self[0] / LINEWIDTH * other, self[1] / LINEWIDTH * other)

    def __rmul__(self, other: float):
        return self.__mul__(other)


GOLDEN_RATIO = FigSize(1, 1 / 1.6)
SQUARE = FigSize(1, 1)
FLATTER = FigSize(1, 1 / 2)
