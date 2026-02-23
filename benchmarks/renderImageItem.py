import typing

import numpy as np
import numpy.typing as npt

import pyqtgraph as pg

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import numba
except ImportError:
    numba = None


def renderQImage(*args, **kwargs):
    imgitem = pg.ImageItem(axisOrder='row-major')
    if 'autoLevels' not in kwargs:
        kwargs['autoLevels'] = False
    imgitem.setImage(*args, **kwargs)
    imgitem.render()

def prime(data, lut, levels):
    shape = (64, 64)
    data = data[:shape[0], :shape[1]]
    kwargs = {}
    if levels is not None:
        kwargs["levels"] = levels
    if lut is not None:
        kwargs["lut"] = lut
    renderQImage(data, **kwargs)  # prime the gpu


class Parameters(typing.NamedTuple):
    sizes: list[tuple[int, int]]
    acceleration: list[str]
    uses_levels: list[bool]
    dtypes: list[npt.DTypeLike]
    channels: list[int]
    lut_lengths: list[npt.DTypeLike | None]

class TimeSuite:
    unit = "seconds"
    param_names = ["size", "acceleration", "use_levels", "dtype", "channels", "lut_length"]
    params = Parameters(
        [
            # (256, 256),               # other sizes useful to test for
            # (512, 512),               # seeing performance scale
            # (1024, 1024),             # but not helpful for tracking history
            # (2048, 2048),             # so we test the most taxing size only
            # (3072, 3072),
            (4096, 4096)
        ],                              # size
        ["numpy"],     # acceleration
        [True, False],                  # use_levels
        ['uint8', 'uint16', 'float32'], # dtype
        [1, 3, 4],                      # channels
        ['uint8', 'uint16', None]       # lut_length
    )
    def __init__(self):
        self.data = np.empty((), dtype=np.uint8)
        self.lut = np.empty((), dtype=np.ubyte)

        # No need to add acceleration that isn't available
        if numba is not None:
            self.params.acceleration.append('numba')

        if cp is not None:
            self.params.acceleration.append('cupy')

        self.levels = None

    def teardown(self, *args, **kwargs):
        # toggle options off
        pg.setConfigOption("useNumba", False)
        pg.setConfigOption("useCupy", False)

    def setup_cache(self) -> dict:
        accelerations = [np]
        if cp is not None:
            accelerations.append(cp)
        cache = {}

        for xp in accelerations:
            cache[xp.__name__] = {"lut": {}, "data": {}}
            random_generator = xp.random.default_rng(42) # answer to everything
            # handle lut caching
            c_map = xp.array([[-500.0, 255.0], [-255.0, 255.0], [0.0, 500.0]])
            for lut_length in self.params.lut_lengths:
                if lut_length is None:
                    continue
                bits = xp.dtype(lut_length).itemsize * 8
                # create the LUT
                lut = xp.zeros((2 ** bits, 4), dtype="ubyte")
                for i in range(3):
                    lut[:, i] = xp.clip(xp.linspace(c_map[i][0], c_map[i][1], 2 ** bits), 0, 255)
                lut[:, -1] = 255
                cache[xp.__name__]["lut"][lut_length] = lut

            # handle data caching
            for dtype in self.params.dtypes:
                cache[xp.__name__]["data"][dtype] = {}
                for channels in self.params.channels:
                    cache[xp.__name__]["data"][dtype][channels] = {}

                    for size in self.params.sizes:
                        size_with_channels = (size[0], size[1], channels) if channels != 1 else size
                        if xp.dtype(dtype) in (xp.float32, xp.float64):
                            data = random_generator.standard_normal(
                                size=size_with_channels,
                                dtype=dtype
                            )
                        else:
                            iinfo = xp.iinfo(dtype)
                            data = random_generator.integers(
                                low=iinfo.min,
                                high=iinfo.max,
                                size=size_with_channels,
                                dtype=dtype,
                                endpoint=True
                            )
                        cache[xp.__name__]["data"][dtype][channels][size] = data
        return cache


    def setup(
            self,
            cache: dict,
            size: tuple[int, int],
            acceleration: str,
            use_levels: bool,
            dtype: npt.DTypeLike,
            channels: int,
            lut_length: typing.Optional[npt.DTypeLike]
    ):
        xp = np
        if acceleration == "numba":
            if numba is None:
                # if numba is not available, skip it...
                raise NotImplementedError("numba not available")
            pg.setConfigOption("useNumba", True)
        elif acceleration == "cupy":
            if cp is None:
                # if cupy is not available, skip it...
                raise NotImplementedError("cupy not available")
            pg.setConfigOption("useCupy", True)
            xp = cp  # use cupy instead of numpy

        # does it even make sense to have a LUT with multiple channels?
        if lut_length is not None and channels != 1:
            raise NotImplementedError(
                f"{lut_length=} and {channels=} not implemented. LUT with multiple channels not supported."
            )

        # skip when the code paths bypass makeARGB
        if acceleration != "numpy":
            if xp.dtype(dtype) == xp.ubyte and not use_levels:
                if lut_length is None:
                    # Grayscale8, RGB888 or RGB[AX]8888
                    raise NotImplementedError(
                        f"{dtype=} and {use_levels=} not tested for {acceleration=} with {lut_length=}"
                    )
                elif channels == 1 and xp.dtype(lut_length) == xp.uint8:
                    # Indexed8
                    raise NotImplementedError(
                        f"{dtype=} and {use_levels=} not tested with {acceleration=} for {channels=} and {lut_length=}"
                    )

            elif xp.dtype(dtype) == xp.uint16 and not use_levels and lut_length is None:
                if channels == 1:
                    # Grayscale16
                    raise NotImplementedError(
                        f"{dtype=} {use_levels=} {lut_length=} and {channels=} not tested for {acceleration=}"
                    )
                elif channels == 4:
                    # RGBA64
                    raise NotImplementedError(
                        f"{dtype=} {use_levels=} {lut_length=} and {channels=} not tested with {acceleration=}"
                    )

        if use_levels:
            if xp.dtype(dtype) == xp.float32:
                self.levels = (-4.0, 4.0)
            elif xp.dtype(dtype) == xp.uint16:
                self.levels = (250, 3000)
            elif xp.dtype(dtype) == xp.uint8:
                self.levels = (20, 220)
            else:
                raise ValueError(
                    "dtype needs to be one of {'float32', 'uint8', 'uint16'}"
                )
        elif xp.dtype(dtype) in (xp.float32, xp.float64):
            # float images always need levels
            raise NotImplementedError(
                f"{use_levels=} {dtype=} is not supported. Float images always need levels."
            )
        else:
            self.levels = None

        if lut_length is None:
            self.lut = None
        else:
            self.lut = cache[xp.__name__]["lut"][lut_length]

        self.data = cache[xp.__name__]["data"][dtype][channels][size]
        if acceleration in {"numba", "cupy"}:
            prime(self.data, self.lut, self.levels)

    def time_test(self, *args, **kwargs):
        kwargs = {}
        if self.lut is not None:
            kwargs["lut"] = self.lut
        if self.levels is not None:
            kwargs["levels"] = self.levels
        renderQImage(self.data, **kwargs)
