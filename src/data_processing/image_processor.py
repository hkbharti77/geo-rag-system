from dataclasses import dataclass
from typing import Iterator, Tuple, Dict, Any, Optional
from pathlib import Path

try:
	import rasterio
	from rasterio.windows import Window
except Exception:  # pragma: no cover
	rasterio = None
	Window = None


@dataclass
class RasterTiler:
	tile_size: int = 256
	stride: Optional[int] = None

	def tile(self, raster_path: Path) -> Iterator[Tuple[Any, Dict[str, Any]]]:
		if rasterio is None:
			raise RuntimeError("rasterio not available")
		stride = self.stride or self.tile_size
		with rasterio.open(raster_path) as src:
			width, height = src.width, src.height
			for y in range(0, height, stride):
				for x in range(0, width, stride):
					window = Window(col_off=x, row_off=y, width=min(self.tile_size, width - x), height=min(self.tile_size, height - y))
					transform = src.window_transform(window)
					data = src.read(window=window)
					# Compute geographic bbox from window bounds
					minx, miny = transform * (0, window.height)
					maxx, maxy = transform * (window.width, 0)
					meta = {
						"minx": float(minx),
						"miny": float(miny),
						"maxx": float(maxx),
						"maxy": float(maxy),
						"crs": src.crs.to_string() if src.crs else None,
						"width": int(window.width),
						"height": int(window.height),
					}
					yield data, meta
