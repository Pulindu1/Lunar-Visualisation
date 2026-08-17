# References and data sources

## Data

- **NASA SVS — CGI Moon Kit** (LOLA DEM + LROC colour/illumination composites), the source of the
  `ldem_*` and `lroc_color_poles_*` rasters this project expects:
  <https://svs.gsfc.nasa.gov/4720/>
- **NASA PGDA — LOLA products** (higher-resolution polar DEM products):
  <https://pgda.gsfc.nasa.gov/products/78>
- **M. K. Barker, "LOLA DEMs and lunar landing site analysis"**, NASA NTRS — the calibration and
  terrain-relative-navigation conventions the elevation scaling follows:
  <https://ntrs.nasa.gov/api/citations/20210010668/downloads/Barker_TRNworkshopLOLADEMsMarch2021v1EM.pdf>

## Tooling

- **PyGMT documentation**: <https://www.pygmt.org/latest/>
- **`grdview`** — 3D perspective / displacement rendering:
  <https://docs.generic-mapping-tools.org/latest/grdview>
- **`grdsample`** — grid resampling, including the interpolation options used here:
  <https://docs.generic-mapping-tools.org/dev/grdsample.html>
- **`pygmt.datasets.load_moon_relief`** — reference implementation for lunar relief grids:
  <https://www.pygmt.org/dev/api/generated/pygmt.datasets.load_moon_relief.html>
- **Panel**: <https://panel.holoviz.org/> · **Bokeh**: <https://docs.bokeh.org/>
- **Scientific colour maps (Crameri)**: <https://www.fabiocrameri.ch/colourmaps/>

## Method and technique

- **Bicubic interpolation** — background on the C¹-continuous reconstruction used for the DEM:
  <https://medium.com/@akp83540/bicubic-interpolation-a4ec96593e2b>
- **GMT forum — lunar terrain rendering** (hill-shading and colour choices for high-relief lunar
  terrain): <https://forum.generic-mapping-tools.org/t/lunar-terrain/5304>
- **GMT tutorial — mapping Mars** (planetary-body workflow in PyGMT):
  <https://www.generic-mapping-tools.org/egu22pygmt/mars_maps.html>
- **PyGMT v0.11.0 release thread — other planetary bodies**:
  <https://forum.generic-mapping-tools.org/t/pygmt-v0-11-0-released/4667>
