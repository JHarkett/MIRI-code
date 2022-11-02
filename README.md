# MIRI-code
A collection of scripts to run the JWST calibration pipeline and visulise the data afterwards

For the JWST calibration pipeline:
  Use jwstpipeline_v4.py to combine the dither positions in stage 3
  Use jwstpipeline_singledither_v2.py to leave these dither positions seperate

To map the data using Pat Fry's code, use:
  navigation2.py
  jupkerns.tm
  jwst_kernels
  
To plot spectra from a lat/lon point in the resulting band cubes use:
  plot_pixseperate_v3.py
  
To save each spaxel in a band as a spectral txt file (useful for retrievals) use:
  retrieval_map.py
This will also save latitude, longitude and emission angle (mu) in fits files in the same directory

To visulise the data as a RGB image (using 3 different RGB wavelengths) use:
  plot_pixseperate_v3.py
  visulise_single.py
  build_mosaic_1.py
  build_mosaic_2.py
  visulise_mosaic.py
