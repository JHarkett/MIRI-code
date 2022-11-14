![map_zoom_c2_retrieval_mosaic_EPSC](https://user-images.githubusercontent.com/93939955/199546122-ad248436-b5b6-4249-bf34-2bec04f699e8.png)

# Important

For MIRI MRS stripe correction: When spec3.cube_build.coord_system = 'ifualign': the pipeline aligns in the FOV of the IFU in the x direction. This may result in the 'horizontal' stripes still appearing diagonal. To make the stripes actually be horizontal (required for calibration) the IFU should be aligned in the y direction. To do this, replace ifu_cube.py under the directory path of form:

/opt/anaconda3/envs/jwst/lib/python3.10/site-packages/jwst/cube_build

with the script calibration_pipeline/ifu_align.py

# MIRI-code
A collection of scripts to run the JWST calibration pipeline and visulise the data afterwards


For the JWST calibration pipeline:
  - Use calibration_pipeline/jwstpipeline_v4.py to combine the dither positions in stage 3
  - Use calibration_pipeline/jwstpipeline_singledither_v2.py to leave these dither positions seperate


To map the data using Pat Fry's code, use:
  - navigation/navigation2.py
  - navigation/jupkerns.tm
  - navigation/jwst_kernels


To plot spectra from a lat/lon point in the resulting band cubes use:
  - plot_pixseperate_v3.py


To save each spaxel in a band as a spectral txt file (useful for retrievals) use:
  - retrieval_map.py
This will also save latitude, longitude and emission angle (mu) in fits files in the same directory


To visulise the data as a RGB image (using 3 different RGB wavelengths) use:
  - mosaic_generation/plot_pixseperate_v3.py
  - mosaic_generation/visulise_single.py
  - mosaic_generation/build_mosaic_1.py
  - mosaic_generation/build_mosaic_2.py
  - mosaic_generation/visulise_mosaic.py


To generate spx files (for NEMESIS) using a txt file input (like the ones generated by retrieval_map.py) use:
  - gen_spx.py


To generate contribution functions using the results of a NEMESIS forward model use:
  - plot_contribution.py


![map_zoom_c1_mosaic08](https://user-images.githubusercontent.com/93939955/199546409-93baee96-3da8-49c4-b30d-ec1f49ef909a.png)
