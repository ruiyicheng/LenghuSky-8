Jia25_ensemble.py is used to do astrometric calibration from the preprocessed images. It would resolve the WCS parameters for all the images in a given folder.

calibrate_and_save.py is used to aggregrate the calibration results from Jia25_ensemble to give a global fitting result for given time duration.

allsky_mapper.py use the results of calibrate_and_save.py to map pixel coordinates to sky coordinates and vice versa.