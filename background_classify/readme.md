These files are used to classify the background position (upper or lower) in the cloud camera images. The classification is based on a binary classifier trained on manually collected data. 

binary_2023_two_class_tokenization.py is used to obtain the CLS token of all the images in traing and test set. 

binary_2023_two_class_linear.py is used to train and test a linear classifier on top of the CLS token obtained from binary_2023_two_class_tokenization.py.

binary_2023_two_class_inference.py is used to do inference on all the images after 2023-09-27 using the trained linear classifier from binary_2023_two_class_linear.py. The output is a csv file containing the time, class and probability for each image. Merged results are data/bkg_mask/bkg_binary_classification_merged.csv.