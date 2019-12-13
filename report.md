# OCR assignment report

## Feature Extraction (Max 200 Words)

The feature extraction section is divided into two modes.

Training mode: The training vectors will be reduced to 10 dimensions according to the algorithm, and the model related to the reduced dimension method will be stored. In the training mode, two feature extraction algorithms are used. First, the feature vectors are reduced to 12 dimensions by applying the PCA algorithm. The corresponding eigenvectors are stored in the model. Then the 12-dimensional vectors are applied the feature selection algorithm. By calculating the divergences between the features, 10 features with the largest divergence are selected from the 12 features, and the index of the selected features is stored in the model. At this point we have a 10-dimensional feature vector and stored the corresponding dimension reduce model (with a flag: "trained model").

Classify mode: The vectors to be classified will be reduced in this mode according to the dimension reduce model which generated in the "training mode".

Reason of 2 feature extraction algorithms: PCA cannot obtain vectors with large divergence between classes. Therefore, the vector reduced by PCA is not directly applicable to classification. The feature selection is used here for subsequent processing to obtain features with large divergence.


## Classifier (Max 200 Words)
I used a variant k-Nearest Neighbor classifier to classify the feature vectors where the value of k is dynamic. Before implementation, I did an experiment to classify images with different noise levels using different k values (between 2-1320). I found that when the image is noisier, using larger k values will get better classification results. I performed a regression analysis based on the experimental results, and finally got the equation of the best k value (k = -32.321 * g + 2058.1), where g is the average grayscale in percentage to indicate the degree of noisy. The reason for using average grayscale to indicate noise level is explained in detail in the comments of the code. In the end, the k-Nearest Neighbor classifier will output a series of results for each sample, with the probability from high to low, for subsequent error correction use.

## Error Correction (Max 200 Words)
[Describe and justify the design of any post classification error
correction that you may have attempted.]

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: [Insert percentage here, e.g. 98.1%]
- Page 2: [Insert percentage here, e.g. 98.0%]
- Page 3: [Insert percentage here, e.g. 83.3%]
- Page 4: [Insert percentage here, e.g. 58.1%]
- Page 5: [Insert percentage here, e.g. 38.7%]
- Page 6: [Insert percentage here, e.g. 28.9%]

## Other information (Optional, Max 100 words)
[Optional: highlight any significant aspects of your system that are
NOT covered in the sections above]
