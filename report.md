# OCR assignment report

## Feature Extraction (Max 200 Words)

The feature extraction section is divided into two modes.

Training mode: The training vectors will be reduced to 10 dimensions according to the algorithm, and the model related to the reduced dimension method will be stored. In the training mode, two feature extraction algorithms are used. First, the feature vectors are reduced to 12 dimensions by applying the PCA algorithm. The corresponding eigenvectors are stored in the model. Then the 12-dimensional vectors are applied the feature selection algorithm. By calculating the divergences between the features, 10 features with the largest divergence are selected from the 12 features, and the index of the selected features is stored in the model. At this point we have a 10-dimensional feature vector and stored the corresponding dimension reduce model (with a flag: "trained model").

Classify mode: The vectors to be classified will be reduced in this mode according to the dimension reduce model which generated in the "training mode".

Reason of 2 feature extraction algorithms: PCA cannot obtain vectors with large divergence between classes. Therefore, the vector reduced by PCA is not directly applicable to classification. The feature selection is used here for subsequent processing to obtain features with large divergence.


## Classifier (Max 200 Words)
I used a variant k-Nearest Neighbor classifier to classify the feature vectors where the value of k is dynamic. Before implementation, I did an experiment to classify images with different noise levels using different k values (between 2-1320). I found that when the image is noisier, using larger k values will get better classification results. I performed a regression analysis based on the experimental results, and finally got the equation of the best k value (k=-32.321*g+2058.1), where g is the average grayscale in percentage to indicate the degree of noisy. The reason for using average grayscale to indicate noise level is explained in detail in the comments of the code. In the end, the k-Nearest Neighbor classifier will output a series of results for each sample, with the probability from high to low, for subsequent error correction use.

## Error Correction (Max 200 Words)
In this part, I use boundary boxes to distinguish words and use a word list to check whether the resulting words are valid. Since my classifier can output multiple results for each sample, when the primary result cannot form a valid word, it can try to use the secondary result to form a new word and verify it. However, it would take a lot of time to try all combinations of words, so I converted problem to the shortest valid path problem. I attached a corresponding cost to each result of each sample, where the less the probability, the larger the cost. Here I use Dijkstra's algorithm. Starting from the least costly path, and verify one by one until it finds a valid word. If punctuation is encountered at the beginning or end of a word, it will be removed and then match to the word list. I also set a maximum cost to prevent spending too much time on a word and to prevent finding valid but too different words. Another limit is the number of closed nodes, which is simply limit the time spent on the long words to prevent timeouts.

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:
- Page 1: 97.7%
- Page 2: 98.4%
- Page 3: 92.5%
- Page 4: 77.7%
- Page 5: 65.5%
- Page 6: 53.1%

## Other information (Optional, Max 100 words)
In the feature extraction, I have tried to use Gaussian low-pass filter and binarization algorithm on the original data to reduce noise to improve the accuracy. However, this solution will take a lot of time to process and not improve the accuracy much, Therefore disabled in final version.

Judging the noise level by the average grayscale is theoretically only valid in this typeset page. It is calculated and stored in the queue when the page is loaded, and it is read before the classification starts. The detailed principle is explained in the source code comments (around line 95).
