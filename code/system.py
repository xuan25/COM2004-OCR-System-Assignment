"""Dummy classification system.

Skeleton code for a assignment solution.

To make a working solution you will need to rewrite parts
of the code below. In particular, the functions
reduce_dimensions and classify_page currently have
dummy implementations that do not do anything useful.

version: v1.0
"""
import numpy as np
import scipy.linalg
import utils.utils as utils
import sys
import math

#region Global options & Global variable

# <General options>

# Decide on and off of progress output (Set False to increase speed)
output = False
# Ignore n leading principal components for better classification
pca_ignore_n = 0
# Dimensions after PCA (process before feature selection)
pca_dimensions = 12
# Dimensions of feature selection output (should be 10)
feature_selection_dimensions = 10
# Value k in the k-Nearest Neighbour classification (If k = 1, it will perform a nearest neighbor classification)
nearest_neighbour_k = 5
# How many predictions for each charactor will output by the variant of k-Nearest Neighbour (from high posibility to low posibility)
n_nearest_neighbour = 3
# The maximum cost for the Dijkstra's algorithm in error correction (pervent words that are too far away)
error_correction_cost_limit = 10
# The maximum closed nodes for the Dijkstra's algorithm in error correction (If spend too much time on one word, then give up the error correction, maybe it is too long or not exist in the word list)
error_correction_nodes_limit = 800

# <Additional options>

# Auto adjust k value based on its noise level (The noise level will be judged by the gray mean scale)
# (nearest_neighbour_k will not valid anymore when auto_adjust_k is True)
auto_adjust_k = True

# <Unstable features>

# Preprocessing images using Gaussian filter & Binarization
# This can improve the recognition accuracy, but it will increase the time consuming
# TODO : Turn off when submitting assignment
use_gaussian_filter = False
# TODO : Turn off when submitting assignment
use_binarization = False


# <Global variable>

# Save grayscale means for each page to perform auto_adjust_k
# 
# According to the method of generate noisy image - 
# "Each pixel value (in the range 0 to 255) was randomly perturbed by adding on an amount uniformly distributed between [-x, x]."
# That is, each pixel will be apply on positive or negative offset witch is uniformly distributed. It seems the overall grayscale mean should not change.
# However, there is a additional rule - "After noise was added values above 255 were clipped to 255 and negative values were set to 0."
# In the testing images, the amount of white pixels (value=255) is more than the amount of black pixels (value=0) overall.
# Therefore, more positive offsets will be clipped which are applied on the white pixels, and more negative offsets will be fully applied,
# so the overall grayscale mean will decrease (visually darker).
# Because the more noisy sample will use larger x value, the overall grayscale mean will decrease even more.
# So I measure the noise level by calculating the mean grayscale value。
gray_mean_queue = []

#endregion

#region Loading & metadata

def get_bounding_box_size(images):
    """Compute bounding box size given list of images."""
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    return height, width

def images_to_feature_vectors(images, bbox_size=None):
    """Reformat characters into feature vectors.

    Takes a list of images stored as 2D-arrays and returns
    a matrix in which each row is a fixed length feature vector
    corresponding to the image.abs

    Params:
    images - a list of images stored as arrays
    bbox_size - an optional fixed bounding box size for each image
    """

    # If no bounding box size is supplied then compute a suitable
    # bounding box by examining sizes of the supplied images.
    if bbox_size is None:
        bbox_size = get_bounding_box_size(images)

    bbox_h, bbox_w = bbox_size
    nfeatures = bbox_h * bbox_w
    fvectors = np.empty((len(images), nfeatures))
    for i, image in enumerate(images):
        padded_image = np.ones(bbox_size) * 255
        h, w = image.shape
        h = min(h, bbox_h)
        w = min(w, bbox_w)
        padded_image[0:h, 0:w] = image[0:h, 0:w]
        fvectors[i, :] = padded_image.reshape(1, nfeatures)
    return fvectors

def process_training_data(train_page_names):
    """Perform the training stage and return results in a dictionary.

    Params:
    train_page_names - list of training page names
    """
    print('Building word set')
    wordlist = list()
    with open('corncob_lowercase.txt','r') as f:
        for line in f:
            wordlist.append(line.strip('\n'))
    wordlist.extend(['i\'ll', 'i\'m', 'i\'d', 'you\'re', 'don\'t', 'didn\'t', 'haven\'t', 'who\'s', 'there\'s', 'it\'s'])

    print('Reading data')
    images_train = []
    labels_train = []
    for page_name in train_page_names:
        images_train = utils.load_char_images(page_name, images_train)
        labels_train = utils.load_labels(page_name, labels_train)
    labels_train = np.array(labels_train)

    print('Extracting features from training data')
    bbox_size = get_bounding_box_size(images_train)
    fvectors_train_full = images_to_feature_vectors(images_train, bbox_size)

    model_data = dict()
    model_data['word_list'] = wordlist
    model_data['labels_train'] = labels_train.tolist()
    model_data['bbox_size'] = bbox_size

    print('Reducing to 10 dimensions')
    fvectors_train = reduce_dimensions(fvectors_train_full, model_data)

    model_data['fvectors_train'] = fvectors_train.tolist()
    return model_data

def load_test_page(page_name, model):
    """Load test data page.

    This function must return each character as a 10-d feature
    vector with the vectors stored as rows of a matrix.

    Params:
    page_name - name of page file
    model - dictionary storing data passed from training stage
    """
    bbox_size = model['bbox_size']
    images_test = utils.load_char_images(page_name)

    # Calculate gray mean for noisy detection
    gary_mean = 0
    pixel_count = 0
    for i in range(len(images_test)):
        gary_mean += np.sum(images_test[i])
        pixel_count += images_test[i].shape[0] * images_test[i].shape[1]
    gary_mean /= pixel_count
    gray_mean_queue.append(gary_mean / 255)

    fvectors_test = images_to_feature_vectors(images_test, bbox_size)
    # Perform the dimensionality reduction.
    fvectors_test_reduced = reduce_dimensions(fvectors_test, model)
    return fvectors_test_reduced

#endregion

#region Reduce dimensions

def reduce_dimensions(feature_vectors_full, model):
    """Reduce vectors to 10 dimensions

    Params:
    feature_vectors_full - feature vectors stored as rows
       in a matrix
    model - a dictionary storing the outputs of the model
       training stage
    """

    # Gaussian low-pass filter for denoising
    if use_gaussian_filter:
        if output:
            print('Gaussian filter...')
        for i in range(feature_vectors_full.shape[0]):
            if output:
                print('\r', i, '/', feature_vectors_full.shape[0], end='')
            feature_vectors_full[i] = gaussian_filter(feature_vectors_full[i], model['bbox_size'])
        print()
    
    # Binarization after Gaussian filtering
    if use_binarization:
        if output:
            print('Binarization...')
        for i in range(feature_vectors_full.shape[0]):
            if output:
                print('\r', i, '/', feature_vectors_full.shape[0], end='')
            threshold = 128
            feature_vectors_full[i] = np.where(feature_vectors_full[i] > threshold, 255, 0)
        print()

    # Reduce dimensions
    if(('trained' in model.keys()) and (model['trained'] == True)):
        # Load model and process test data
        # PCA
        flipped_eigenvectors = np.array(model['flipped_eigenvectors'])
        trained_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), flipped_eigenvectors)

        # Feature selection if required
        if('selected_features' in model.keys()):
            selected_features = np.array(model['selected_features'])
            trained_data = trained_data[:, selected_features]
        return trained_data
    else:
        # Generate model form traning data
        # PCA
        if output:
            print('PCA...', '(Reducing to', pca_dimensions, 'dimensions)')
        flipped_eigenvectors = get_flipped_eigenvectors_for_pca(feature_vectors_full, pca_dimensions+pca_ignore_n)[:, pca_ignore_n:]
        trained_data = np.dot((feature_vectors_full - np.mean(feature_vectors_full)), flipped_eigenvectors)
        model['flipped_eigenvectors'] = flipped_eigenvectors.tolist()

        # Feature selection if required
        if(feature_selection_dimensions < pca_dimensions):
            if output:
                print('Feature Selection...', '(Reducing to', feature_selection_dimensions, 'dimensions)')
            selected_features = select_features(trained_data, np.array(model['labels_train']), feature_selection_dimensions)
            trained_data = trained_data[:, selected_features]
            model['selected_features'] = selected_features.tolist()

        model['trained'] = True
        return trained_data

def get_flipped_eigenvectors_for_pca(feature_vectors, dimensions):
    """Compute a series of flipped eigenvectors of feature vectors
    
    feature_vectors - feature vectors, each row is a sample, each column is a feature
    dimensions - the number of flipped eigenvectors calculated by this function, also as the output dimensions of PCA
    
    returns: flipped_eigenvectors - a array of flipped eigenvectors
    """
    covariance_matrix = np.cov(feature_vectors, rowvar=0)
    rows = covariance_matrix.shape[0]
    eigenvalues, eigenvectors = scipy.linalg.eigh(covariance_matrix, eigvals=(rows - dimensions, rows - 1))
    flipped_eigenvectors = np.fliplr(eigenvectors)
    return flipped_eigenvectors

def divergence(class1, class2):
    """Compute a vector of 1-D divergences
    
    class1 - data matrix for class 1, each row is a sample
    class2 - data matrix for class 2
    
    returns: d12 - a vector of 1-D divergence scores
    """

    # Compute the mean and variance of each feature vector element
    m1 = np.mean(class1, axis=0)
    m2 = np.mean(class2, axis=0)
    v1 = np.var(class1, axis=0)
    v2 = np.var(class2, axis=0)

    # Plug mean and variances into the formula for 1-D divergence.
    # (Note that / and * are being used to compute multiple 1-D
    #  divergences without the need for a loop)
    d12 = 0.5 * (v1 / v2 + v2 / v1 - 2) + 0.5 * ( m1 - m2 ) * (m1 - m2) * (1.0 / v1 + 1.0 / v2)

    return d12

def select_features(feature_vectors, labels, dimensions):
    """Find a number of features with the highest divergence between a number of classes
    
    feature_vectors - feature vectors, each row is a sample, each column is a feature
    labels - an array of labels, corresponding to each row of the feature_vectors, to distinguish classes
    dimensions - the number of features selected by this function, also as the output dimensions of feature selection
    
    returns: features - an array of features with the highest divergence
    """
    nfeatures = feature_vectors.shape[1]
    classes_label = np.array(list(set(labels)))
    classes_count = classes_label.shape[0]

    # Calculate divergences sum of features between each pair of classes
    divergences = np.zeros(nfeatures)
    for i in range(classes_count):
        if output:
            print('\r', 'Selecting features...', i+1, '/', classes_count, end='')
        for j in range(i+1, classes_count):
            data1 = feature_vectors[labels == classes_label[i], :]
            data2 = feature_vectors[labels == classes_label[j], :]
            if((not (data1.shape[0] < 2)) and (not (data2.shape[0] < 2))):
                divergences += divergence(data1, data2)
    
    # Select features with the highest divergence
    sorted_indexes = np.argsort(-divergences)
    features = sorted_indexes[0:dimensions]
    if output:
        print()
        print('Selected features: ', features)
    return features

def gaussian_filter(feature_vector, bbox_size):
    """Apply gaussian filter to an image
    
    feature_vector - a 1-D vector represents an image
    bbox_size - size of the border box
    
    returns: new_features - a 1-D vector represents an image after filtered
    """

    radius = 1 # Template radius
    sigema = 1 # Sigema

    # Reshape to 2x2
    image_value = feature_vector.reshape(bbox_size[0], bbox_size[1])

    # Add white edges for edge sampling
    image_value = np.pad(image_value, radius, 'constant', constant_values=255)

    # Generate filter template
    side_length = radius*2 + 1
    result = np.zeros((side_length, side_length))
    for i in range(side_length):
        for j in range(side_length):
            x = i-radius
            y = j-radius
            result[i, j]= 1/(2*math.pi*sigema*sigema) * math.exp(-(x*x+y*y)/(2*sigema*sigema))
    all = result.sum()  
    template = result / all   

    # 2x2 output buffer
    image_value_new = np.zeros((bbox_size[0], bbox_size[1]))

    # Apply the filter to the image
    height = image_value.shape[0]
    width = image_value.shape[1]
    for i in range(radius, height-radius):
        for j in range(radius, width-radius):
            t=image_value[i-radius:i+radius+1, j-radius:j+radius+1]
            a= np.multiply(t, template)
            image_value_new[i-radius, j-radius] = a.sum()

    # Image.fromarray(image_value_new).show()

    # Reshape to a vector
    new_features = image_value_new.reshape(bbox_size[0]*bbox_size[1])
    return new_features

#endregion

#region Classifier

def classify_page(page, model):
    """Classifier.

    parameters:

    page - matrix, each row is a feature vector to be classified
    model - dictionary, stores the output of the training stage

    returns: label - the prediction labels of the classifier on the test data
    """
    # Use different k values for different noise levels
    if auto_adjust_k:
        global nearest_neighbour_k
        gray_mean = gray_mean_queue.pop(0)
        k = round(-32.321*100*gray_mean+2058.1)
        if(k < 2):
            k = 2
        if(k > 200):
            k = 200
        nearest_neighbour_k = int(k)

        if output:
            print('Auto adjust', 'k =', nearest_neighbour_k)

    # Run classify
    label = classify(np.array(model['fvectors_train']), np.array(model['labels_train']), page, nearest_neighbour_k, n_nearest_neighbour)
    return label

def classify(train, train_labels, test, k, n):
    """k-Nearest neighbour classification.(Variant: Return the first n predictions for each element)
    
    train - data matrix storing training data, one sample per row
    train_label - a vector storing the training data labels
    test - data matrix storing the test data
    test_lables - a vector storing the test data labels for evaluation
    k - value k in the k-Nearest Neighbour classification 
        (If k = 1, it will perform a nearest neighbor classification)
             
    returns: label - the prediction labels of the classifier on the test data
    """
    if output:
        print('Processing classify...')

    # Calculate diatances
    x= np.dot(test, train.transpose())
    modtest = np.sqrt(np.sum(test * test, axis=1))
    modtrain = np.sqrt(np.sum(train * train, axis=1))
    dist = x / np.outer(modtest, modtrain.transpose()); # cosine distance

    if k == 1:
        # Nearest neighbour classification
        nearest = np.argmax(dist, axis=1)
        label = train_labels[nearest]
        return label
    else:
        # k-Nearest neighbour classification
        knearest = np.argsort(-dist, axis=1)[:, :k]
        klabel = train_labels[knearest]

        # Deduplicated labels
        classes_label = list(set(train_labels))
        classes_label.sort(key=(lambda l : train_labels[train_labels == l].shape[0]))
        classes_label = np.array(classes_label)
        
        label = []

        # classify each char
        for i in range(knearest.shape[0]):
            if output and (i%100 == 0):
                print('\r', i, '/', knearest.shape[0], end='')
            labelsum = np.zeros(classes_label.shape[0])
            for j in range(knearest.shape[1]):
                lindex = np.argwhere(classes_label == klabel[i, j])
                labelsum[lindex] += dist[i, knearest[i, j]]
            label.append(classes_label[np.argsort(-labelsum)[:n]])
        if output:
            print('\r', knearest.shape[0], '/', knearest.shape[0], end='')
            print()
        return np.array(label)

#endregion

#region Error correction

class Node:
    def __init__(self, pos, path, cost):
        self.pos = pos
        self.path = path
        self.cost = cost

def correct_errors(page, labels, bboxes, model):
    """Error correction.

    parameters:

    page - 2d array, each row is a feature vector to be classified
    labels - the output classification label for each feature vector
    bboxes - 2d array, each row gives the 4 bounding box coords of the character
    model - dictionary, stores the output of the training stage
    """
    if(len(labels.shape) == 1):
        print('Error correction skipped (k=1)')
        return labels

    if output:
        print('Processing error correction...')

    # Make a word set
    wordset = set(model['word_list'])

    # Ready for error correction
    total_length = labels.shape[0]
    startIndex = 0
    output_labels = []

    # Try to split words based on border boxes
    for i in range(bboxes.shape[0]):
        if(i == total_length-1):
            correct(labels, startIndex, i, wordset, output_labels)
            startIndex = i+1
        elif(abs(bboxes[i+1][0] - bboxes[i][2]) > 6):
            correct(labels, startIndex, i, wordset, output_labels)
            startIndex = i+1
        if output:
            print('\r', startIndex, '/', total_length, end='')
    if output:
        print()
    output_labels_array = np.array(output_labels)
    return output_labels_array

def correct(labels, start, end, wordset, output_labels):
    """Correct the word using Dijkstra's algorithm

    parameters:

    labels - the full label array, each element is a array of perdictions for the charactor
    start - the start position of the word
    end - the end position of the word
    wordset - the knownd word set
    output_labels - the list of corrected labels, the result will append to this list
    """
    # Dijkstra's algorithm
    predictions = labels.shape[1]
    word_length = end - start
    open_list = []
    # closed_list = []
    closed_count = 0

    open_list.append(Node(-1, [], 0))

    while len(open_list) > 0:
        node = open_list[0]

        if((node.cost > error_correction_cost_limit) or (closed_count > error_correction_nodes_limit)):
            # Over limits, give up
            output_labels.extend(labels[start:end+1, 0])
            return
        
        if(node.pos == word_length):
            # Length matched, verify
            predict_word = ''.join(node.path)
            if(predict_word.replace('\'\'', '').strip(',.?!') in wordset):
                # Matched, success
                output_labels.extend(node.path)
                return
        
        # Finish the node, find successor
        # closed_list.append(node)
        open_list.remove(node)
        closed_count += 1
        next_pos = node.pos+1
        if(next_pos <= word_length):
            for i in range(predictions):
                insert_node(open_list, Node(next_pos, node.path+[labels[start+next_pos, i]], node.cost+i+1))

    # No result found, give up
    output_labels.extend(labels[start:end+1, 0])
    return

def insert_node(node_list, node):
    """Insert a node to the node list. make the cost in ascending order.

    parameters:

    node_list - the list to be instert a node
    node - the node to be instert into the list
    """
    for i in range(len(node_list)):
        if node.cost < node_list[i].cost:
            node_list.insert(i, node)
            return
    node_list.append(node)

#endregion
