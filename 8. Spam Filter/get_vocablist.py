def get_vocablist():
    vocabulary = []
    with open(r'C:\Users\mackt\Python\Machine Learning\Data\vocab.txt') as f:
        for line in f:
            idx, word = line.split('\t')
            vocabulary.append(word.strip()) # Remove leading and trailing spaces
    return vocabulary

    """
    Reads the fixed vocabulary list in vocab.txt and returns a list of the words.
    Returns
    -------
    list
        The vocabulary list.
    """