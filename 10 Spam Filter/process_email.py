import re
from nltk import PorterStemmer

from get_vocablist import get_vocablist

def split(delimiters, string, maxsplit=0):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, string, maxsplit)

def process_email(email_contents):
    # vocab
    vocab_list = get_vocablist() # Array of words in vocab.txt

    # email
    email_contents = email_contents.lower() # All letters lowercase
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    email_contents = re.sub('[0-9]+', 'number', email_contents) # Replace Numbers with 'number'
    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents) # Replace html with 'httpaddr'
    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents) # Replace email address with 'emailaddr'
    email_contents = re.sub('[$]+', 'dollar', email_contents) # Replaces $ with 'dollar'

    words = split(""" @$/#.-:&*+=[]?!(){},'">_<;%\n\r""", email_contents) # Any of the characters """ @$/#.-:&*+=[]?!(){},'">_<;%\n\r""" replace with ''
    word_indices = []
    email = []
    stemmer = PorterStemmer() # All words subed with root word (buses -> bus)
    for word in words:
        word = re.sub('[^a-zA-Z0-9]', '', word) # Ignore '' terms
        if word == '':
            continue
        word = stemmer.stem(word)
        email.append(word)
        if word in vocab_list:
            idx = vocab_list.index(word)
            word_indices.append(idx)
    email_sentence = " ".join(email)

    return word_indices, words, email_sentence

    # Example
    # Anyone is the first word in Email1.txt
    # Anyon is word 86 listed in vocab.txt
    # Prints email without punctuation with words found in vocab.txt
    # Returns (word #)-1 in word_indices

    """
    Preprocesses a the body of an email and returns a list of word indices.
    Parameters
    ----------
    email_contents : string
        The email content.
    Returns
    -------
    list
        A list of word indices.
    """


