
import os
import pandas as pd
import numpy as np
import requests
import time
import re

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    
    req = requests.get(url)
    book = req.text
    
    # clean newlines
    book = re.sub('\r\n','\n',book)
    
    # get the start and end of the book (including title, author, TOC)
    start = re.search('START.*\*\*\*', book).span()[1]
    end = re.search('\*\*\*.*END', book).span()[0]
    book = book[start:end]
    
    # making sure we adhere to the robots.txt of the Project Guttenberg website
    # pause code for 5 seconds
    time.sleep(5)
    
    return book
    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """

    
    book_string = '\x02' + book_string + '\x03'
    # replace \n\n with \x02 and \x03 using regular expressions - these are start and stop characters
    book_string = re.sub('[\n]{2,}',' \x02 \x03 ',book_string)
    book_string = re.sub('\\b', ' ', book_string)
    
    # convert to array
    onespace = ' '.join(book_string.split())
    words = list(onespace.split(' '))

    return words
    
# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        # returns series of probabilities of unique tokens, should all be equal probabilities
        unique_tokens = pd.Series(pd.Series(tokens).unique()).value_counts(normalize = True)

        return unique_tokens
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        token_ser = self.mdl
        words_list = list(words)

        # update probability of "words" appearing the language model by mutiplying existing
        # probability by the probability of each "word" in "words_list"
        probability = 1
        for word in words_list:
            try:
                probability = probability * token_ser[word]
            except:
                return 0
        return probability
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        ret_str = ''
        token_ser = self.mdl
        for i in range(M):
            ret_str += np.random.choice(token_ser.index) + ' '
        return ret_str.strip(' ')

            
# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        # series of probability that each word occurs in given corpus
        ret_tokens = pd.Series(tokens).value_counts(normalize = True)
        return ret_tokens
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        # same thing as uniform language model in terms of calculating final probaility
        token_ser = self.mdl
        words_list = list(words)
        probability = 1
        for word in words_list:
            try:
                probability = probability * token_ser[word]
            except:
                return 0
        return probability
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """

        ret_str = ''
        token_ser = self.mdl
        for i in range(M):
            ret_str += np.random.choice(token_ser.index, p = token_ser.tolist()) + ' '
        return ret_str.strip(' ')
        
    
# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.tokens = tokens

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2: # if N == 2, use unigram model
            self.prev_mdl = UnigramLM(tokens)
        else: # if N > 2, use N-gram model
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        # returns list of N-grams, this is based off tokens keyword
        out = []
        for i in range(len(tokens) - self.N + 1):
            new = tokens[i:i+ self.N]
            out.append(new)
        
        return out
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe indexed on distinct tokens, with three
        columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        ngram_cnt = []
        n1grams = []

        # getting ngram count in order to compute conditional prob. of ngram
        for token in self.ngrams:  
            cnt = self.ngrams.count(token)
            ngram_cnt.append(cnt)
            n1gram = token[:-1]
            n1grams.append(n1gram)
            
        # getting n1gram count in order to compute conditional prob. of ngram
        n1gram_cnt = []
        for token2 in n1grams:
            cnt2 = n1grams.count(token2)
            n1gram_cnt.append(cnt2)
        
        # stitching together output dataframe with columns "ngram, "n1gram"; indicies are distinct tokens
        df = pd.DataFrame(columns=['ngram', 'n1gram'], index = self.ngrams)
        df['ngram'] = self.ngrams
        df['n1gram'] = n1grams
        df['ngram_cnt'] = ngram_cnt
        df['n1gram_cnt'] = n1gram_cnt
        df['prob'] = df['ngram_cnt'] / df['n1gram_cnt']
        df = df.drop(['ngram_cnt', 'n1gram_cnt'], axis=1)
        df = df.loc[~df.index.duplicated(keep='first')]

        return df
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """

        # get the prob for the unigram (first word in words)
        P_one = self.tokens.count(words[0]) / len(self.tokens)
        total_prob = P_one

        # loop through the different kind of Ngrams we need to create
        for i in range(1, self.N):
            # creates first Ngram with tokens
            newNgrams = NGramLM(N = i + 1, tokens = self.tokens)
            # creates secondary Ngram with words so that we can use its ngrams attribute
            ngram2 = NGramLM(N = i + 1, tokens = tuple(words))
            # only passes if it is the Nth gram (if N = 3, the trigram)
            if (i + 1) == self.N:
                # ngram2.ngram is a list of the ngrams that we want to find in newNgrams.mdl
                for j in ngram2.ngrams:
                    # checks to see if it is in the newNgrams.mdl (if not return 0)
                    if j in newNgrams.mdl.index:
                        # get prob and multiply onto total
                        prob = newNgrams.mdl.loc[[j]]['prob'][0]
                        total_prob = total_prob * prob
                    else:
                        return 0
            else:
                if ngram2.ngrams[0] in newNgrams.mdl.index:
                        prob = newNgrams.mdl.loc[[ngram2.ngrams[0]]]['prob'][0]
                        total_prob = total_prob * prob
                else:
                    return 0

        return total_prob


    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        # had to create this dummy instance because ran into weird error where 
        # I could not access the tokens attribute from Ngram when using sample_helper
        dummy = NGramLM(self.N, self.tokens)
        base = dummy.sample_helper(M)

        # Transform the tokens to strings
        out = ' '.join(base)

        return out

    def sample_helper(self, M):
            # Use a helper function to generate sample tokens of length `length`
            base = ['\x02']
            """
            bigram = NGramLM(N = 2, tokens = self.tokens)
            base_df = bigram.mdl[bigram.mdl['n1gram'] == tuple(base)]
            if len(base_df) == 0:
                base.append('\x03')
            else:
                nxt_wrd = np.random.choice(base_df['ngram'], p=base_df['prob'])
                base = list(nxt_wrd)
            nxt_wrd = base
            """
            # really messy helper function
            # essentially, keep getting next word until "length" is hit, then stop
            # ran into problem where I couldn't access tokens from NGram model
            nxt_wrd = base
            for i in range(1, self.N):
                if len(base) >= M + 1:
                    return base
                if len(nxt_wrd) > self.N:
                    nxt_wrd = nxt_wrd[:-self.N]
                else:
                    nxt_wrd = base[0:i]

                if (i + 1) == self.N:
                    for j in range(len(base), M + 1):
                        nxt_wrd = base[-self.N + 1:]
                        nextGram = NGramLM(N = i + 1, tokens = self.tokens)
                        df = nextGram.mdl

                        base_df = df[df['n1gram'] == tuple(nxt_wrd)]
                        if len(base_df) == 0:
                            base.append('\x03')
                        else:
                            nxt_gram = np.random.choice(base_df['ngram'], p=base_df['prob'])
                            to_add = nxt_gram[-1]
                            base.append(to_add)
                else:
                    nextGram = NGramLM(N = i + 1, tokens = self.tokens)
                    df = nextGram.mdl

                    base_df = df[df['n1gram'] == tuple(nxt_wrd)]
                    if len(base_df) == 0:
                        base.append('\x03')
                    else:
                        nxt_gram = np.random.choice(base_df['ngram'], p=base_df['prob'])
                        to_add = nxt_gram[-1]
                        base.append(to_add)

            return base

def predict_next_word(NGram_instance, tokens):
    '''Build a predictor that predicts the most likely word to follow a given sentence. 
    The predictions will be the maximum likelihood estimate given by your N-gram model. 
    Recall that your N-gram model contains the probabilities that a token follows an (n-1)-gram; 
    your predictor will pick the token with the highest probability of occurring.'''

    ngram_df = NGram_instance.mdl # gives dataframe that contains probabilities of ngram and n1gram
    max_prob = ngram_df[ngram_df['n1gram'] == tokens]['prob'].max()  # gets max prob of ngram

    return ngram_df[(ngram_df['n1gram'] == tokens) & (ngram_df['prob'] == max_prob)]['ngram']


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
