import string
import nltk
import os.path as path
import re
import numpy as np
from tqdm import tqdm
from nltk.corpus import cmudict
from curses.ascii import isdigit

def syllable_count(text):
    """
    Calculates number of syllables in a text
    I/P - a text
    O/P - number of syllable words
    """
    exclude = list(string.punctuation)
    count = 0
    vowels = 'aeiouy'
    text = text.lower()
    text = "".join(x for x in text if x not in exclude)

    if text is None:
        return 0
    elif len(text) == 0:
        return 0
    else:
        if text[0] in vowels:
            count += 1
        for index in range(1, len(text)):
            if text[index] in vowels and text[index - 1] not in vowels:
                count += 1
        if text.endswith('e'):
            count -= 1
        if text.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        count = count - (0.1 * count)
        return count


def flesch_kincaid_grade_level(text, token_count):
    """
    Takes a text and returns its FK Grade Level
    :param text: A string text
    :param token_count: the number of tokens in the text
    :return: FK Grade Level
    """

    # Extracted of textstat 0.3.1 which is only available for python 2 (https://github.com/shivam5992/textstat)
    # Alternatives:
    #   http://sylli.sourceforge.net/cmd_clips.html
    #   "Fundamentals of Python: First Programs" Chapter 4.6.4
    #   https://codegolf.stackexchange.com/questions/47322/how-to-count-the-syllables-in-a-word
    #   https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests

    def avg_syllables_per_word(text, token_count):
        syllable = syllable_count(text)
        if token_count > 0:
            return float(syllable) / float(token_count)
        else:
            return 0

    if len(nltk.sent_tokenize(text)) <= 0 or token_count <= 0:
        return 0

    ASL = float(token_count / len(nltk.sent_tokenize(text)))  # avg sentence length
    ASW = avg_syllables_per_word(text, token_count)
    FKRA = float(0.39 * ASL) + float(11.8 * ASW) - 15.59
    return FKRA


def flesch_kincaid_reading_ease(text, token_count):
    """
    Takes a text and returns its FK Reading Ease
    :param text: A string text
    :param token_count: the number of tokens in the text
    :return: FK Reading Ease
    """

    # Partly extracted of textstat 0.3.1 which is only available for python 2 (https://github.com/shivam5992/textstat)

    def avg_syllables_per_word(text, token_count):
        syllable = syllable_count(text)
        if token_count > 0:
            return float(syllable) / float(token_count)
        else:
            return 0

    if len(nltk.sent_tokenize(text)) <= 0 or token_count <= 0:
        return 0

    ASL = float(token_count / len(nltk.sent_tokenize(text)))  # avg sentence length
    ASW = avg_syllables_per_word(text, token_count)
    FKRA = 206.835 - float(1.015 * ASL) - float(84.6 * ASW)
    return FKRA


def gunning_fog_index(text, token_count):
    # Extracted of textstat 0.3.1 which is only available for python 2 (https://github.com/shivam5992/textstat)
    """
    Calculates the gunning fog index
    :param text: A string text
    :param token_count: the number of tokens in the text
    :return: Gunning fog index
    """

    def get_easy_word_set():
        lexicon_path = "%s/../../data/lexicons/gunning_fog/" % (path.dirname(path.dirname(path.abspath(__file__))))
        easy_word_set = set()
        with open(lexicon_path + 'easy_words.txt', 'r') as f:
            for row in f.readlines():
                easy_word_set.add(row.rstrip().lower())
        return easy_word_set

    def difficult_words(text):
        text_list = nltk.word_tokenize(text)
        diff_words_set = set()
        for value in text_list:
            if value not in easy_word_set:
                if syllable_count(value) > 1:
                    if value not in diff_words_set:
                        diff_words_set.add(value)

        return len(diff_words_set)

    if len(nltk.sent_tokenize(text)) <= 0 or token_count <= 0:
        return 0

    easy_word_set = get_easy_word_set()

    ASL = float(token_count / len(nltk.sent_tokenize(text)))  # avg sentence length

    per_diff_words = ((difficult_words(text) / token_count) * 100)
    grade = 0.4 * (ASL + per_diff_words)

    return grade


def smog_index(text):
    """
    Takes a text and returns its SMOG index
    :param text: A string text
    :return: SMOG Index
    """

    # Partly extracted of textstat 0.3.1 which is only available for python 2 (https://github.com/shivam5992/textstat)

    def polysyllabcount(text):
        # return the number of words with 3 or more syllables
        count = 0
        for word in nltk.word_tokenize(text):
            wrds = syllable_count(word)  # check how many syllables word has
            if wrds >= 3:
                count += 1
        return count

    sent_count = len(nltk.sent_tokenize(text))
    if sent_count >= 30:
        poly_syllab = polysyllabcount(text)
        SMOG = (1.043 * ((poly_syllab * (30 / sent_count)) ** .5)) + 3.1291
        return SMOG
    else:
        return 0


def coleman_liau_index(text, token_count):
    """
    Takes a text and returns its Coleman Liau Index
    :param text: A string text
    :return: Coleman Liau Index
    """

    # Partly extracted of textstat 0.3.1 which is only available for python 2 (https://github.com/shivam5992/textstat)

    def char_count(text):
        """
        Function to return total character counts in a text
        """
        count_chars = 0
        text = text.replace(" ", "")
        for char in text:
            if char not in string.punctuation:
                count_chars += 1
        return count_chars

    def avg_letters_per_word(text):
        ALPW = float(float(char_count(text)) / token_count)
        return ALPW

    def avg_sentence_per_word(text):
        ASPW = float(len(nltk.sent_tokenize(text)) / float(token_count))
        return ASPW

    if token_count <= 0:
        return 0

    L = avg_letters_per_word(text) * 100  # avg letters per 100 words
    S = avg_sentence_per_word(text) * 100  # avg sentences per 100 words
    CLI = float((0.0588 * L) - (0.296 * S) - 15.8)
    return CLI


def automated_readability_index(text, token_count):
    """
    Takes a text and returns its Automated Readability Index
    :param text: A string text without punctuation
    :return: Automated Readability Index
    """

    # Partly extracted of textstat 0.3.1 which is only available for python 2 (https://github.com/shivam5992/textstat)
    def char_count(text):
        """
        Function to return total character counts in a text
        """
        count_chars = 0
        text = text.replace(" ", "")
        for char in text:
            if char not in string.punctuation:
                count_chars += 1
        return count_chars

    chrs = char_count(text)
    wrds = token_count
    snts = len(nltk.sent_tokenize(text))

    if wrds == 0 or snts == 0:
        return 0

    a = (float(chrs) / float(wrds))
    b = (float(wrds) / float(snts))
    ARI = (4.71 * a) + (0.5 * b) - 21.43
    return ARI


def lix_index(text, token_count):
    """
    A readability measure developed by Carl-Hug BjÃ¶rnsson
    Formula adapted from: https://en.wikipedia.org/wiki/LIX
    :param text: A string text without punctuation
    :return: LIX Index
    """

    def get_long_word_count(text):
        """
        Returns the number of words with more than 6 letters
        """
        long_word_count = 0
        for word in nltk.word_tokenize(text):
            if len(word) > 6:
                long_word_count += 1
        return long_word_count

    A = token_count  # number of words
    B = 0  # number of sentences (also split at ':')
    for sent in nltk.sent_tokenize(text):
        B += len(re.split(':', sent))
    C = get_long_word_count(text)  # number of words with more than 6 letters
    if B > 0 and A > 0:
        LIX = float(A / B) + float((C * 100) / A)
        return LIX
    else:
        return 0


def rix_index(text):
    """
    A readability measure developed by Anderson, simplifies LIX index
    Anderson, Jonathan. "Analysing the Radability of English and Non-English
    Texts in the Classroom with Lix"
    source: http://www.jstor.org/stable/40031755?seq=1#page_scan_tab_contents
    :param text: A string text without punctuation
    :return: RIX Index
    """

    def get_long_word_count(text):
        """
        Returns the number of words with more than 6 letters
        """
        long_word_count = 0
        for word in nltk.word_tokenize(text):
            if len(word) > 6:
                long_word_count += 1
        return long_word_count

    sent_count = 0  # number of sentences (also split at ':' and ';')
    for sent in nltk.sent_tokenize(text):
        sent_count += len(re.split('[:;]', sent))

    long_word_count = get_long_word_count(text)  # number of words with mroe than 6 letters

    if sent_count > 0:
        return float(long_word_count / sent_count)
    else:
        return 0


def mcalpine_eflaw_index(text):
    """
    A readability score defined by Rachel McAlpine
    See https://strainindex.wordpress.com/2009/04/30/mcalpine-eflaw-readability-score/

    EFLAW index = (#tokens + #miniwords) / #sentences

    :param text: A string text without punctuation
    :return: McAlpine EFLAW Index
    """

    tokenized_sents = nltk.sent_tokenize(text)
    sentence_count = len(tokenized_sents)
    token_count = 0
    miniword_count = 0  # words with 1,2 or 3 letters
    for sent in tokenized_sents:
        for token in nltk.word_tokenize(sent):
            if token not in string.punctuation:
                token_count += 1
                if len(token) <= 3:
                    miniword_count += 1

    if sentence_count >= 1:
        return float((token_count + miniword_count) / sentence_count)
    else:
        return 0


def strain_index(text):
    """
    Strain index in its general form. Sources:
        https://strainindex.wordpress.com/2007/09/25/hello-world/
        https://strainindex.wordpress.com/2011/02/21/the-origins-of-the-strain-index/
    Strain index = #Syllables in a sentence * 0.3 => years of schooling needed
    :param text: A string text without punctuation
    :return: Strain index
    """
    tokenized_sents = nltk.sent_tokenize(text)
    syllables_count = []
    for sent in tokenized_sents:
        syllables_count.append(syllable_count(sent))
    if len(tokenized_sents) > 0:
        return float((sum(syllables_count) * 0.3) / len(tokenized_sents))
    else:
        return 0


        # ========================================================================


def fk_reading_ease_features(headlines, bodies):
    def get_features(text, head_features=True):
        tokenized = nltk.word_tokenize(text)
        token_count = 0
        for token in tokenized:
            if token not in string.punctuation:
                token_count += 1

        # Flesh Kincaid Reading Ease index
        FK_reading_ease = flesch_kincaid_reading_ease(text, token_count)

        return [FK_reading_ease]

    head_features = [get_features(head, head_features=True) for head in tqdm(headlines)]
    body_features = [get_features(body, head_features=False) for body in tqdm(bodies)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features


def paragraph_features(headlines, bodies):
    def get_paragraph_breaks(text):
        """Identifies indented text or line breaks as the beginning of
        paragraphs and returns a list with indices of paragraph
        beginnings. List always starts with a 0 => from TextTilingTokenizer"""

        MIN_PARAGRAPH = 100
        pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
        matches = pattern.finditer(text)

        last_break = 0
        pbreaks = [0]
        for pb in matches:
            if pb.start() - last_break < MIN_PARAGRAPH:
                continue
            else:
                pbreaks.append(pb.start())
                last_break = pb.start()

        return pbreaks

    def get_avg_paragraph_length(text, pbreaks):
        """
        Takes a text and the indices of the paragraph breaks and reaturn the average
        paragraph lengths
        """
        paragraph_list = []
        counter = 0
        for index in pbreaks:
            if counter > 0:
                paragraph_list.append(text[pbreaks[counter - 1]:index])
            counter += 1
        paragraph_list.append(text[pbreaks[-1]:])

        paragraph_lengths = []
        for para in paragraph_list:
            tokenized = nltk.word_tokenize(para)
            para_length = 0
            for token in tokenized:
                if token not in string.punctuation:
                    para_length += 1
            paragraph_lengths.append(para_length)

        if len(paragraph_lengths) > 0:
            return sum(paragraph_lengths) / len(paragraph_lengths)
        else:
            return 0

    def get_features(text):

        # amount of paragraphs and their average lengths
        pbreaks = get_paragraph_breaks(text)
        paragraph_count = len(pbreaks) - 1
        avg_paragraph_length = get_avg_paragraph_length(text, pbreaks)

        return [paragraph_count, avg_paragraph_length]

    body_features = [get_features(body) for body in tqdm(bodies)]

    return body_features


def quoted_words_features(headlines, bodies):
    def get_features(text, head_features=True):
        tokenized = nltk.word_tokenize(text)
        token_count = 0
        for token in tokenized:
            if token not in string.punctuation:
                token_count += 1

        # get number of quotes http://stackoverflow.com/questions/28037857/how-to-extract-all-quotes-in-a-document-text-using-regex
        # and then calculate the ratio #quoted words / #words
        match = re.findall('(?:[\â€œ\'\"](.*?)[\â€\'\"])', text)
        quoted_words_count = 0
        quoted_word_ratio = 0
        for quote in match:
            tokenized_quote = nltk.word_tokenize(quote)
            for token in tokenized_quote:
                if token not in string.punctuation:
                    quoted_words_count += 1
        if token_count > 0:
            quoted_word_ratio = float(quoted_words_count / token_count)

        return [quoted_word_ratio]

    head_features = [get_features(head, head_features=True) for head in tqdm(headlines)]
    body_features = [get_features(body, head_features=False) for body in tqdm(bodies)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features


def SMOG_index_features(headlines, bodies):
    body_features = [[smog_index(body)] for body in tqdm(bodies)]

    return body_features


def CL_index_features(headlines, bodies):
    def get_features(text):
        tokenized = nltk.word_tokenize(text)
        token_count = 0
        for token in tokenized:
            if token not in string.punctuation:
                token_count += 1

        # get coleman liau index
        CL_index = coleman_liau_index(text, token_count)

        return [CL_index]

    head_features = [get_features(head) for head in tqdm(headlines)]
    body_features = [get_features(body) for body in tqdm(bodies)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features


def AR_index_features(headlines, bodies):
    def get_features(text):
        tokenized = nltk.word_tokenize(text)
        token_count = 0
        for token in tokenized:
            if token not in string.punctuation:
                token_count += 1

        # get automated readability index
        AR_index = automated_readability_index(text, token_count)

        return [AR_index]

    head_features = [get_features(head) for head in tqdm(headlines)]
    body_features = [get_features(body) for body in tqdm(bodies)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features


def LIX_index_features(headlines, bodies):
    def get_features(text):
        tokenized = nltk.word_tokenize(text)
        token_count = 0
        for token in tokenized:
            if token not in string.punctuation:
                token_count += 1

        # LIX readability index
        LIX_index = lix_index(text, token_count)

        return [LIX_index]

    head_features = [get_features(head) for head in tqdm(headlines)]
    body_features = [get_features(body) for body in tqdm(bodies)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features


def RIX_index_features(headlines, bodies):
    def get_features(text):
        tokenized = nltk.word_tokenize(text)
        token_count = 0
        for token in tokenized:
            if token not in string.punctuation:
                token_count += 1

        # RIX readability index
        RIX_index = rix_index(text)

        return [RIX_index]

    head_features = [get_features(head) for head in tqdm(headlines)]
    body_features = [get_features(body) for body in tqdm(bodies)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features


def EFLAW_index_features(headlines, bodies):
    head_features = [[mcalpine_eflaw_index(head)] for head in tqdm(headlines)]
    body_features = [[mcalpine_eflaw_index(body)] for body in tqdm(bodies)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features


def strain_index_features(headlines, bodies):
    head_features = [[strain_index(head)] for head in tqdm(headlines)]
    body_features = [[strain_index(body)] for body in tqdm(bodies)]

    features = np.concatenate([head_features, body_features], axis=1)

    return features

#####################################################################################
# MTLD Implementation from : https://github.com/jfrens/lexical_diversity           #
#####################################################################################

#Copyright 2017 John Frens
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# Global trandform for removing punctuation from words
remove_punctuation = str.maketrans('', '', string.punctuation)

# MTLD internal implementation
def mtld_calc(word_array, ttr_threshold):
    current_ttr = 1.0
    token_count = 0
    type_count = 0
    types = set()
    factors = 0.0

    for token in word_array:
        token = token.translate(remove_punctuation).lower()  # trim punctuation, make lowercase
        token_count += 1
        if token not in types:
            type_count += 1
            types.add(token)
        current_ttr = type_count / token_count
        if current_ttr <= ttr_threshold:
            factors += 1
            token_count = 0
            type_count = 0
            types = set()
            current_ttr = 1.0

    excess = 1.0 - current_ttr
    excess_val = 1.0 - ttr_threshold
    factors += excess / excess_val
    if factors != 0:
        return len(word_array) / factors
    return -1


# MTLD implementation
# Good explanation https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3813439/#R41
#   https://de.wikipedia.org/wiki/Type-Token-Relation
def mtld(word_array, ttr_threshold=0.72):
    if isinstance(word_array, str):
        raise ValueError("Input should be a list of strings, rather than a string. Try using string.split()")
    if len(word_array) < 50:
        return 0


    return (mtld_calc(word_array, ttr_threshold) + mtld_calc(word_array[::-1], ttr_threshold)) / 2

#####################################################################################
# Flesch Kincaid Implementations from : https://github.com/stephenhky/PyReadability #
#####################################################################################

# load CMU pronunciation dictionary
prondict = cmudict.dict()
#http://www.mortengoodwin.net/publicationfiles/webmining_2007_reportgroup6.pdf.1
# count the number of syllables
numsyllables_pronlist = lambda l: len(list(filter(lambda s: isdigit(s.encode('ascii', 'ignore').lower()[-1]), l)))
def numsyllables(word):
    try:
        return list(set(map(numsyllables_pronlist, prondict[word.lower()])))
    except KeyError:
        return [0]

not_punctuation = lambda w: not (len(w)==1 and (not w.isalpha()))
get_word_count = lambda text: len(list(filter(not_punctuation, nltk.word_tokenize(text))))
get_sent_count = lambda text: len(nltk.sent_tokenize(text))

def text_statistics(text):
    word_count = get_word_count(text)
    sent_count = get_sent_count(text)
    syllable_count = sum(map(lambda w: max(numsyllables(w)), nltk.word_tokenize(text)))
    return word_count, sent_count, syllable_count

# Flesch reading ease
# reference: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
flesch_formula = lambda word_count, sent_count, syllable_count : 206.835 - 1.015*word_count/sent_count - 84.6*syllable_count/word_count
def flesch_reading_ease(text):
    word_count, sent_count, syllable_count = text_statistics(text)
    if word_count <= 0 or sent_count <= 0:
        return 0
    else:
        return flesch_formula(word_count, sent_count, syllable_count)

# Flesch-Kincaid grade level
# reference: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
fk_formula = lambda word_count, sent_count, syllable_count : 0.39 * word_count / sent_count + 11.8 * syllable_count / word_count - 15.59
def flesch_grade_level(text):
    word_count, sent_count, syllable_count = text_statistics(text)
    if word_count <= 0 or sent_count <= 0:
        return 0
    else:
        return fk_formula(word_count, sent_count, syllable_count)

############################################################################
############################################################################
############################################################################