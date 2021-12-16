"""
https://www.analyticsvidhya.com/blog/2021/06/must-known-techniques-for-text-preprocessing-in-nlp/
https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a
https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8
"""

import pandas as pd
import pickle
import unidecode
import re
import string
from autocorrect import Speller
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def unique_records(df, col_name):
    df = df.drop_duplicates(col_name)
    return df


def remove_extra_spaces(text):
    """
    Following function will remove the leading and trailing extra spaces and the tabs in the sentences
    :param row: Row of DataFrame
    :return: preprocessed row
    """
    text = text.lstrip()
    text = text.rstrip()
    text = re.sub(' +', ' ', text)
    return text


def remove_links(text):
    """
    Following function removes the hyper links from the text
    :param text:
    :return:
    """
    # Removing all the occurrences of links that starts with https
    remove_https = re.sub(r'http\S+', '', text)
    # Remove all the occurrences of text that ends with .com
    remove_com = re.sub(r"\ [A-Za-z]*\.com", " ", remove_https)
    # Remove the extra spaces caused after the removal
    clean_text = remove_extra_spaces(remove_com)
    return clean_text


def toUTF(text):
    """
    Followign converts all the text to UTF 8
    :param text:
    :return:
    """
    utf = unidecode.unidecode(text)
    clean = remove_extra_spaces(utf)
    return clean


def to_lower(text):
    """
    Convert the entire text to lower case
    :param text:
    :return:
    """
    return text.lower()


def remove_digits(text):
    ordinals = re.findall('(\d+(?:st|nd|rd|th))', text)
    for ordinal in ordinals:
        text = text.replace(ordinal, '')
    digit_free = ''.join([i for i in text if not i.isdigit()])
    return remove_extra_spaces(digit_free)


def remove_contraction(text):
    """
    Following function converts the shorthand words to full.
    Eg: ain't: is not
    :param text:
    :return:
    """
    with open('contraction-map.pkl', 'rb') as f:
        map = pickle.load(f)
        text = text.replace(',', '')

        # Tokenizing text into tokens.
        list_Of_tokens = text.split(' ')

        # Checking for whether the given token matches with the Key & replacing word with key's value.

        # Check whether Word is in lidt_Of_tokens or not.
        for Word in list_Of_tokens:
            # Check whether found word is in dictionary "Contraction Map" or not as a key.
            if Word in map:
                # If Word is present in both dictionary & list_Of_tokens, replace that word with the key value.
                list_Of_tokens = [item.replace(Word, map[Word]) for item in list_Of_tokens]

        # Converting list of tokens to String.
        text = ' '.join(str(e) for e in list_Of_tokens)
        return text


def remove_punctuations(text):
    """
    Remove the punctuations from the text
    :param text:
    :return:
    """
    punct_free = text.translate(str.maketrans('', '', string.punctuation))
    return remove_extra_spaces(punct_free)


def spelling_correction(text):
    '''
    This function will correct spellings.

    arguments:
         input_text: "text" of type "String".

    return:
        value: Text after corrected spellings.

    Example:
    Input : This is Oberois from Dlhi who came heree to studdy.
    Output : This is Oberoi from Delhi who came here to study.


    '''
    # Check for spellings in English language
    spell = Speller(lang='en')
    Corrected_text = spell(text)
    return Corrected_text


def preprocess(row):  # row, col_name='tweet_text'
    # df = unique_records(df)
    text = row['tweet_text']
    text = remove_extra_spaces(text)
    text = remove_links(text)
    text = toUTF(text)
    text = to_lower(text)
    text = remove_contraction(text)
    text = remove_digits(text)
    punct_free = remove_punctuations(text)
    correct_text = spelling_correction(punct_free)
    t1 = ' '.join([w for w in correct_text.split() if len(w) > 2])
    t2 = remove_extra_spaces(t1)
    row['clean'] = t2
    return row


def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


if __name__ == '__main__':
    # keep_cols = ['event_name', 'tweet_id', 'tweet_text', 'label']
    truth = {'not_humanitarian': 0, 'other_relevant_information': 1, 'rescue_volunteering_or_donation_effort': 2,
             'infrastructure_and_utility_damage': 3, 'affected_individuals': 4}
    tdf = pd.read_csv('preprocess_Train_tweets.csv', sep=',')
    vdf = pd.read_csv('preprocess_Val_tweets.csv', sep=',')
    combine = pd.read_csv('combine.csv', sep=',')
    test = pd.read_csv('preprocess_Test_tweets.csv', sep=',')
    print('Train Df Shape:', tdf.shape)
    print('Val Df Shape:', vdf.shape)
    print('Combine Df Shape:', combine.shape)
    print('Test Df Shape:', test.shape)
    # combine = tdf.append(vdf)
    tdf = tdf.drop('label_id', 1)
    vdf = vdf.drop('label_id', 1)
    combine = combine.drop('label_id', 1)
    test = test.drop('label_id', 1)
    print('Train Df Shape:', tdf.shape)
    print('Val Df Shape:', vdf.shape)
    print('Combine Df Shape:', combine.shape)
    print('Test Df Shape:', test.shape)
    tdf['label_id'] = tdf['label'].map(lambda x: truth[x])
    vdf['label_id'] = vdf['label'].map(lambda x: truth[x])
    combine['label_id'] = combine['label'].map(lambda x: truth[x])
    test['label_id'] = test['label'].map(lambda x: truth[x])
    # print('Combine Df Shape:', combine.shape)
    # df = df[keep_cols]
    # df = df.drop_duplicates('tweet_id')
    # print('Unique Df Shape:', df.shape)
    # df['label_id'] = pd.factorize(df['label'])[0]
    # df['clean'] = None
    # print('New Df Shape:', df.shape)
    # df = df.apply(preprocess, axis=1)
    # print(df.head(5).to_string())
    # print('Unique Tweets:', df.shape[0])
    print('Train Dataset Details ::')
    print(tdf.value_counts(subset=['label_id']))
    print('Val Dataset Details ::')
    print(vdf.value_counts(subset=['label_id']))
    print('Combine Dataset Details ::')
    print(combine.value_counts(subset=['label_id']))
    print('Test Dataset Details ::')
    print(test.value_counts(subset=['label_id']))

    # df["clean"] = df["clean"].apply(lambda text: ' '.join([w for w in text.split() if len(w) > 2]))
    # df["clean"] = df["clean"].apply(lambda text: remove_extra_spaces(text))
    tdf.to_csv('preprocess_Train_tweets.csv', sep=',', index=False)
    vdf.to_csv('preprocess_Val_tweets.csv', sep=',', index=False)
    combine.to_csv('combine.csv', sep=',', index=False)
    test.to_csv('preprocess_Test_tweets.csv', sep=',', index=False)
    # print('Unique Tweets:', df.shape[0])
    # df.to_csv('unique_tweets.csv', sep=',', index=False)
    #
    # df = pd.read_csv('unique_tweets.csv', sep=',')
    # print('Total Records::', df.shape[0])
    # text = "RT @Ciscokid__: Calistoga Fire #tubbsfire #napafire #ABC7now #kron4news #fire #california #napa https://t.co/aTcggkpkNe"
    #
    # print(preprocess(text))
