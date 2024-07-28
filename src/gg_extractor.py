import spacy
from spacy.lang.en import English
import json
import pprint

import pandas as pd
import re
import sys
import numpy as np

from itertools import combinations
from collections import defaultdict
import tabulate

import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

import pprint

nlp = spacy.load('en_core_web_sm')

hardcoded_awards = ['cecil b. demille award',
                   'best motion picture drama',
                   'best performance by an actress in a motion picture - drama',
                   'best performance by an actor in a motion picture - drama',
                   'best motion picture - comedy or musical',
                   'best performance by an actress in a motion picture - comedy or musical',
                   'best performance by an actor in a motion picture - comedy or musical',
                   'best animated feature film',
                   'best foreign language film',
                   'best performance by an actress in a supporting role in a motion picture',
                   'best performance by an actor in a supporting role in a motion picture',
                   'best director - motion picture',
                   'best screenplay - motion picture',
                   'best original score - motion picture',
                   'best original song - motion picture',
                   'best television series - drama',
                   'best performance by an actress in a television series - drama',
                   'best performance by an actor in a television series - drama',
                   'best television series - comedy or musical',
                   'best performance by an actress in a television series - comedy or musical',
                   'best performance by an actor in a television series - comedy or musical',
                   'best mini-series or motion picture made for television',
                   'best performance by an actress in a mini-series or motion picture made for television',
                   'best performance by an actor in a mini-series or motion picture made for television',
                   'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television',
                   'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']

# defining path function
def read(path):
    with open(path, mode='rt') as file:
        data = pd.read_json(file)
    
    # removing duplicates based on 'text' field
    data.drop_duplicates(subset='text', inplace=True)

    # find all hashtags and store them into a new field 'hashtag'
    data['hashtag'] = data['text'].apply(lambda rec: re.findall(r'#(\w+)', rec))

    # find all mentions and add to new field 'mention'
    data['mention'] = data['text'].apply(lambda rec: re.findall(r'@(\w+)', rec))

    # removing 'Golden globes' like words from text field
    data['text'] = data['text'].str.replace(r'[G|g]olden\s?[G|g]lobes*', ' ')

    # removing hashtags and mentions from text field
    data['text'] = data['text'].str.replace(r'#|@|RT', '')

    # removing URLs from text field
    data['text'] = data['text'].str.replace(r'http\S+|www.\S+', '')

    return data

# Defining function that will preprocess the text field
def pre_process_data(data):
    # removing all numbers in the data
    data = data.astype(str).str.replace(r'\d+', ' ')

    # making every text lowercase
    lower_text = data.str.lower()
    
    # defining function to lemmatize text
    def lemmatize_text(text):
        doc = nlp(text)
        return [token.lemma_ for token in doc]
    
    # defining function to remove punctuations
    def remove_punctuations(words):
        return [re.sub(r'[^\w\s]', '', word) for word in words if re.sub(r'[^\w\s]', '', word) != '']
    
    words = lower_text.apply(lemmatize_text)
    words = words.apply(remove_punctuations)

    return pd.DataFrame(words)

# Removing stopwords
stopwords_set = set(nlp.Defaults.stop_words)

def remove_stop_words(df, field_name):
    return df[field_name].apply(lambda rec: [word for word in rec if word not in stopwords_set])

# filter tweets using regex string, input is tokenized column
def filter_tweets(df, field_name, regex_string):
    temp = pd.DataFrame(df[field_name].apply(lambda rec: " ".join(rec)))
    return df[temp[field_name].str.contains(regex_string, regex=True)]

# capture nouns, Input: dataframe and 'text' field 
def capture_nouns(df, field_name):
    return pd.DataFrame([token.text.lower() for doc in df[field_name].apply(nlp) for token in doc if token.text.lower() not in stopwords_set and token.pos_ == 'NOUN'])

# defining function to return frequency of words
def word_freq(df, field_name):
    return df[field_name].value_counts().index.tolist()

def vals_freq(df, field_name):
    return df[field_name].value_counts()

def duplicate_tweet_removal(df, field_name):
    return df.drop_duplicates(subset=[field_name], keep='first')

# capturing proper nouns
def capture_proper_nouns(df, field_name):
    all_nouns = []
    bad_words = ['oscar', 'nominee', 'emmy', 'golden', 'globes', 'worst', 'president','goldenglobes','goldenglobesnex','@']
    [bad_words.append(word) for award in hardcoded_awards for word in award.replace("- ","").split()]
    freq = defaultdict(int)

    for text in df[field_name]:
        nouns = []
        name = []
        flag = False

        words_token = nlp(text.replace(":", " "))
        words_token = [token.text if token.text.lower() not in bad_words else "a" for token in words_token]
        tagged = [(token.text, token.tag_) for token in nlp(" ".join(words_token))]

        for (word, tag) in tagged:
            if tag == 'NNP' and not word.isupper():
                name.append(word)
                flag = True
            elif flag:
                str_name = " ".join(name)
                nouns.append(str_name)
                freq[str_name] += 1
                name = []
                flag = False
        all_nouns.append(nouns)
        nouns = []
    df['associated_proper_nouns'] = all_nouns
    df['associated_proper_nouns'] = df['associated_proper_nouns'].apply(lambda rec: [name for name in rec if freq[name] > 1])
    df = df[df.astype(str)['associated_proper_nouns'] != '[]'] 

    sorted_freq = sorted(freq.items(), 
                        key=lambda rec: rec[1],
                        reverse=True)
    return df, sorted_freq

# defining function to find hosts, input = dataframe with 'text' and 'pp_text'
def capture_hosts(df):
    tweets_filtered = filter_tweets(df, 'pp_text', '(?:host|hosts|hosted)')
    potential_hosts, host_freq = capture_proper_nouns(tweets_filtered, 'text')

    if host_freq[0][1] - host_freq[1][1] > 100:
        return host_freq[0][0]
    else:
        return host_freq[0][0], host_freq[1][0]
    
# function to capture best dressed with input df
def capture_best_dressed(df):
    tweets_filtered = filter_tweets(df, 'pp_text', '(?:best dressed)')
    potential_best_dressed, best_dressed_freq = capture_proper_nouns(tweets_filtered, 'text')
    print('potential_best_dressed',potential_best_dressed)
    print('best_dressed_freq',best_dressed_freq)
    return best_dressed_freq[0][0]

# function to capture worst dressed with input df
def capture_worst_dressed(df):
    tweets_filtered = filter_tweets(df, 'pp_text', '(?:worst dressed)')
    potential_worst_dressed, worst_dressed_freq = capture_proper_nouns(tweets_filtered, 'text')
    print('potential_worst_dressed',potential_worst_dressed)
    print('worst_dressed_freq',worst_dressed_freq)
    return worst_dressed_freq[0][0]

# function to capture funniest with input df
def capture_funniest(df):    
    tweets_filtered = filter_tweets(df, 'pp_text', '(?:funniest|hilarious)')
    potential_funniest, funniest_freq = capture_proper_nouns(tweets_filtered, 'text')
    return funniest_freq[0][0]

# function to capture awards input df
def capture_awards(df):
    tweets_filtered = filter_tweets(df, 'pp_text', '(?:best)')
    tweet_text = tweets_filtered['text']
    award_freqs = vals_freq(tweet_text.str.extract(r'for (best[\w ,-]+) (?:for|[!#])', re.IGNORECASE).dropna(), 0)
    awards = word_freq(tweet_text.str.extract(r'for (best [\w,-]+ (?:for|[!#]))', re.IGNORECASE).dropna(), 0)

    awards_filtered = []
    for i, val in enumerate(award_freqs):
        if val > 3: 
            awards_filtered.append(awards[i].lower())
    
    return awards_filtered

# function to capture presenters 
def link_presenter(presenter_freq):
    return [presenter_freq[0][0]]

# function to find nominees
def capture_nominees(potential_nominees, nominee_freq):
    nominees = potential_nominees['associated_proper_nouns'].tolist()
    nominees_list = []
    counter = 0

    for nominee in nominee_freq:
        name = nominee[0]
        if [name] in nominees:
            counter += 1
            nominees_list.append(name)

            if counter == 5:
                return nominees_list
    return nominees_list

def format_regex_str(award):
    award_filtered = award

    words = award_filtered.split(' ')

    formatted_regex = ''
    for w in words:
        if w != '-':
            formatted_regex = formatted_regex + '(?=.*' + w + ')'

    award_filtered = formatted_regex

    stop_words = ['(?=.*by)',
                  '?=.*an',
                  '(?=.*in)',
                  '(?=.*a)',
                  '(?=.*or)',
                  '(?=.*made)',
                  '(?=.*for)',
                  '(?=.*b.)'
                  ]
    
    for w in stop_words:
        award_filtered = award_filtered.replace(w, '')

    award_filtered = award_filtered.replace('-', '')

    return award_filtered

def process_award(df, awards):
    presenters_nominees_winner = {}

    for award in awards:
        award_regex = '|'.join(award.split())
        
        nominee_tweets = filter_tweets(df, 'pp_text', '(?=.*(nomination|nominee|nominated|nominate|snubbed|snub))(?=.*(' + award_regex + '))')
        potential_nominees, nominee_freq =  capture_proper_nouns(nominee_tweets, 'text')
        nominees = capture_nominees(potential_nominees, nominee_freq)

        winner_tweets = filter_tweets(df, 'pp_text', '(?=.*(win|wins|winner|winners|won))(?=.*(' + award_regex + '))')
        potential_winners, winner_freq = capture_proper_nouns(winner_tweets, 'text')

        presenter_tweets = filter_tweets(df, 'pp_text', '(?=.*(present|presenter|presents|presented|presenters))(?=.*(' + award_regex + '))')
        potential_presenters, presenter_freq = capture_proper_nouns(presenter_tweets, 'text')

        presenters = link_presenter(presenter_freq)
        for candidate in winner_freq[:6]:
            if candidate[0] in nominees:
                winner = candidate[0]
                break
        else: 
            print('Candidate was', candidate[0])
            winner = 'Not present in Nominees'

        presenters_nominees_winner[award] = {'Presenters': presenters,
                                             'Nominees': nominees,
                                             'Winner': winner}
    return presenters_nominees_winner

# printing output
def output(data, awards):
    hosts = capture_hosts(data)
    presenters_nominees_winner = process_award(data, awards)

    presenters_nominees_winner['Host'] = hosts

    best_dressed = capture_best_dressed(data)
    worst_dressed = capture_worst_dressed(data)
    funniest = capture_funniest(data)

    for key in presenters_nominees_winner:
        print(key, presenters_nominees_winner[key])
        
    print("Best Dressed: " + best_dressed)
    print("Worst Dressed: " + worst_dressed)
    print("Funniest: " + funniest)

    with open('Output.txt', 'w') as file:
        for key, nested in presenters_nominees_winner.items():
            if key != 'Host':
                file.write(f'\n')
                file.write(f'Award: {key} \n')
                for k2, v2 in nested.items():
                    if type(v2) != type('g'):
                        temp = ", ".join(v2)
                        file.write(f'{k2}: {temp} \n')
                    else: 
                        file.write(f'{k2}: {v2} \n')
            else:
                temp = ", ".join(nested)
                file.write(f'\n')
                file.write(f'{key}: {temp} \n')
            file.write(f'\n')
            file.write(f'Best Dressed: {best_dressed}')
            file.write(f'Worst Dressed: {worst_dressed}')
            file.write(f'Funniest:  {funniest}')      
            
    return presenters_nominees_winner
