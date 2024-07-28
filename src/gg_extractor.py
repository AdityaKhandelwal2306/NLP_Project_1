import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

import spacy
from collections import defaultdict
import pandas as pd
import re
import numpy as np

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nlp = spacy.load('en_core_web_sm')

preset_awards = [
    'cecil b. demille award', 'best motion picture drama',
    'best performance by an actress in a motion picture - drama',
    'best performance by an actor in a motion picture - drama',
    'best motion picture - comedy or musical',
    'best performance by an actress in a motion picture - comedy or musical',
    'best performance by an actor in a motion picture - comedy or musical',
    'best animated feature film', 'best foreign language film',
    'best performance by an actress in a supporting role in a motion picture',
    'best performance by an actor in a supporting role in a motion picture',
    'best director - motion picture', 'best screenplay - motion picture',
    'best original score - motion picture', 'best original song - motion picture',
    'best television series - drama', 'best performance by an actress in a television series - drama',
    'best performance by an actor in a television series - drama',
    'best television series - comedy or musical',
    'best performance by an actress in a television series - comedy or musical',
    'best performance by an actor in a television series - comedy or musical',
    'best mini-series or motion picture made for television',
    'best performance by an actress in a mini-series or motion picture made for television',
    'best performance by an actor in a mini-series or motion picture made for television',
    'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television',
    'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television'
]

# Read and preprocess the data
def read(path):
    with open(path, 'r') as file:
        data = pd.read_json(file)

    data.drop_duplicates(subset='text', inplace=True)
    data['hashtag'] = data['text'].apply(lambda x: re.findall(r'#(\w+)', x))
    data['mention'] = data['text'].apply(lambda x: re.findall(r'@(\w+)', x))
    data['text'] = data['text'].str.replace(r'[G|g]olden\\s?[G|g]lobes*', ' ', regex=True)
    data['text'] = data['text'].str.replace(r'#|@|RT', '', regex=True)
    data['text'] = data['text'].str.replace(r'http\S+|www.\S+', '', regex=True)

    return data

# Preprocess the text data
def pre_process_data(data):
    data = data.astype(str).str.replace(r'\d+', ' ', regex=True).str.lower()
    lemmatizer = WordNetLemmatizer()
    tokenizer = TweetTokenizer()

    def lemmatize(text):
        return [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(text)]

    def remove_punctuation(words):
        return [re.sub(r'[^\w\s]', ' ', word) for word in words if re.sub(r'[^\w\s]', ' ', word) != '']

    words = data.apply(lemmatize).apply(remove_punctuation)
    return pd.DataFrame(words)

# Remove stop words
stop_words = set(stopwords.words('english'))

def remove_stop_words(df, column_name):
    return df[column_name].apply(lambda x: [word for word in x if word not in stop_words])

# Filter tweets using regex
def filter_tweets(df, column_name, regex_pattern):
    temp_df = pd.DataFrame(df[column_name].apply(lambda x: " ".join(x)))
    return df[temp_df[column_name].str.contains(regex_pattern, regex=True)]

# Extract proper nouns
def capture_proper_nouns(df, column_name):
    proper_nouns = []
    filter_words = ['oscar', 'nominee', 'emmy', 'golden', 'globes', 'worst', 'president','goldenglobes','goldenglobesnex','@']
    [filter_words.append(word) for award in preset_awards for word in award.replace("- ","").split()]
    freq = defaultdict(int)

    for text in df[column_name]:
        nouns = []
        name = []
        flag = False

        words_token = nltk.word_tokenize(text.replace(":", " "))
        words_token = [word if word.lower() not in filter_words else "a" for word in words_token]
        tagged = nltk.pos_tag(words_token)

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
        proper_nouns.append(nouns)
        nouns = []
    df['proper_nouns'] = proper_nouns
    df['proper_nouns'] = df['proper_nouns'].apply(lambda x: [name for name in x if freq[name] > 1])
    df = df[df.astype(str)['proper_nouns'] != '[]'] 

    sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return df, sorted_freq

# Capture hosts
def capture_hosts(df):
    filtered_tweets = filter_tweets(df, 'pp_text', '(?:host|hosts|hosted)')
    potential_hosts, host_freq = capture_proper_nouns(filtered_tweets, 'text')
    if host_freq[0][1] - host_freq[1][1] > 100:
        return [host_freq[0][0]]
    else:
        return [host_freq[0][0], host_freq[1][0]]

# Capture best dressed
def capture_best_dressed(df):
    filtered_tweets = filter_tweets(df, 'pp_text', '(?:best dressed)')
    potential_best_dressed, best_dressed_freq = capture_proper_nouns(filtered_tweets, 'text')
    return [best_dressed_freq[0][0]]

# Capture worst dressed
def capture_worst_dressed(df):
    filtered_tweets = filter_tweets(df, 'pp_text', '(?:worst dressed)')
    potential_worst_dressed, worst_dressed_freq = capture_proper_nouns(filtered_tweets, 'text')
    return [worst_dressed_freq[0][0]]

# Capture funniest
def capture_funniest(df):    
    filtered_tweets = filter_tweets(df, 'pp_text', '(?:funniest|hilarious)')
    potential_funniest, funniest_freq = capture_proper_nouns(filtered_tweets, 'text')
    return [funniest_freq[0][0]]

# Capture awards
def capture_awards(df):
    filtered_tweets = filter_tweets(df, 'pp_text', '(?:best)')
    tweet_text = filtered_tweets['text']
    award_freqs = tweet_text.str.extract(r'for (best[\w ,-]+) (?:for|[!#])', re.IGNORECASE).dropna()[0].value_counts()
    awards = tweet_text.str.extract(r'for (best [\w,-]+ (?:for|[!#]))', re.IGNORECASE).dropna()[0].value_counts().index.tolist()

    return [award.lower() for award, freq in zip(awards, award_freqs) if freq > 3]

# Link presenters
def link_presenter(presenter_freq):
    return [presenter_freq[0][0]]

# Identify nominees
def identify_nominees(potential_nominees, nominee_freq):
    nominees = potential_nominees['proper_nouns'].tolist()
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

# Process each award and identify related entities
def process_award(df, awards):
    presenters_nominees_winner = {}

    for award in awards:
        award_regex = '|'.join(award.split())
        
        nominee_tweets = filter_tweets(df, 'pp_text', f'(?=.*(nomination|nominee|nominated|nominate|snubbed|snub))(?=.*({award_regex}))')
        potential_nominees, nominee_freq = capture_proper_nouns(nominee_tweets, 'text')
        nominees = identify_nominees(potential_nominees, nominee_freq)

        winner_tweets = filter_tweets(df, 'pp_text', f'(?=.*(win|wins|winner|winners|won))(?=.*({award_regex}))')
        potential_winners, winner_freq = capture_proper_nouns(winner_tweets, 'text')

        presenter_tweets = filter_tweets(df, 'pp_text', f'(?=.*(present|presenter|presents|presented|presenters))(?=.*({award_regex}))')
        potential_presenters, presenter_freq = capture_proper_nouns(presenter_tweets, 'text')

        presenters = link_presenter(presenter_freq)
        for candidate in winner_freq[:6]:
            if candidate[0] in nominees:
                winner = candidate[0]
                break
        else: 
            winner = 'Not present in Nominees'

        presenters_nominees_winner[award] = {'Presenters': presenters,
                                             'Nominees': nominees,
                                             'Winner': winner}
    return presenters_nominees_winner
