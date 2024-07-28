import logging
import json
from .gg_extractor import (
    read, pre_process_data, remove_stop_words,
    capture_hosts, capture_awards, process_award,
    capture_best_dressed, capture_worst_dressed, capture_funniest
)

def main():
    try:
        logging.basicConfig(level=logging.INFO)
        
        logging.info("Loading data...")
        texts = read('data/gg2013.json')
        
        logging.info("Preprocessing text data...")
        texts['pp_text'] = pre_process_data(texts['text'])
        texts['pp_text'] = remove_stop_words(texts, 'pp_text')
        
        logging.info("Extracting hosts and awards...")
        hosts = capture_hosts(texts)
        awards = capture_awards(texts)
        
        logging.info("Extracting presenters, nominees, and winners...")
        presenters_nominees_winner = process_award(texts, awards)
        
        logging.info("Extracting best and worst dressed...")
        best_dressed = capture_best_dressed(texts)
        worst_dressed = capture_worst_dressed(texts)
        
        logging.info("Extracting jokes...")
        jokes = capture_funniest(texts)
        
        # Placeholder functions for extracting parties, sentiments, and performances
        parties = extract_parties(texts)
        positive_sentiments, negative_sentiments = extract_sentiment(texts)
        performances = extract_performances(texts)
        
        logging.info("Formatting results...")
        human_readable_output = format_human_readable(
            hosts, awards, presenters_nominees_winner, 
            best_dressed, worst_dressed, jokes, parties,
            positive_sentiments, negative_sentiments, performances
        )
        json_output = format_json(
            hosts, awards, presenters_nominees_winner, 
            best_dressed, worst_dressed, jokes, parties,
            positive_sentiments, negative_sentiments, performances
        )
        
        logging.info("Saving results...")
        save_output('output/human_readable_output.txt', human_readable_output, 'text')
        save_output('output/json_output.json', json_output, 'json')
        
        logging.info("Process completed successfully.")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def format_human_readable(
    hosts, awards, presenters_nominees_winner, 
    best_dressed, worst_dressed, jokes, parties,
    positive_sentiments, negative_sentiments, performances
):
    output = []
    output.append(f"Hosts: {', '.join(hosts)}\n")
    
    for award in awards:
        details = presenters_nominees_winner.get(award, {})
        output.append(f"Award: {award}")
        output.append(f"Presenters: {', '.join(details.get('Presenters', []))}")
        output.append(f"Nominees: {', '.join(details.get('Nominees', []))}")
        output.append(f"Winner: {details.get('Winner', '')}\n")
    
    output.append(f"Best Dressed: {', '.join(best_dressed)}")
    output.append(f"Worst Dressed: {', '.join(worst_dressed)}")
    output.append(f"Humor (Jokes): {', '.join(jokes)}")
    output.append(f"Parties: {', '.join(parties)}")
    output.append(f"Positive Sentiments: {', '.join(positive_sentiments)}")
    output.append(f"Negative Sentiments: {', '.join(negative_sentiments)}")
    output.append(f"Performances: {', '.join(performances)}")
    
    return '\n'.join(output)

def format_json(
    hosts, awards, presenters_nominees_winner, 
    best_dressed, worst_dressed, jokes, parties,
    positive_sentiments, negative_sentiments, performances
):
    output = {"Hosts": hosts}
    
    for award in awards:
        details = presenters_nominees_winner.get(award, {})
        output[award] = {
            "Presenters": details.get("Presenters", []),
            "Nominees": details.get("Nominees", []),
            "Winners": [details.get("Winner", "")]
        }
    
    output["Best Dressed"] = best_dressed
    output["Worst Dressed"] = worst_dressed
    output["Humor (Jokes)"] = jokes
    output["Parties"] = parties
    output["Positive Sentiments"] = positive_sentiments
    output["Negative Sentiments"] = negative_sentiments
    output["Performances"] = performances
    
    return output

def save_output(filepath, data, format):
    try:
        if format == 'text':
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(data)
        elif format == 'json':
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)
    except Exception as e:
        logging.error(f"Failed to save {format} output to {filepath}: {e}")

def extract_parties(texts):
    # Placeholder implementation
    return ["Party1", "Party2"]

def extract_sentiment(texts):
    # Placeholder implementation
    return ["PositiveSentiment1", "PositiveSentiment2"], ["NegativeSentiment1", "NegativeSentiment2"]

def extract_performances(texts):
    # Placeholder implementation
    return ["Performance1", "Performance2"]

if __name__ == '__main__':
    main()
