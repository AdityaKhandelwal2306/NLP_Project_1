import spacy

def download_model(model_name):
    try:
        spacy.cli.download(model_name)
    except Exception as e:
        print(f"Error downloading model {model_name}: {e}")

if __name__ == "__main__":
    download_model("en_core_web_sm")
