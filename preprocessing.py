import re
import os
import nltk
import pickle
import logging
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

# Ensure punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def insert_space_in_glued_words(text):
    """
    Inserts spaces between glued words and fixes common concatenations.
    """
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add space between camelCase words
    text = re.sub(r'(?<=[0-9])(?=[a-zA-Z])', ' ', text)  # Add space between numbers and letters
    text = re.sub(r'(?<=[a-zA-Z])(?=[0-9])', ' ', text)  # Add space between letters and numbers

    glue_fixes = [
        (r'anindependently', 'an independently'),
        (r'siteabout', 'site about'),
        (r'muchneeded', 'much needed'),
        (r'neededupdates', 'needed updates'),
        (r'publicon', 'public on'),
        (r'itsanticipated', 'its anticipated'),
        (r'GEMorg', 'GEM org'),
        (r'THEGEMIS', 'THE GEM IS'),
        (r'GrandEgyptianMuseumorg', 'Grand Egyptian Museum org'),
        (r'GLORIOUSpast', 'GLORIOUS past'),
    ]
    for pattern, replacement in glue_fixes:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    return text


def remove_special_characters(text):
    """
    Removes special characters and non-ASCII symbols.
    """
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Remove emojis
    return text


def clean_text(text):
    """
    Cleans the text by normalizing spaces, inserting spaces in glued words,
    and adding sentence boundaries.
    """
    text = insert_space_in_glued_words(text)
    text = remove_special_characters(text)

    sentence_markers = [
        r'(Feature \d)',
        r'(Read more)',
        r'(GEM Soft Opening)',
        r'(Sign Up to Receive Updates)',
        r'(is announced)',
        r'(sign up for our announcement list)',
        r'(will not only be)',
        r'(including nearly all artifact galleries)'
    ]
    for marker in sentence_markers:
        text = re.sub(marker, r'. \1.', text, flags=re.IGNORECASE)

    text = re.sub(r'\s*\.\s*', '. ', text)  # Normalize punctuation spacing
    text = re.sub(r'\.\s+\.', '.', text)  # Remove duplicate periods
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    return text


def postprocess_sentences(sentences):
    """
    Merges fragmented sentences and removes invalid ones.
    """
    processed = []
    buffer = ""

    for s in sentences:
        s = s.strip()
        if not s or len(s) < 8 or s in [".", ".."] or s[0].islower():
            buffer += " " + s
        else:
            if buffer:
                processed.append((buffer + " " + s).strip())
                buffer = ""
            else:
                processed.append(s)

    if buffer:
        processed.append(buffer.strip())

    return processed


def chunk_text(text, max_chunk_size=150):
    """
    Splits the text into smaller chunks based on sentence boundaries and punctuation.
    """
    raw_sentences = sent_tokenize(text)
    sentences = postprocess_sentences(raw_sentences)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        while len(sentence) > max_chunk_size:
            split_index = sentence.rfind(' ', 0, max_chunk_size)
            if split_index == -1:
                split_index = max_chunk_size

            # Ensure the split ends with punctuation
            punct_index = sentence.rfind('.', 0, max_chunk_size)
            if punct_index == -1:
                split_index = max_chunk_size
            else:
                split_index = punct_index + 1  # include punctuation


            chunks.append(sentence[:split_index].strip())
            sentence = sentence[split_index:].strip()

        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def filter_irrelevant_content(chunks, min_length=10, max_length=300):
    """
    Filters out irrelevant content based on length and context.
    """
    filtered = []
    for chunk in chunks:
        if len(chunk) < min_length or len(chunk) > max_length:
            continue
        if chunk.startswith(("http", "www.", "Sign up", "Skip to")):
            continue
        filtered.append(chunk)
    return filtered


def save_chunks_with_metadata(chunks, output_file):
    """
    Saves chunks along with metadata to a pickle file.
    """
    data = {
        "source": output_file,
        "total_chunks": len(chunks),
        "chunks": chunks
    }
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    logging.info("✅ Preprocessed data saved to '%s'.", output_file)


def main(input_file, output_file, chunk_size):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_text(raw_text)
    logging.info("🧽 Cleaned text preview:\n%s", repr(cleaned_text[:300]))

    sentences = postprocess_sentences(sent_tokenize(cleaned_text))
    logging.info("🧾 Total sentences: %d", len(sentences))

    chunks = chunk_text(cleaned_text, max_chunk_size=chunk_size)
    logging.info("🔖 Total chunks before filtering: %d", len(chunks))

    chunks = filter_irrelevant_content(chunks)
    logging.info("🔖 Total chunks after filtering: %d", len(chunks))

    # Save chunks with metadata
    save_chunks_with_metadata(chunks, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess website text for chunking.")
    parser.add_argument("--input", type=str, default="sample_data/example_website.txt", help="Input .txt file path")
    parser.add_argument("--output", type=str, default="preprocessed_data.pkl", help="Output .pkl file path")
    parser.add_argument("--chunk-size", type=int, default=150, help="Max chunk size in characters")

    args = parser.parse_args()

    try:
        main(args.input, args.output, args.chunk_size)
    except Exception as e:
        logging.error("❌ An error occurred: %s", e)