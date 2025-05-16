import requests
from bs4 import BeautifulSoup
import re
import os
import json

# Configure logging
import logging
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

def fetch_html(url: str) -> BeautifulSoup:
    """
    Fetches HTML content from a URL and returns a BeautifulSoup object.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        return BeautifulSoup(response.content, "html.parser")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch {url}: {e}")
        raise


def extract_visible_text(soup: BeautifulSoup) -> str:
    """
    Extracts visible text from a BeautifulSoup object.
    """
    # Remove unwanted elements (scripts, styles, footers, etc.)
    for tag in soup(['script', 'style', 'footer', 'nav', 'form', 'noscript']):
        tag.decompose()

    # Join all stripped strings (already clean visible text)
    texts = soup.find_all(string=True)
    visible_texts = filter(is_visible_text, texts)
    return ' '.join(visible_texts)


def extract_images(soup: BeautifulSoup) -> list:
    """
    Extracts image URLs and alt text from the page.
    """
    images = []
    for img in soup.find_all("img"):
        src = img.get("src")
        alt = img.get("alt", "")
        if src:
            images.append({"url": src, "alt": alt})
    return images


def extract_metadata(soup: BeautifulSoup) -> dict:
    """
    Extracts metadata such as title, description, and keywords.
    """
    metadata = {}
    metadata["title"] = soup.title.string.strip() if soup.title else ""
    metadata["description"] = soup.find("meta", attrs={"name": "description"})
    metadata["keywords"] = soup.find("meta", attrs={"name": "keywords"})
    return metadata


def is_visible_text(element):
    """
    Filters out invisible or irrelevant tags.
    """
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'footer', 'nav']:
        return False
    if isinstance(element, str):
        return True
    return False


def clean_text(text: str) -> str:
    """
    Cleans text: removes duplicates, fixes punctuation, and whitespace.
    """
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text)  # Remove repeated words
    text = re.sub(r'(?<=[a-zA-Z])\.(?=\s*[A-Z])', '. ', text)  # Fix missing spaces after dots
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text


def save_to_file(data: dict, filename: str):
    """
    Saves data to a JSON file.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    logging.info(f"Data saved to '{filename}'.")


def scrape_website(url: str, output_path: str):
    """
    Scrapes a website and saves the cleaned text, metadata, and images to a JSON file.
    """
    logging.info(f"🔍 Scraping content from: {url}")
    try:
        soup = fetch_html(url)

        # Extract visible text
        combined_text = extract_visible_text(soup)
        cleaned_text = clean_text(combined_text)

        # Extract metadata
        metadata = extract_metadata(soup)

        # Extract images
        images = extract_images(soup)

        # Save scraped data
        data = {
            "url": url,
            "text": cleaned_text,
            "metadata": {
                "title": metadata.get("title", ""),
                "description": metadata.get("description", {}).get("content", "") if metadata.get("description") else "",
                "keywords": metadata.get("keywords", {}).get("content", "") if metadata.get("keywords") else ""
            },
            "images": images
        }
        save_to_file(data, output_path)

    except Exception as e:
        logging.error(f"❌ Error during scraping: {e}")


if __name__ == "__main__":
    # Define the target URL and output path
    url = "https://grandegyptianmuseum.org/"
    output_path = "sample_data/example_website.json"

    # Scrape the website
    scrape_website(url, output_path)