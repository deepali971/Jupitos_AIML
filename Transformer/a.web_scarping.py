import requests
import csv
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

def scrape_books_to_csv(url, csv_filename):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        data = []
        for book in soup.find_all('article', class_='product_pod'):
            title = book.h3.a['title']  # Get title from the 'a' tag inside h3
            price = book.find('p', class_='price_color').text  # Get price
            data.append({'title': title, 'price': price})

        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'price']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(data)

        print(f"Book data scraped and stored in {csv_filename}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the website: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage
website_url = "http://books.toscrape.com/catalogue/page-1.html"  # URL of the first page
output_csv_file = "books_data2.csv"
scrape_books_to_csv(website_url, output_csv_file)