import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_website(url):
    """Scrape data from the specified website."""
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract data as needed, for example:
        data = []
        for item in soup.find_all('div', class_='item-class'):  # Adjust the selector as needed
            title = item.find('h2').text
            link = item.find('a')['href']
            data.append({'title': title, 'link': link})
        return data
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return []

def call_api(api_url):
    """Call the API and retrieve data."""
    response = requests.get(api_url)
    if response.status_code == 200:
        return response.json()  # Assuming the API returns JSON data
    else:
        print(f"Failed to call API: {response.status_code}")
        return []

def save_to_csv(data, filename):
    """Save the collected data to a CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    website_url = 'https://example.com'  # Replace with the actual website URL
    api_url = 'https://jsonplaceholder.typicode.com/posts'  # Using a valid API URL for demonstration


    scraped_data = scrape_website(website_url)
    api_data = call_api(api_url)

    # Combine or process data as needed
    combined_data = scraped_data + api_data  # Adjust as necessary

    save_to_csv(combined_data, 'output_data.csv')
    print("Data has been saved to output_data.csv")
