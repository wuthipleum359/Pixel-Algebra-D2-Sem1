import requests, os

API_KEY = ""
BASE_URL = "https://pixabay.com/api/"

def fetch_images(query, count=50):
    params = {
        "key": API_KEY,
        "q": query,
        "image_type": "photo",
        "per_page": count,
        "safesearch": "true"
    }
    r = requests.get(BASE_URL, params=params)
    data = r.json()
    return [hit["largeImageURL"] for hit in data["hits"]]

def download_images(query, folder, count=50):
    os.makedirs(folder, exist_ok=True)
    urls = fetch_images(query, count)
    for i, url in enumerate(urls):
        filename = os.path.join(folder, f"{query}_{i+1}.jpg")
        img = requests.get(url)
        with open(filename, "wb") as f:
            f.write(img.content)
        print(f"ดาวน์โหลด {filename}")

download_images("plants", "images/plants", 30)


download_images("buildings", "images/buildings", 30)
download_images("people", "images/people", 30)
download_images("food", "images/food", 30)
download_images("environment", "images/environment", 30)
