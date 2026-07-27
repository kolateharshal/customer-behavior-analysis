import os
import urllib.request
import time

# Create dataset directories
categories = ["apple", "banana", "orange"]
base_dir = "dataset"

print("Creating directories...")
for cat in categories:
    os.makedirs(os.path.join(base_dir, cat), exist_ok=True)
    print(f"Created folder: {base_dir}/{cat}")

# Download 20 images for each category
images_per_category = 20
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.0.0 Safari/537.36'}

for cat in categories:
    print(f"\nDownloading {images_per_category} images for: {cat}...")
    for i in range(1, images_per_category + 1):
        url = f"https://loremflickr.com/320/320/{cat},fruit?random={i}"
        filename = os.path.join(base_dir, cat, f"{cat}_{i:02d}.jpg")
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                with open(filename, 'wb') as f:
                    f.write(response.read())
            print(f"Downloaded {filename}")
            time.sleep(0.5) # Sleep to avoid spamming the server
        except Exception as e:
            print(f"Error downloading image {i} for {cat}: {e}")

print("\nDone! Dataset is ready in the 'dataset' directory.")
