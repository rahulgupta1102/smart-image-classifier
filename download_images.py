from icrawler.builtin import BingImageCrawler
import os

classes = ["cats", "dogs", "cars"]

# Train + Val folders
base_dir = "dataset/train"
val_dir = "dataset/val"

def download_images(folder, limit=30):
    os.makedirs(folder, exist_ok=True)
    for cls in classes:
        print(f"📸 Downloading {limit} images of {cls} in {folder}...")
        cls_folder = os.path.join(folder, cls)
        os.makedirs(cls_folder, exist_ok=True)

        crawler = BingImageCrawler(storage={'root_dir': cls_folder})
        crawler.crawl(keyword=cls, max_num=limit)

# Download training images
download_images(base_dir, limit=100)

# Download validation images
download_images(val_dir, limit=20)

print("✅ All images downloaded successfully!")