import os
import sys
import csv
from PIL import Image
from tqdm import tqdm

def process_images(root_dir):
    output_dir = os.path.join(root_dir, "resized")
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(root_dir, "images.csv")

    with open(csv_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path", "file_name"])

        for subdir, _, files in os.walk(root_dir):
            if subdir == output_dir:
                continue  
            pbar=tqdm(files)
            for file in pbar:
                if file.lower().endswith(".jpg"):
                    file_path = os.path.join(subdir, file)

                    try:
                        img = Image.open(file_path).convert("RGB")
                        img_resized = img.resize((224, 224), Image.LANCZOS)

                        output_path = os.path.join(output_dir, file)
                        img_resized.save(output_path, "JPEG")

                        writer.writerow([output_path, file])

                    except Exception as e:
                        print(f"Ошибка обработки {file_path}: {e}")

    print(f"Готово! Результаты сохранены в {output_dir}, список в {csv_path}")

if __name__ == "__main__":
    process_images("Pear/Pear/leaves")
