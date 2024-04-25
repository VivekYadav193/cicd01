import requests
import shutil
from tqdm import tqdm

urls = [
    ("https://pjreddie.com/media/files/mnist_train.csv", "data/train_data.csv"),
    ("https://pjreddie.com/media/files/mnist_test.csv", "data/test_data.csv")
]

for url, output_file in urls:
    # Download the file
    print(f"Downloading {output_file.split('/')[-1]}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(output_file, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("Failed to download the file completely.")
    else:
        print(f"Downloaded {output_file} successfully!")
