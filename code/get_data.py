# Get the data which will be used for training
# Save that data under the ./data directory

import requests
import shutil

urls = [
    ("https://pjreddie.com/media/files/mnist_train.csv", "data/train_data.csv"),
    ("https://pjreddie.com/media/files/mnist_test.csv", "data/test_data.csv")
]

for url, output_file in urls:
    # Download the file
    print(f"Downloading {output_file.split('/')[-1]}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Save the downloaded file with a new name
        with open(output_file, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        print(f"Downloaded {output_file} successfully!")
    else:
        print(f"Failed to download {output_file}: {response.status_code}")