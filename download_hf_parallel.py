from huggingface_hub import HfApi
import os
import glob
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import List, Optional, Tuple
import logging
import requests
import os

log_file = "download_hf_parallel.log"
logging.basicConfig(filename=log_file, level=logging.INFO)

def check_file_size_local(file_path, file_size) -> bool:
    """
    Check if the file exists and has the correct size
    Will remove the file if the size is not correct
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        return False
    # remove the file if the size is not correct
    if os.path.getsize(file_path) != file_size:
        os.remove(file_path)
        return False
    return True

def download_chunk(url, start, end, path):
    if check_file_size_local(path, end - start + 1):
        logging.info(f"File {path} already exists, skipping download")
        return path
    headers = {'Range': f'bytes={start}-{end}'}
    try:
        r = requests.get(url, headers=headers, stream=True)
        with open(path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        # check if the file is downloaded correctly
        if not check_file_size_local(path, end - start + 1):
            logging.error(f"File {path} is not downloaded correctly, expected size {end - start + 1}, actual size {os.path.getsize(path)}")
            return None
        return path
    except Exception as e:
        logging.error(f"Error downloading chunk {path}: {str(e)}")
        return None

def download_chunks_parallel(requests_list):
    pbar_local = tqdm(total=len(requests_list), desc="Downloading chunks")
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_chunk = {executor.submit(download_chunk, url, start, end, path): (url, start, end, path) for url, start, end, path in requests_list}
        for future in as_completed(future_to_chunk):
            chunk_info = future_to_chunk[future]
            pbar_local.update(1)
            try:
                path = future.result()
                if path:
                    logging.info(f"Successfully downloaded {path}")
            except Exception as e:
                logging.error(f"Error in downloading chunk {chunk_info[-1]}: {str(e)}")

CHUNK_SIZE = 1024 * 1024 * 10  # 10MB
# Now, given the content-length, we can calculate the number of chunks to download
def generate_requests_for_chunks(repository: str, filename: str, chunk_size: int, download_path: str, repo_type:str = "dataset") -> Tuple[List[Tuple[str, int, int, str]], int, int]:
    url = f"https://huggingface.co/{repo_type}s/{repository}/resolve/main/{filename}"
    response = requests.head(url, allow_redirects=True)
    total_size = int(response.headers.get('content-length', 0))
    # check if file already exists and has the correct size
    if check_file_size_local(f"{download_path}/{filename}", total_size):
        logging.info(f"File {filename} already exists, skipping download")
        return [], 0, 0
    num_chunks = total_size // chunk_size + (total_size % chunk_size != 0)
    requests_list = []
    file_names = [f"{download_path}/{filename}_{i}" for i in range(num_chunks)]
    file_sizes = [chunk_size for _ in range(num_chunks)]
    file_sizes[-1] = total_size - (num_chunks - 1) * chunk_size
    for i, (file_name, file_size) in enumerate(zip(file_names, file_sizes)):
        if not check_file_size_local(file_name, file_size):
            requests_list.append((url, i * chunk_size, (i + 1) * chunk_size - 1, file_name))
    return requests_list, num_chunks, total_size

def merge_chunks(file_name, required_chunk_count, download_path, required_size) -> Optional[str]:
    files_match = f"{download_path}/{file_name}_*"
    files = sorted(glob.glob(files_match), key=lambda x: int(x.split('_')[-1]))
    if len(files) != required_chunk_count:
        logging.error(f"Expected {required_chunk_count} chunks, found {len(files)}. Aborting merge.")
        return None
    final_path = f"{download_path}/{file_name}"
    with open(final_path, "wb") as f:
        for file_path in files:
            with open(file_path, "rb") as chunk:
                f.write(chunk.read())
    if os.path.getsize(final_path) != required_size:
        logging.error(f"File {final_path} is not complete, required size {required_size}, actual size {os.path.getsize(final_path)}")
        return None
    for file in files:
        os.remove(file)
    logging.info(f"File {final_path} is downloaded and merged successfully")
    return final_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download files from Hugging Face Hub in parallel")
    parser.add_argument("--repo_id", type=str, help="Repository ID in the format 'owner/repo_name'", default="AngelBottomless/Danbooru-images-latents") #"AngelBottomless/Danbooru-images-latents"
    parser.add_argument("--repo_type", type=str, help="Repository type", default="dataset")
    parser.add_argument("--download_path", type=str, help="Path to download the files", default="G:/Danbooru-images-latents")
    parser.add_argument("--chunk_size", type=int, help="Chunk size in bytes, WARNING : If you change this, previous downloaded state will be lost", default=CHUNK_SIZE)
    parser.add_argument("--retry_count", type=int, help="Number of retries for file download", default=3)
    args = parser.parse_args()
    api = HfApi()
    repository = args.repo_id
    files_list = (api.list_repo_files(
        repo_id=repository,
        repo_type=args.repo_type
    ))
    download_path = args.download_path # Path to download the files
    # Assume the functions `generate_requests_for_chunks`, `download_chunks_parallel`, and `merge_chunks` are defined as described in the previous response.
    CHUNK_SIZE = 1024 * 1024 * 10  # 10MB
    for file in tqdm(files_list, desc="Downloading files"):
        for i in range(args.retry_count):
            requests_list, num_chunks, total_size = generate_requests_for_chunks(repository, file, CHUNK_SIZE, download_path, repo_type=args.repo_type)
            if requests_list:  # Proceed only if there are chunks to download
                download_chunks_parallel(requests_list)
                result = merge_chunks(file, num_chunks, download_path, total_size)
                if result:
                    break
                else:
                    logging.error(f"Error in merging chunks for file {file}, retrying {i+1}/{args.retry_count}")
            else:
                logging.info(f"File {file} already exists, skipping download")
                break
        else:
            logging.error(f"Failed to download file {file} after {args.retry_count} retries")
    input("Press Enter to continue...")
