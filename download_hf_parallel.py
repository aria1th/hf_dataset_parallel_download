from huggingface_hub import HfApi
import os
import glob
from tqdm import tqdm
from concurrent.futures import as_completed, ThreadPoolExecutor
from typing import List, Optional, Tuple
import logging
import requests
import os
import threading
import tarfile
TOKEN = ""

log_file = "download_hf_parallel.log"
logging.basicConfig(filename=log_file, level=logging.INFO,encoding='utf-8', filemode='w')

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
        logging.error(f"File {file_path} is not complete, required size {file_size}, actual size {os.path.getsize(file_path)}")
        os.remove(file_path)
        return False
    return True

def download_chunk(url, start, end, path):
    if check_file_size_local(path, end - start + 1):
        logging.info(f"File {path} already exists, skipping download")
        return path
    headers = {'Range': f'bytes={start}-{end}'}
    if TOKEN:
        headers['Authorization'] = f"Bearer {TOKEN}"
    should_use_pbar = end - start >= 1024 * 1024 * 80
    if should_use_pbar:
        pbar = tqdm(total=end - start + 1, desc=f"Downloading {path}", unit="B", unit_scale=True)
    for _i in range(3):
        try:
            r = requests.get(url, headers=headers, allow_redirects=True, stream=True)
            r.raise_for_status()
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    if should_use_pbar:
                        pbar.update(len(chunk))
            # check if the file is downloaded correctly
            if not check_file_size_local(path, end - start + 1):
                print(f"File {path} is not downloaded correctly, expected size {end - start + 1}, actual size {os.path.getsize(path)}.. retrying {_i+1}..")
                logging.error(f"File {path} is not downloaded correctly, expected size {end - start + 1}, actual size {os.path.getsize(path)}")
                continue
            return path
        except Exception as e:
            logging.error(f"Error downloading chunk {path}: {str(e)}")
    return None

def download_chunks_parallel(requests_list):
    pbar_local = tqdm(total=len(requests_list), desc="Downloading chunks")
    with ThreadPoolExecutor(max_workers=32) as executor:
        future_to_chunk = {executor.submit(download_chunk, url, start, end, path): (url, start, end, path) for url, start, end, path in requests_list}
        for future in as_completed(future_to_chunk):
            chunk_info = future_to_chunk[future]
            pbar_local.update(1)
            try:
                path = future.result()
                if path:
                    logging.info(f"Successfully downloaded {path}")
            except Exception as e:
                print(f"Error in downloading chunk {chunk_info[-1]}: {str(e)}")
                logging.error(f"Error in downloading chunk {chunk_info[-1]}: {str(e)}")

CHUNK_SIZE = 1024 * 1024 * 10  # 10MB
# Now, given the content-length, we can calculate the number of chunks to download
def generate_requests_for_chunks(repository: str, filename: str, chunk_size: int, download_path: str, repo_type:str = "dataset", result_dir:str = None, no_chunks:bool = False
                                 ) -> Tuple[List[Tuple[str, int, int, str]], int, int]:
    url = f"https://huggingface.co/{repo_type}s/{repository}/resolve/main/{filename}"
    if repo_type == "model":
        url = f"https://huggingface.co/{repository}/resolve/main/{filename}"
    if TOKEN:
        headers = {'Authorization': f"Bearer {TOKEN}"}
    else:
        headers = {}
    response = requests.head(url, allow_redirects=True, headers=headers)
    total_size = int(response.headers.get('content-length', 0))
    # check if file already exists and has the correct size
    if check_file_size_local(f"{download_path}/{filename}", total_size) or (result_dir and check_file_size_local(f"{result_dir}/{filename}", total_size)):
        logging.info(f"File {filename} already exists, skipping download")
        return [], 0, 0
    download_path = download_path or result_dir
    if no_chunks:
        return [(url, 0, total_size - 1, f"{download_path}/{filename}")], 1, total_size
    num_chunks = total_size // chunk_size + (total_size % chunk_size != 0)
    requests_list = []
    file_names = [f"{download_path}/{filename}_{i}" for i in range(num_chunks)]
    file_sizes = [chunk_size for _ in range(num_chunks)]
    file_sizes[-1] = total_size - (num_chunks - 1) * chunk_size
    if file_sizes[-1] <= 0:
        file_sizes.pop()
        file_names.pop()
        num_chunks -= 1
    for i, (file_name, file_size) in enumerate(zip(file_names, file_sizes)):
        if not check_file_size_local(file_name, file_size):
            requests_list.append((url, i * chunk_size, min((i + 1) * chunk_size - 1, file_size + i * chunk_size - 1), file_name))
    return requests_list, num_chunks, total_size

def merge_chunks(file_name, required_chunk_count, download_path, required_size, dest_dir) -> Optional[str]:
    files_match = f"{download_path}/{file_name}_*"
    matched = glob.glob(files_match)
    if dest_dir:
        matched += glob.glob(f"{dest_dir}/{file_name}_*")
    files = sorted(matched, key=lambda x: int(x.split('_')[-1]))
    if len(files) != required_chunk_count:
        print(f"Expected {required_chunk_count} chunks, found {len(files)}. Aborting merge.")
        logging.error(f"Expected {required_chunk_count} chunks, found {len(files)}. Aborting merge.")
        return None
    final_path = f"{dest_dir}/{file_name}"
    with open(final_path, "wb") as f:
        for file_path in tqdm(files, desc="Merging chunks"):
            with open(file_path, "rb") as chunk:
                f.write(chunk.read())
    if os.path.getsize(final_path) != required_size:
        logging.error(f"File {final_path} is not complete, required size {required_size}, actual size {os.path.getsize(final_path)}")
        return None
    for file in files:
        os.remove(file)
    logging.info(f"File {final_path} is downloaded and merged successfully")
    return final_path

def auto_untar(file_path, dest_dir, remove_after_untar=False):
    try:
        file_base = os.path.basename(file_path)
        file_base = file_base[:file_base.rfind(".")]
        os.makedirs(dest_dir + "/" + file_base, exist_ok=True)
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(dest_dir + "/" + file_base)
        print(f"Untarred {file_path} to {dest_dir + '/' + file_base}")
        logging.info(f"Untarred {file_path} to {dest_dir + '/' + file_base}")
        if remove_after_untar:
            os.remove(file_path)
    except Exception as e:
        print(f"Error in untarring {file_path}: {str(e)}")
        logging.error(f"Error in untarring {file_path}: {str(e)}")

def download_for_file(repository, file, chunk_size, download_path, repo_type, result_dir, pbar, no_chunks, auto_untar_dir=None, remove_after_untar=False):
    if file in fully_finished:
        logging.info(f"File {file} already exists, skipping download (fully finished)")
        pbar.update(1)
        if os.path.exists(f"{download_path}/{file}") or (result_dir and os.path.exists(f"{result_dir}/{file}")):
            # auto untar
            if auto_untar_dir:
                filepath = f"{download_path}/{file}" if not result_dir else f"{result_dir}/{file}"
                print(f"Auto untarring {filepath} to {auto_untar_dir}")
                auto_untar(filepath, auto_untar_dir, remove_after_untar)
                return None
            return f"{download_path}/{file}" if not result_dir else f"{result_dir}/{file}"
        else:
            return None
    for _i in range(10):
        requests_list, num_chunks, total_size = generate_requests_for_chunks(repository, file, chunk_size, download_path if not no_chunks else result_dir, repo_type, result_dir, no_chunks)
        if requests_list:  # Proceed only if there are chunks to download
            download_chunks_parallel(requests_list)
            if not no_chunks:
                print(f"Downloaded {file} in {download_path}, merging chunks...")
                merge_result = merge_chunks(file, num_chunks, download_path, total_size, result_dir)
                if merge_result:
                    pbar.update(1)
                    return merge_result
                else:
                    print(f"Error in merging chunks for {file}, retrying {_i+1}..")
                    logging.error(f"Error in merging chunks for {file}, retrying {_i+1}..")
            else:
                print(f"Downloaded {file} in {download_path} without splitting into chunks")
                pbar.update(1)
                return f"{download_path}/{file}"
        else:
            logging.info(f"File {file} already exists, skipping download")
            with open(fully_finished_files_path, "a", encoding="utf-8") as f:
                f.write(f"{file}\n")
            pbar.update(1)
            filepath = f"{download_path}/{file}" if not result_dir else f"{result_dir}/{file}"
            if auto_untar_dir:
                print(f"Auto untarring {filepath} to {auto_untar_dir}")
                auto_untar(filepath, auto_untar_dir, remove_after_untar)
                return None
            return filepath
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download files from Hugging Face Hub in parallel")
    parser.add_argument("--repo_id", type=str, help="Repository ID in the format 'owner/repo_name'", default="AngelBottomless/Danbooru-images-latents") #"AngelBottomless/Danbooru-images-latents"
    parser.add_argument("--repo_type", type=str, help="Repository type", default="dataset")
    parser.add_argument("--download_path", type=str, help="Path to download the files", default="G:/Danbooru-images-latents")
    # cache_dir
    parser.add_argument("--cache_dir", type=str, help="Cache directory", default="F:/Danbooru-images-latents")
    parser.add_argument("--chunk_size", type=int, help="Chunk size in bytes, WARNING : If you change this, previous downloaded state will be lost", default=CHUNK_SIZE)
    parser.add_argument("--auth_token", type=str, help="Hugging Face API token", default="")
    parser.add_argument("--no_chunks", action="store_true", help="Download the file without splitting into chunks", default=False)
    # auto_untar
    parser.add_argument("--auto_untar_dir", type=str, help="Directory to auto untar the downloaded files", default="")
    # remove_after_untar
    parser.add_argument("--remove_after_untar", action="store_true", help="Remove the tar file after untarring", default=False)
    #parser.add_argument("--retry_count", type=int, help="Number of retries for file download", default=3)
    args = parser.parse_args()
    if args.auth_token:
        print("Using auth token")
        TOKEN = args.auth_token
    api = HfApi(token=TOKEN)
    repository = args.repo_id
    files_list = (api.list_repo_files(
        repo_id=repository,
        repo_type=args.repo_type
    ))
    fully_finished_files_path = f"{args.repo_id.split('/')[1]}_fully_finished.txt"
    fully_finished = set()
    if os.path.exists(fully_finished_files_path):
        with open(fully_finished_files_path, "r", encoding="utf-8") as f:
            fully_finished = set(f.read().split("\n"))
            fully_finished = set([x for x in fully_finished if x])
            print(f"Found {len(fully_finished)} fully finished files")
    download_path = args.download_path
    CHUNK_SIZE = 1024 * 1024 * 100  # 100MB
    jobs = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        pbar = tqdm(total=len(files_list), desc="Downloading files")
        for file in files_list:
            #download_for_file(repository, file, CHUNK_SIZE, args.cache_dir or download_path, args.repo_type, download_path)
            jobs.append(executor.submit(download_for_file, repository, file, CHUNK_SIZE, args.cache_dir or download_path, args.repo_type, download_path, pbar, args.no_chunks, args.auto_untar_dir, args.remove_after_untar))
        for job in as_completed(jobs):
            result = job.result()
            if result:
                print(f"Downloaded {result} successfully")
                logging.info(f"Downloaded {result} successfully")
