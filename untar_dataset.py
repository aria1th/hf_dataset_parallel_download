# untars the file and extracts the contents to the specified directory
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import tarfile
import argparse
import logging

log_file = "untar_dataset.log"
logging.basicConfig(filename=log_file, level=logging.INFO,encoding='utf-8', filemode='w')

def extract_tarfile(file_path, extract_path, remove_after_extract=False) -> None:
    """
    Extracts the tar file to the specified directory
    """
    try:
        with tarfile.open(file_path, 'r', dereference=True) as tar:
            tar.extractall(extract_path)
        logging.info(f"Extracted {file_path} to {extract_path}")
        if remove_after_extract:
            os.remove(file_path)
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            logging.error(f"Keyboard interrupt: {str(e)}")
            raise e
        logging.error(f"Error extracting {file_path}: {str(e)}")

def main(dataset_folder, result_folder, max_parallel_actions, remove_after_extract: bool = False):
    """
    Extracts all the tar files in the specified folder to the result folder
    """
    file_lists = os.listdir(dataset_folder)
    file_lists = [file for file in file_lists if file.endswith('.tar')]
    print(file_lists)
    jobs = []
    result_folder_base = result_folder
    with ThreadPoolExecutor(max_workers=max_parallel_actions) as executor:
        pbar = tqdm(total=len(file_lists), desc="Extracting files")
        for file in file_lists:
            result_folder = os.path.join(result_folder_base, file.split('.')[0])
            jobs.append(executor.submit(extract_tarfile, os.path.join(dataset_folder, file), result_folder, remove_after_extract))
        for job in as_completed(jobs):
            try:
                job.result()
                pbar.update(1)
            except Exception as e:
                if isinstance(e, KeyboardInterrupt):
                    logging.error(f"Keyboard interrupt: {str(e)}")
                    raise e
                logging.error(f"Error extracting file: {str(e)}")
    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts all the tar files in the specified folder to the result folder")
    parser.add_argument("--dataset_folder", type=str, required=True, help="The folder containing the tar files")
    parser.add_argument("--result_folder", type=str, required=True, help="The folder where the tar files will be extracted")
    parser.add_argument("--max_parallel_actions", type=int, default=4, help="The maximum number of parallel actions")
    parser.add_argument("--remove_after_extract", action="store_true", help="Remove the tar file after extracting")
    args = parser.parse_args()
    main(args.dataset_folder, args.result_folder, args.max_parallel_actions, args.remove_after_extract)
