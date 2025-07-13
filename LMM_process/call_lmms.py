# -----------------------------------------------------------------------------
# Nationally Representative, Locally Misaligned: The Biases of Generative
# Artificial intelligence in Neighborhood Perception
# -----------------------------------------------------------------------------
# Authors: Paige Bollen, Joe Higton, and Melissa L Sands
#
# Description:
# This script allows users to re-run the data collection tool used to generate the
# Generative AI (GenAI) model ratings analyzed in the paper. It is designed
# to systematically query large multimodal models (LMMs) with a set of street-view
# images and specific questions about neighborhood characteristics (e.g.,
# wealth, safety, disorder).
# -----------------------------------------------------------------------------

import httpx
import os
import base64
import json
import csv
import pandas as pd
import random
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
import datetime
import argparse
from tqdm import tqdm
from collections import Counter
import logging
import sys
import time
import time
import threading
from threading import Lock
from collections import deque
import concurrent.futures
from together import Together


# Set up argument parser
parser = argparse.ArgumentParser(prog='gpt_images.py', description='Label images with gpt api')
parser.add_argument('--image_filename', metavar='FILENAME', help='csv with image urls in "url" column')
parser.add_argument('--api_key', metavar='FILENAME', help='api key for gpt')
parser.add_argument('--json_skip', metavar='FILENAME', help='json file with image ids to skip', default=None)
parser.add_argument('--questions_file', metavar='FILENAME', help='text file with each question on a separate line')
parser.add_argument('--one_image', action='store_true', help='if set, only one image will be processed')
parser.add_argument('--explanation', action='store_true', help='if set, model will exlpain rating')
parser.add_argument('--model', help='Model to use', required=True)
parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO', help='Set the logging level')
parser.add_argument('--workers', type=int, default=10, help='Number of parallel workers (default: 10)')
parser.add_argument('--number', type=int, default=30, help='Number of queries per question.')
parser.add_argument('--together', action='store_true')
parser.add_argument('--rpm', type=int, default=None,  help='Maximum requests per minute (e.g., 60 for 60 requests/minute)')
parser.add_argument('--batch_size', type=int, default=5, help='Number of images to process before saving results (default: 5)')
args = parser.parse_args()

# Set up timestamp for file naming
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
script_start_time = time.time()

# Configure logging
safe_model_name = args.model.replace('/', '-')
log_filename = f"{timestamp}_{safe_model_name}_log.txt"
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Locks for thread-safe operations
json_lock = Lock()
stats_lock = Lock()

# Initialize statistics tracking
stats = {
    "total_images": 0,
    "processed_images": 0,
    "skipped_images": 0,
    "failed_images": 0,
    "retried_images": 0,
    "successful_retries": 0,
    "total_api_calls": 0,
    "successful_api_calls": 0,
    "failed_api_calls": 0,
    "questions_count": 0,
    "errors_by_type": Counter(),
    "errors_by_image": Counter(),
    "start_time": datetime.datetime.now(),
}

logger.info(f"Script started with model: {safe_model_name}")
logger.info(f"Arguments: {vars(args)}")
logger.info(f"Using {args.workers} parallel workers")

class RequestsPerMinuteLimiter:
    def __init__(self, max_rpm):
        self.max_rpm = max_rpm
        self.request_timestamps = deque(maxlen=max_rpm)
        self.lock = threading.Lock()
    
    def wait(self):
        if not self.max_rpm:
            return  # No rate limiting
        
        with self.lock:
            current_time = time.time()
            
            # Remove timestamps older than 1 minute
            while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
                self.request_timestamps.popleft()
            
            # If we've made max_rpm requests in the last minute, wait until we can make another
            if len(self.request_timestamps) >= self.max_rpm:
                # Calculate how long to wait until the oldest request is 1 minute old
                wait_time = 60 - (current_time - self.request_timestamps[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
            
            # Record this request's timestamp
            self.request_timestamps.append(time.time())

# After parsing arguments, create a rate limiter instance
rate_limiter = RequestsPerMinuteLimiter(args.rpm) if args.rpm else None
logger.info(f"Rate limit set to {args.rpm} requests/minute" if args.rpm else "No rate limiting applied")

# Initialize API client - one per worker to avoid conflicts
def create_client():
    if "gpt" in args.model:
        return OpenAI(api_key=args.api_key)
    elif "gemini" in args.model:
        return OpenAI(
            api_key=args.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    else: return Together(api_key = args.api_key)

# Get model name based on args
def get_model_name(for_retry=False):
    return args.model

# Load required files, exiting if any fail
try:
    questions = [line.strip() for line in open(args.questions_file, 'r') if line.strip()]
    stats["questions_count"] = len(questions)
    logger.info(f"Loaded {len(questions)} questions.")

    image_urls = pd.read_csv(args.image_filename)['url'].tolist()
    stats["total_images"] = len(image_urls)
    logger.info(f"Loaded {len(image_urls)} image URLs.")
except Exception as e:
    logger.critical(f"FATAL: Could not load required questions or image file. Error: {e}")
    sys.exit(1)

# Handle one-image mode if enabled
if args.one_image:
    image_urls = image_urls[:1]
    stats["total_images"] = 1
    logger.info("One-image mode: processing only the first image.")

# Load optional skip file
skip_image_ids = set()
existing_data = []
if args.json_skip:
    try:
        with open(args.json_skip, 'r') as f:
            skip_data = json.load(f)
        image_id_counter = Counter(entry['Image ID'] for entry in skip_data)
        skip_image_ids = {img_id for img_id, count in image_id_counter.items() if count >= 5}
        existing_data = skip_data
        stats["skipped_images"] = len(skip_image_ids)
        logger.info(f"Identified {len(skip_image_ids)} image IDs to skip.")
    except Exception as e:
        logger.warning(f"Could not parse skip file. Continuing without skipping. Error: {e}")

# Define response models based on args
if args.explanation:
    class IncomeGuess(BaseModel):
        response: int  # Ensures the output is strictly an integer
        explanation: str  # Explanation for the response
    logger.info("Using response model with explanation field")
else:
    class IncomeGuess(BaseModel):
        response: int  # Ensures the output is strictly an integer
    logger.info("Using response model without explanation field")

# Convert args to a dictionary and remove the 'api_key'
args_dict = vars(args).copy()
args_dict.pop('api_key', None)  # Remove 'api_key' from the dictionary

# Write the remaining arguments to 'arguments_used.txt'
with open(f"{timestamp}_args.txt", 'w') as f:
    f.write(f"Timestamp: {timestamp}\n")
    for key, value in args_dict.items():
        f.write(f"{key}: {value}\n")

# Set up base64 encoding for gemini 
def get_data_uri(image_url):
    try:
        logger.debug(f"Downloading image from {image_url}")
        image = httpx.get(image_url)
        encoded_image = base64.b64encode(image.content).decode('utf-8')
        
        # Determine the MIME type based on the file extension
        ext = image_url.split('.')[-1].lower()
        if ext in ['jpg', 'jpeg']:
            mime = 'image/jpeg'
        elif ext == 'png':
            mime = 'image/png'
        elif ext == 'gif':
            mime = 'image/gif'
        else:
            mime = 'image/jpeg'  # Default MIME type
        
        data_uri = f"data:{mime};base64,{encoded_image}"
        logger.debug(f"Successfully converted image to data URI with MIME type {mime}")
        return data_uri
    except httpx.RequestError as e:
        logger.error(f"Failed to download image from {image_url}: {str(e)}")
        with stats_lock:
            stats["errors_by_type"]["image_download"] = stats["errors_by_type"].get("image_download", 0) + 1
        return None
    except Exception as e:
        logger.error(f"Unexpected error processing image {image_url}: {str(e)}")
        with stats_lock:
            stats["errors_by_type"]["image_processing"] = stats["errors_by_type"].get("image_processing", 0) + 1
        return None

# Data storage
data_for_json = []
invalid_urls = []
processed_batches = 0

# Function to save results to JSON file
def save_results_to_json():
    filename = args.json_skip if args.json_skip else f"{timestamp}_{safe_model_name}.json"
    try:
        with open(filename, 'w') as jsonfile:
            json.dump(existing_data if args.json_skip else data_for_json, jsonfile, indent=4)
        logger.info(f"Updated JSON data in {filename}")
    except Exception as e:
        logger.error(f"Failed to write to JSON file {filename}: {str(e)}")
        with stats_lock:
            stats["errors_by_type"]["json_write_error"] = stats["errors_by_type"].get("json_write_error", 0) + 1

# Process a single attempt for a question-image pair
def process_single_attempt(client, model_name, question, data_uri, image_id, question_index, attempt):
    logger.debug(f"Attempt {attempt+1}/{args.number} for question {question_index+1} and image {image_id}")
    with stats_lock:
        stats["total_api_calls"] += 1

    # Apply rate limiting if configured
    if rate_limiter:
        rate_limiter.wait()
    
    with stats_lock:
        stats["total_api_calls"] += 1
    
    try:
        if "gemini" in args.model:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question}, 
                        {"type": "image_url", 
                        "image_url": {
                            "url": data_uri, 
                            "detail": "high"
                        }}]
                }],
                temperature = 1, #default
                response_format=IncomeGuess,
            )
        elif "gpt" in args.model or "o4" in args.model or "o3" in args.model:
            completion = client.beta.chat.completions.parse(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question}, 
                        {"type": "image_url", 
                        "image_url": {
                            "url": data_uri, 
                            "detail": "high"
                        }}]
                }],
                temperature = 1, #default
                response_format=IncomeGuess,
            )
        elif args.together:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question}, 
                        {"type": "image_url", 
                        "image_url": {
                            "url": data_uri, 
                            "detail": "high"
                        }}]
                }],
                response_format={
                    "type": "json_object",
                    "schema": IncomeGuess.model_json_schema(),
                },
                temperature = 0.7, #default
            )

        with stats_lock:
            stats["successful_api_calls"] += 1
        
        # Attempt to parse and access the response
        try:
            response_val = None
            explanation_val = None

            # Handle Together API's JSON string response
            if args.together:
                raw_response = completion.choices[0].message.content
                try:
                    parsed_data = json.loads(raw_response)
                    response_val = parsed_data.get("response")
                    explanation_val = parsed_data.get("explanation")
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from Together API for Q{question_index+1}/I{image_id}: {raw_response}")
                    with stats_lock:
                        stats["errors_by_type"]["json_decode_error"] += 1
                    return "JSON Decode Error"
            
            # Handle OpenAI/Gemini's Pydantic object response
            else:
                parsed_object = completion.choices[0].message.parsed
                if parsed_object:
                    response_val = parsed_object.response
                    explanation_val = getattr(parsed_object, 'explanation', None)
                else:
                    # If parsing by the client library failed, log the raw content
                    raw_response = completion.choices[0].message.content
                    logger.warning(f"Model returned unparsable content for Q{question_index+1}/I{image_id}: {raw_response}")
                    with stats_lock:
                        stats["errors_by_type"]["unparsed_response"] += 1
                    return "Unparsed Response"

            # Check extracted values and return the appropriate result
            if args.explanation:
                if response_val is not None and explanation_val is not None:
                    logger.debug(f"Response {attempt+1} for Q{question_index+1}/I{image_id}: {response_val} with explanation")
                    return {"response": response_val, "explanation": explanation_val}
                
                logger.warning(f"Incomplete response for Q{question_index+1}/I{image_id}. Response: '{response_val}', Explanation: '{explanation_val}'")
                with stats_lock:
                    stats["errors_by_type"]["incomplete_response"] += 1
                return "Incomplete Response"
            
            # If no explanation is expected
            if response_val is not None:
                logger.debug(f"Response {attempt+1} for Q{question_index+1}/I{image_id}: {response_val}")
                return response_val
            
            logger.warning(f"Missing response field for Q{question_index+1}/I{image_id}")
            with stats_lock:
                stats["errors_by_type"]["missing_response_field"] += 1
            return "Missing Response Field"

        except AttributeError as e:
            # Catches issues like `completion.choices` being empty or having an unexpected structure
            logger.error(f"Attribute error while parsing response for Q{question_index+1}/I{image_id}: {str(e)}")
            with stats_lock:
                stats["failed_api_calls"] += 1
                stats["errors_by_type"]["attribute_error"] += 1
                stats["errors_by_image"][image_id] += 1
            return "Parsing Error"
    
    except AttributeError as e:
        # If any other parsing issue arises, handle it gracefully
        logger.error(f"Attribute error for Q{question_index+1}/I{image_id}: {str(e)}")
        with stats_lock:
            stats["failed_api_calls"] += 1
            stats["errors_by_type"]["attribute_error"] = stats["errors_by_type"].get("attribute_error", 0) + 1
            stats["errors_by_image"][image_id] = stats["errors_by_image"].get(image_id, 0) + 1
        return "Parsing Error"
    
    except Exception as e:
        # Catch any other exceptions that might occur
        logger.error(f"Unexpected error for Q{question_index+1}/I{image_id}: {str(e)}")
        with stats_lock:
            stats["failed_api_calls"] += 1
            stats["errors_by_type"]["api_call_error"] = stats["errors_by_type"].get("api_call_error", 0) + 1
            stats["errors_by_image"][image_id] = stats["errors_by_image"].get(image_id, 0) + 1
        return "Error"

# Process all attempts for a question-image pair (X attempts)
def process_question_image_pair(client, model_name, question, data_uri, image_id, question_index, for_retry=False):
    logger.debug(f"Processing question {question_index+1}/{len(questions)} for image {image_id}")
    
    # Process attempts in parallel within each question-image pair
    responses = []
    prefix = "Retry: " if for_retry else ""
    
    # Create a ThreadPoolExecutor for parallel processing of attempts
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, args.workers)) as executor:
        # Submit all X attempts to the executor
        future_to_attempt = {
            executor.submit(
                process_single_attempt, client, model_name, question, data_uri, image_id, question_index, attempt
            ): attempt for attempt in range(args.number)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_attempt):
            attempt = future_to_attempt[future]
            try:
                result = future.result()
                responses.append(result)
            except Exception as e:
                logger.error(f"{prefix}Error in attempt {attempt+1} for Q{question_index+1}/I{image_id}: {str(e)}")
                responses.append("Error")
                with stats_lock:
                    stats["errors_by_type"]["attempt_execution_error"] = stats["errors_by_type"].get("attempt_execution_error", 0) + 1
    
    logger.info(f"{prefix}Completed {args.number} attempts for Q{question_index+1}/I{image_id}")
    
    return {
        "Question": question,
        "Image ID": image_id,
        "Responses": responses
    }

# Process a single image (all questions)
def process_image(url_index, url, total_urls, retry=False):
    image_id = url.split('/')[-1].split('.')[0]
    remaining = total_urls - url_index - 1
    
    prefix = "Retry: " if retry else ""
    logger.info(f"{prefix}Processing image {image_id} ({url_index+1}/{total_urls}), {remaining} images left")
    
    if image_id in skip_image_ids:
        logger.info(f"{prefix}Skipping image {image_id} as it is in the skip list")
        return None, False
    
    if 'Detroit_processed' in url:
        url = url.replace('Detroit_processed', 'for_gpt')
        logger.debug(f"{prefix}URL modified to: {url}")
    
    # Initialize client
    try:
        client = create_client()
        model_name = get_model_name(retry)
    except Exception as e:
        logger.error(f"{prefix}Failed to initialize client for image {image_id}: {str(e)}")
        with stats_lock:
            stats["errors_by_type"]["client_init_error"] = stats["errors_by_type"].get("client_init_error", 0) + 1
        return None, True  # Return None and mark as invalid for retry
    
    # Prepare data URI for Gemini or direct URL for GPT
    if "gemini" in args.model and not retry:
        # Convert the image URL to a data URI
        data_uri = get_data_uri(url)
        if not data_uri:
            logger.warning(f"{prefix}Failed to process image {image_id}, adding to invalid URLs for retry")
            with stats_lock:
                stats["failed_images"] += 1
                stats["errors_by_image"][image_id] = stats["errors_by_image"].get(image_id, 0) + 1
            return None, True  # Return None and mark as invalid for retry
    else:
        data_uri = url  # For GPT models, use the direct URL
        logger.debug(f"{prefix}Using direct URL for model: {url}")
    
    image_results = []
    image_success = True
    
    try:
        # Process each question for this image
        for question_index, question in enumerate(questions):
            result = process_question_image_pair(
                client, model_name, question, data_uri, image_id, question_index, retry
            )
            image_results.append(result)
            
            # Add to global results with thread safety
            with json_lock:
                data_for_json.append(result)
                if args.json_skip:
                    existing_data.append(result)
    
    except Exception as e:
        logger.error(f"{prefix}Error processing image {image_id}: {str(e)}")
        with stats_lock:
            stats["failed_images"] += 1
            stats["errors_by_type"]["image_processing_error"] = stats["errors_by_type"].get("image_processing_error", 0) + 1
            stats["errors_by_image"][image_id] = stats["errors_by_image"].get(image_id, 0) + 1
        image_success = False
        return None, True  # Return None and mark as invalid for retry if not already a retry
    
    # Update statistics
    if image_success:
        with stats_lock:
            if retry:
                stats["successful_retries"] += 1
            stats["processed_images"] += 1
            
    return image_results, not image_success

# Process images in batches with parallel execution
def process_image_batch(batch_urls, batch_indices, total_urls, retry=False):
    batch_results = []
    batch_invalids = []
    
    # Create a ThreadPoolExecutor for parallel processing of images
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all images in batch to the executor
        future_to_url = {
            executor.submit(process_image, idx, url, total_urls, retry): (idx, url) 
            for idx, url in zip(batch_indices, batch_urls)
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_url):
            idx, url = future_to_url[future]
            try:
                results, invalid = future.result()
                if results:
                    batch_results.extend(results)
                if invalid and not retry:
                    batch_invalids.append(url)
            except Exception as e:
                logger.error(f"Error in batch processing for URL index {idx}: {str(e)}")
                if not retry:
                    batch_invalids.append(url)
                with stats_lock:
                    stats["errors_by_type"]["batch_processing_error"] = stats["errors_by_type"].get("batch_processing_error", 0) + 1
    
    return batch_results, batch_invalids

# Main processing logic
def main():
    global processed_batches, invalid_urls
    
    # Filter out skipped images
    filtered_urls = []
    filtered_indices = []
    
    for idx, url in enumerate(image_urls):
        image_id = url.split('/')[-1].split('.')[0]
        if image_id not in skip_image_ids:
            filtered_urls.append(url)
            filtered_indices.append(idx)
        else:
            logger.debug(f"Filtering out skipped image {image_id}")
    
    logger.info(f"Processing {len(filtered_urls)} images after filtering out skipped images")
    
    # Process in batches
    for i in range(0, len(filtered_urls), args.batch_size):
        batch_urls = filtered_urls[i:i+args.batch_size]
        batch_indices = filtered_indices[i:i+args.batch_size]
        
        logger.info(f"Processing batch {processed_batches+1}, images {i+1}-{min(i+args.batch_size, len(filtered_urls))} of {len(filtered_urls)}")
        
        _, batch_invalids = process_image_batch(batch_urls, batch_indices, len(filtered_urls))
        invalid_urls.extend(batch_invalids)
        
        # Save results after each batch
        with json_lock:
            save_results_to_json()
        
        processed_batches += 1
        logger.info(f"Completed batch {processed_batches}, found {len(batch_invalids)} invalid URLs")
    
    # Process invalid URLs in parallel
    if invalid_urls:
        logger.info(f"Retrying {len(invalid_urls)} invalid URLs...")
        with stats_lock:
            stats["retried_images"] = len(invalid_urls)
        
        # Retry in smaller batches for better error handling
        retry_batch_size = max(1, min(5, args.batch_size // 2))
        
        retry_batches = 0
        for i in range(0, len(invalid_urls), retry_batch_size):
            batch_urls = invalid_urls[i:i+retry_batch_size]
            batch_indices = list(range(i, min(i+retry_batch_size, len(invalid_urls))))
            
            logger.info(f"Processing retry batch {retry_batches+1}, images {i+1}-{min(i+retry_batch_size, len(invalid_urls))} of {len(invalid_urls)}")
            
            process_image_batch(batch_urls, batch_indices, len(invalid_urls), retry=True)
            
            # Save results after each retry batch
            with json_lock:
                save_results_to_json()
            
            retry_batches += 1
            logger.info(f"Completed retry batch {retry_batches}")

# Convert data to CSV format
def generate_csv():
    logger.info("Generating CSV output from collected data")
    
    csv_data = []
    try:
        for item in data_for_json:
            for response in item["Responses"]:
                question = item['Question']

                # Directly assign the category based on the question content
                if 'wealth' in question.lower():
                    category = "Wealth"
                elif 'daylight' in question.lower():
                    category = "Safety - day"
                elif 'disorder' in question.lower():
                    category = "Disorder"
                elif 'dark' in question.lower():
                    category = "Safety - night"
                else:
                    category = "Other"

                csv_entry = {
                    "Question": category,
                    "question_text": question,
                    "Image ID": item['Image ID'],
                    "Response": response
                }
                csv_data.append(csv_entry)

        # Convert extracted data to CSV
        csv_filename = f"{timestamp}_{safe_model_name}.csv"
        with open(csv_filename, 'w', newline='') as csvfile:
            fieldnames = ["Question", "question_text", "Image ID", "Response"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for entry in csv_data:
                writer.writerow(entry)
        logger.info(f"Successfully wrote CSV data to {csv_filename}")
    except Exception as e:
        logger.error(f"Error generating CSV output: {str(e)}")
        with stats_lock:
            stats["errors_by_type"]["csv_generation_error"] = stats["errors_by_type"].get("csv_generation_error", 0) + 1

# Generate execution summary
def generate_summary():
    # Calculate script runtime
    script_end_time = time.time()
    runtime_seconds = script_end_time - script_start_time
    hours, remainder = divmod(runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    runtime_formatted = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    with stats_lock:
        stats["runtime"] = runtime_formatted
        stats["runtime_seconds"] = runtime_seconds

    # Generate execution summary
    summary_filename = f"{timestamp}_{safe_model_name}_summary.txt"
    with open(summary_filename, 'w') as summary_file:
        summary_file.write("==== EXECUTION SUMMARY ====\n\n")
        summary_file.write(f"Timestamp: {timestamp}\n")
        summary_file.write(f"Model: {safe_model_name}\n")
        summary_file.write(f"Runtime: {runtime_formatted}\n")
        summary_file.write(f"Parallel workers: {args.workers}\n")
        summary_file.write(f"Batch size: {args.batch_size}\n\n")
        
        summary_file.write("== PROCESSING STATISTICS ==\n")
        summary_file.write(f"Total images: {stats['total_images']}\n")
        processed_pct = stats['processed_images']/stats['total_images']*100 if stats['total_images'] > 0 else 0
        summary_file.write(f"Images processed: {stats['processed_images']} ({processed_pct:.2f}%)\n")
        summary_file.write(f"Images skipped: {stats['skipped_images']}\n")
        summary_file.write(f"Images failed: {stats['failed_images']}\n")
        summary_file.write(f"Images retried: {stats['retried_images']}\n")
        summary_file.write(f"Successful retries: {stats['successful_retries']}\n")
        summary_file.write(f"Questions per image: {stats['questions_count']}\n\n")
        
        summary_file.write("== API CALL STATISTICS ==\n")
        summary_file.write(f"Total API calls: {stats['total_api_calls']}\n")
        success_rate = stats['successful_api_calls'] / stats['total_api_calls'] * 100 if stats['total_api_calls'] > 0 else 0
        summary_file.write(f"Successful API calls: {stats['successful_api_calls']} ({success_rate:.2f}%)\n")
        summary_file.write(f"Failed API calls: {stats['failed_api_calls']}\n")
        if stats['total_api_calls'] > 0 and runtime_seconds > 0:
            calls_per_second = stats['total_api_calls'] / runtime_seconds
            summary_file.write(f"API calls per second: {calls_per_second:.2f}\n\n")
        
        summary_file.write("== ERROR BREAKDOWN ==\n")
        if stats["errors_by_type"]:
            for error_type, count in sorted(stats["errors_by_type"].items(), key=lambda x: x[1], reverse=True):
                summary_file.write(f"{error_type}: {count}\n")
        else:
            summary_file.write("No errors recorded\n")
        
        summary_file.write("\n== PROBLEMATIC IMAGES ==\n")
        if stats["errors_by_image"]:
            for image_id, count in sorted(stats["errors_by_image"].items(), key=lambda x: x[1], reverse=True)[:10]:
                summary_file.write(f"{image_id}: {count} errors\n")
        else:
            summary_file.write("No problematic images recorded\n")
        
        summary_file.write("\n== FILES GENERATED ==\n")
        summary_file.write(f"Log file: {log_filename}\n")
        summary_file.write(f"JSON output: {args.json_skip if args.json_skip else f'{timestamp}_{safe_model_name}.json'}\n")
        summary_file.write(f"CSV output: {timestamp}_{safe_model_name}.csv\n")
        summary_file.write(f"Args file: {timestamp}_args.txt\n")
        summary_file.write(f"Summary file: {summary_filename}\n")
        
        summary_file.write("\n== PARALLEL PROCESSING ==\n")
        summary_file.write(f"Workers: {args.workers}\n")
        summary_file.write(f"Batches processed: {processed_batches}\n")
        summary_file.write(f"Batch size: {args.batch_size}\n")
        
# Print summary to console
    logger.info("\n" + open(summary_filename, 'r').read())
    logger.info("Script execution completed")

# Main execution
if __name__ == "__main__":
    try:
        # Execute the main processing
        main()
        
        # Generate CSV from collected data
        generate_csv()
        
        # Generate and print summary
        generate_summary()
    except KeyboardInterrupt:
        logger.warning("Script interrupted by user")
        logger.info("Saving partial results...")
        with json_lock:
            save_results_to_json()
        generate_summary()
        logger.info("Partial results saved")
    except Exception as e:
        logger.critical(f"Unhandled exception in main execution: {str(e)}")
        logger.info("Attempting to save partial results...")
        with json_lock:
            save_results_to_json()
        generate_summary()
        logger.info("Partial results saved")
        raise