# Nationally Representative, Locally Misaligned: The Biases of Generative Artificial Intelligence in Neighborhood Perception
**Paige Bollen, Joe Higton, Melissa Sands**

This is the code repository for 'Nationally Representative, Locally Misaligned: The Biases of Generative Artificial Intelligence in Neighborhood Perception' available at [Political Analysis](https://www.cambridge.org/core/journals/political-analysis).

## Key Findings
- ✅ GenAI aligns well with **national** perspectives 
- ❌ GenAI poorly represents **local** residents 
- 🎯 Performance varies by demographic groups and neighborhood attributes
- 🗺️ [**Explore interactive results**](https://www.joehigton.com/genai_local_bias_app.html)

| | |
|:---:|:---:|
| <img src="https://joehigton.com/Detroit_images/985413675361949_2.jpg" width="500"/> | <img src="https://joehigton.com/Detroit_images/339901604398274_2.jpg" width="500"/> |

We ask whose perspectives GenAI aligns with more closely: the generalized perspectives of the national sample or the localized, context-sensitive views of residents. We analyse ratings of 85 streetview images across Detroit from a US sample, a Detroit sample, and ratings from 5 leading Large Multimodal Models. Examining perceptions of neighborhood safety, wealth, and disorder reveals a clear bias in GenAI toward national averages over local perspectives.

## Replication guide
Our analysis is contained in `analysis.R` which will reproduce all of the plots and tables from the paper.

We also include code to enable the re-running of our LMM analysis, in `lmm_calls.py`. With an API key from OpenAI, Google Gemini, or Together.ai, users can easily re-label the images with any of the models included in our paper or any [open source models hosted by Together.ai](https://docs.together.ai/docs/serverless-models). URLs for our images are available in `LMM_process/images_links_coordinates.csv`. 

### Re-running our analysis 
There is one R script: analysis.R. This creates all of the tables and figures in the paper and the supplementary information. The tables are outputted into a folder called 'tables' and the figures into a folder called 'plots' in /results/. 

#### LMM data
Files in the `[model]_[month]_run.csv` are runs of each model at a certain point in time as described in the paper. The model is the LMM used. 

`gemini1.5_repeat_test.csv` - this is a later repeat of `gemini1.5_november_run.csv` to check consistency over time. It is used in fig_si11.png. 

`houses_cars_per_image.csv` - this reports an LMM run counting houses and cars per image as in SI section S9.1.

`images_with_coordinates.csv` - this reports each image's coordinates and a URL to the image. It is used in fig_si9.png. 

#### Survey data
`dmacs_data.csv` - this is one of two main datasets used in the paper, the Detroit sample.

`prolific_data.csv` - this is the other of two main datasets used in the paper, the US sample. 

`pilot_data_order_analysis.csv` - this is pilot data from Prolific used to analyse the effect of image order (S7).

`pilot_data_power_analysis.csv` - this is pilot data used to test how many labelers are needed for the variance to stabilise (S8).


### Labeling our images with an LMM
We also provide code that facilitates the re-labelling of our images using a Large Multimodal Model (LMM). To do this, you can use the `call_lmms.py` script located in the `LMM_process/` directory.

#### Environment Setup
The Python environment is managed with `uv`. To set it up, first install `uv`, then create and sync the virtual environment.

#### Running the Script
The script is called from the command line. You must provide your API key and the model you wish to use. The code also requires a text file with questions/prompts, and a .csv file with images. The ones we used are provided here.

Here is an example command:
```bash
python call_lmms.py --image_filename images_with_coordinates.csv --api_key YOUR_API_KEY --questions_file questions.txt --model gemini-2.5-flash --workers 5 --batch_size 3 --log_level INFO
```
**Key Arguments:**
* `--image_filename`: CSV file containing image URLs. 
* `--api_key`: Your API key for the chosen service (e.g., Google, OpenAI).
* `--questions_file`: A text file with one question/prompt per line.
* `--model`: The specific model to use (e.g., `gpt-4o`, `gemini-1.5-pro`, needs to align with API model inputs).
* `--workers`: Number of parallel API calls to make.
* `--batch_size`: Number of images to process before saving results.
* `--log_level`: Sets the verbosity of the log file (e.g., `INFO`, `DEBUG`).



