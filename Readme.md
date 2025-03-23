# Introduction to LLM
<p style="text-align: center;">
<img src="screenshots/gita_RAG_app_interface.png" alt="Sample Image" width="400" height="400">
</p>
This repository contains various Jupyter notebooks that demonstrate the use of Large Language Models (LLMs) for different tasks such as text summarization, podcast writing, and self-supervised training. Follow the instructions below to set up the environment and explore the notebooks.

## Environment Setup

1. **Create a virtual environment:**

    ```sh
    python -m venv llm_env
    ```

2. **Activate the virtual environment:**

    - On Windows:
        ```sh
        .\llm_env\Scripts\activate
        ```
    - On macOS and Linux:
        ```sh
        source llm_env/bin/activate
        ```

## Installation of Libraries

Make sure you have the following libraries installed:

- `torch`
- `transformers`
- `tqdm`
- `numpy`
- `sentence-transformers`
- `matplotlib`
- `PyPDF2`
- `IPython`

You can install them using the following command:

```sh
pip install torch transformers tqdm numpy sentence-transformers matplotlib PyPDF2 IPython
```

## How to Access the Llama 3.2 Model from Hugging Face [(video to access llama 3.2)](https://youtu.be/7HlHPWNS-20)

### To access Meta's Llama 3.2 model on Hugging Face, follow these steps:

1. Create a Hugging Face Account
2. Visit Hugging Face and sign up for an account.​
3. Navigate to the Llama 3.2 Model Repository
4. Go to the Meta Llama organization page on Hugging Face.​
5. Select the specific Llama 3.2 model you wish to access, such as Llama-3.2-1B 

### Request Access to the Model

1. On the model's page, click the "Request Access" button.​
2. Fill out the required information and agree to the license terms and acceptable use policy.​
3. Submit your request

### Set Up Your Hugging Face Authentication Token

After approval, generate a "READ" token in your Hugging Face account settings.
```python
from huggingface_hub import notebook_login
notebook_login()
```
This will prompt you to enter your access token directly within the notebook interface.



## Running the Jupyter Notebooks

1. **Start Jupyter Notebook:**

    ```sh
    jupyter notebook
    ```

2. **Open a notebook:**

    In the Jupyter interface, navigate to the notebook you want to open and click on it.

## Jupyter Notebooks


- kokoro_TTS.ipynb: Demonstrates text-to-speech conversion using a pre-trained model.

- llama_introduction.ipynb: Provides an introduction to using the LLaMA model for various text generation tasks.

- llama_meeting_summerizer.ipynb: Summarizes meeting transcripts using the LLaMA model.

- llama_podcastwriter_part_1.ipynb: The first part of a series that demonstrates how to use LLaMA for writing podcast scripts.

- llama_podcastwriter_part_2.ipynb: The second part of the series that continues the demonstration of using LLaMA for podcast script writing.

- llama_RAG.ipynb: Implements Retrieval-Augmented Generation (RAG) using the LLaMA model.

- SLM_instruction_finrtuning.ipynb: Shows how to fine-tune the LLaMA model for specific instructions and tasks.

- SLM_self_supervised_training.ipynb: Demonstrates self-supervised training techniques using the LLaMA model.