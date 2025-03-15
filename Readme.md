# Introduction to LLM

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
