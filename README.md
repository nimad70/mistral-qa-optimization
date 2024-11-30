# Optimizing Question-Answering (QA) Performance in Large Language Models (LLMs)

## Project Overview
This project focuses on optimizing domain-specific question-answering (QA) systems using advanced fine-tuning and augmentation techniques. The goal is to enhance the performance of the Mistral 7B v0.2 model in medical HR applications, enabling accurate and efficient recruitment of medical professionals.

We evaluate three key approaches:
- **Parameter-Efficient Fine-Tuning (PEFT)**
- **Retrieval-Augmented Generation (RAG)**
- **Retrieval-Augmented Fine-Tuning (RAFT)**

## Features
- **Base Model**: Utilized the Mistral 7B v0.2 for its efficient architecture and strong NLP capabilities.
- **Fine-Tuning**: Implemented PEFT methods like LoRA and QLoRA for resource-efficient fine-tuning.
- **Retrieval-Augmented Systems**: Integrated RAG and RAFT to enhance response accuracy and context relevance.
- **Domain-Specific Dataset**: Trained and evaluated on the MedQuad-MedicalQnADataset.

## Prerequisites
- **Hugging Face Account**: Create an account [here](https://huggingface.co/).
- **Access Token**: Generate a write-permissions token from your Hugging Face account.

## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/nimad70/mistral-qa-optimization.git
   cd mistral-qa-optimization
   ```

2. Explore the repository for scripts and data relevant to your experiments.

## Setup Instructions

### 1. Creation of QA Datasets
- **Notebook**: `QADataset.ipynb`
- Steps:
  1. Run all cells to generate HR general and advanced medical QA datasets.
  2. Outputs are saved in both CSV and JSON formats.

### 2. Dataset Preparation
- **Notebook**: `NLPProject_Dataset_MedicalQA.ipynb`
- Steps:
  1. Log in to Hugging Face with the generated access token.
  2. Load the `MedQuad-MedicalQnADataset` or other preferred datasets.
  3. Adjust token limits and the `k` parameter for dataset size.
  4. Push the dataset to your Hugging Face repository.

### 3. Fine-Tuning the Model
- **Notebook**: `Mistral_7B_Instruct_v0_2_finetuning_medicaldb.ipynb`
- Steps:
  1. Define the base model (`Mistral 7B v0.2` by default).
  2. Customize fine-tuning parameters (temperature, top-p, etc.).
  3. Push the fine-tuned model to Hugging Face.

### 4. Querying Models
- **Notebooks**:
  - `NLPProject_Mistral_7B_Intrsuct_v02_Prompting_QA(P1).ipynb`
  - `NLPProject_Mistral_7B_Intrsuct_v02_Prompting_QA(P2-P5).ipynb`
- Steps:
  1. Upload the HR and advanced medical QA datasets.
  2. Customize precision (`4-bit` or `8-bit`) and pipeline parameters.
  3. Save generated responses for evaluation.

### 5. RAG System Implementation
- **Notebooks**:
  - `NLPProject_Mistral2_7B_RAG.ipynb`
  - `NLPProject_Mistral2_7B_RAG(FT).ipynb`
- Steps:
  1. Upload research papers and QA datasets to the `papers` directory in Colab.
  2. Customize pipeline parameters.
  3. Save responses for evaluation.

### 6. RAFT Integration and Evaluation
- **Notebook**: `Mistral_7B_Instruct_v0_2_finetuning_medicaldb.ipynb`
- Steps:
  1. Upload research papers to the `papers` directory in Colab.
  2. Save and evaluate generated responses.

## Key Results
- Accuracy improved from **84.09% to 95.45%**.
- Enhanced the model’s precision and contextual understanding through systematic tuning.

## Contributions
- **Nima Daryabar**
- **Shakiba Farjood Fashalami**
- **Marcos Tidball**
- **Tobia Pavona**
- **Giacomo Ferrante**

## Future Work
- Multilingual support.
- Real-time data integration.
- Scalability for larger LLMs.

## References
Refer to the detailed [Project Report](https://www.dropbox.com/scl/fi/4lritaoahwxgml7cvzzgu/NLP-project-report.pdf?rlkey=8pifa0ezaac8ttjjnyvxybygf&st=fqjaxn2d&dl=0) for more insights into the methodologies and evaluations.

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
