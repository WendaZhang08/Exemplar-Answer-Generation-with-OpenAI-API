# Exemplar Answer Generation Project

A Python application that leverages **OpenAI's GPT models** (GPT-4o mini) to automatically generate high-quality exemplar answers for educational assessment questions. This project is developed in collaboration with [Cura Education](https://www.curaeducation.com/), who provided both the training dataset (`cura-llm-training-data.json`) and OpenAI API access.

## Background 

Cura Education hosts an extensive library of online courses designed for student engagement and learning. Within these courses, students complete various tasks and submit responses to assessment questions. Exemplar answers are crucial for teachers as they provide clear benchmarks for evaluating student submissions.

## Project Overview

- Processes educational task content, questions, and assessment rubrics
- Generates aligned exemplar answers using OpenAI API
- Includes evaluation metrics for answer quality
- Contains automated testing suite
- Supports educational assessment workflows


## Tech Stack
- Python
- OpenAI API
- pytest

## Setup and Installation

### Prerequisites
- Python 3.9+
- Conda environment manager (recommended)

---

### Environment Setup

1. Clone the repository

```bash
git clone [https://github.com/WendaZhang08/Exemplar-Answer-Generation-with-OpenAI-API.git]
cd Exemplar-Answer-Generation-with-OpenAI-API
```

2. Create and activate conda environment

```bash
conda create -n exemplar-env python=3.9
conda activate exemplar-env
```

3. Install required packages

```bash
pip install -r requirements.txt
```

## Running the Project

1. **Data Analysis**
   Open and run `Section1_Data_Analysis.ipynb` to perform data analysis.

2. **OpenAI Integration**
   Open and run `Section2_OpenAI_Integration.py` to perform OpenAI Integration.

3. **Testing**
   Execute the test suite:

```bash
pytest test_openai_handler.py -v
```

## Project Structure

### 1. Data Analysis Section (`Section1_Data_Analysis.ipynb`)

This notebook contains comprehensive analysis of the training dataset used for generating exemplar answers. The analysis is structured as follows:

#### Environment Setup
- Initial setup of required dependencies including **pandas**, **numpy**, **matplotlib**, and **seaborn**
- Configuration of visualization settings and data processing utilities

#### Data Loading and Processing
- Loading of training data from JSON format
- Basic data validation and structure examination
- Conversion to pandas DataFrame for analysis

#### Exploratory Data Analysis
- Statistical analysis of key characteristics:
  - Question length distribution (majority between 0-200 characters)
  - Answer length patterns (concentrated between 100-400 characters)
  - Task content analysis (ranging from 0 to 40,000 characters)
  - Rubric scoring distribution (focused on 2-3 point scales)
  
#### Visualization
- Distribution plots for:
  - Question lengths
  - Answer lengths
  - Task content lengths
  - Rubric total scores

#### Key Findings
- Questions tend to be concise with occasional longer descriptions
- Exemplar answers follow a balanced length distribution
- Task content varies significantly in length based on topic complexity
- Assessment rubrics show standardized scoring patterns

This analysis provides crucial insights for:
- Understanding data patterns and characteristics
- Identifying potential preprocessing requirements
- Informing the development of answer generation strategies
- Establishing baseline metrics for evaluation

The structured analysis aids in developing a solution for generating appropriate exemplar answers that align with the existing patterns in the training data.

---

### 2. OpenAI Integration Section (`Section2_OpenAI_Integration.ipynb`)

This section implements the core functionality for generating exemplar answers using OpenAI's API, featuring a class hierarchy for API integration and evaluation:

#### Environment Setup
- Installation of required dependencies:
    - Core ML libraries: **numpy**, **scikit-learn**
    - OpenAI API related: **openai**, **tiktoken** 
    - NLP tools: **nltk**, **sentence-transformers**
    - Data processing: **pandas**
    - Visualization: **matplotlib**, **seaborn**

#### Data Processing
- Loading preprocessed training data
- Data cleaning and standardization
 - HTML entity removal
 - Text normalization
 - Length truncation for efficient processing
- Training/validation data split preparation
- Data formatting for API consumption

#### Implementation Architecture

1. Base OpenAI Handler
- API initialization and token management
- Basic tokenization functionality
- Token usage tracking and statistics

2. Prompt Handler
- Multiple prompt template implementations
- Context formatting for API requests
- Template optimization logic

3. Generation Handler
- Answer generation functionality
- Retry mechanism for API calls
- Token usage optimization

4. Training Handler
- Prompt template selection and optimization
- Best example selection for few-shot learning
- Training process management

5. Evaluation Handler

- Comprehensive evaluation metrics implementation
   - Content relevance scoring
   - Rubric alignment assessment
   - Semantic similarity calculation
   - Answer structure analysis
   - Reasoning depth evaluation
- K-fold cross-validation implementation
- Correlation analysis between metrics

#### Evaluation Metrics

The evaluation system includes multiple dimensions:

- Basic Metrics
  - Quality scoring (0-1 scale)
  - Rubric alignment (0-1 scale)
  - Semantic similarity with reference answers

- Detailed Metrics
  - Length ratio analysis
  - Keyword coverage
  - Structure similarity
  - Reasoning depth assessment

#### Visualization Components
- Performance metrics across folds
- Correlation heatmaps
- Metric distribution analysis
- Cross-fold performance trends

![combined-plot](https://github.com/COMP90054-2024s2/a3-zhuoyoudeking/blob/master/img/experiment1-middlegames2.png)

![cross-fold-performance-trends-plot](https://github.com/COMP90054-2024s2/a3-zhuoyoudeking/blob/master/img/experiment1-middlegames1.png)

#### Analysis of Evaluation Results
>[!Note]
> Full analysis available in Section 3.3 of `Section2_OpenAI_Integration.ipynb`

The evaluation system demonstrates the model's performance with:
- Average Quality Score: **0.513 ± 0.019**
- Rubric Alignment: **0.641 ± 0.017**
- Semantic Similarity: **0.671 ± 0.041**

and: 
- Length Ratio: **9.142 ± 5.161**
- Keyword Coverage: **0.559 ± 0.021**
- Structure Similarity: **0.539 ± 0.087**
- Reasoning Depth: **0.770 ± 0.027**

The evaluation reveals several key insights about the model's performance:

- **Robust Performance Metrics**:
  - Strong semantic alignment (0.671 ± 0.041)
  - Consistent rubric adherence (0.641 ± 0.017)
  - Stable quality scores (0.513 ± 0.019)

- **Generalization Capabilities**:
  - Consistent performance across k-fold validation
  - Stable metrics across different question types
  - Reliable pattern recognition in varied content areas

- **Areas for Optimization**:
  - Length ratio variability needs improvement
  - Trade-off between rubric alignment and natural language fluency
  - Potential for enhanced domain-specific terminology handling

- **Evaluation Framework Strengths**:
  - Multi-dimensional assessment approach
  - Rigorous scoring methodology
  - Comprehensive correlation analysis between metrics

For detailed analysis and visualizations, refer to [Section 3.3: The Analysis of the Evaluation Results](Section2_OpenAI_Integration.ipynb#3.3-The-Analysis-of-the-Evaluation-Results)

