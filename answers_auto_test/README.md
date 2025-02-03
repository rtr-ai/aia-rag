# Answer Auto Test

## Overview
The `answer_auto_test` application automates the comparison of legal expert-generated answers with AI-generated responses from the AiAct chatbot. The process is designed to analyze the differences and similarities between human and AI-generated answers using OpenAI's comparison model.

## Workflow
1. **Data Fetching**: The application first retrieves the ground truth answers from `fragen-qa.md`, a document created by RTR legal experts.
2. **AI Answer Generation**: The app triggers an API call to the AiAct chatbot to generate answers for the same set of questions.
3. **Comparison**: OpenAI's model is used to evaluate and compare both answers, identifying similarities and differences.
4. **Processing Duration**: The entire process takes approximately **25 minutes**.
5. **User Interface**: After processing, a **UI browser** is opened, allowing users to click on any question and view the comparison results.
6. **Report Export**: Users can download a comprehensive Excel report containing the analyzed questions and their corresponding comparison results.

## Installation & Setup
Clone the repository and install the dependencies:

```sh
# Clone the repository
git clone --branch auto_test https://BojanMakivic@github.com/rtr-ai/aia-rag.git
cd aia-rag

# Install dependencies
pip install -r requirements.txt
```

## Running the Application
Execute the `app.py` script to start the comparison process:

```sh
python app.py
```

Once the processing is complete, the browser UI will open automatically for interactive review.

## Features
- Automated **comparison** of legal expert and AI-generated answers.
- **UI browser** for easy question-wise result exploration.
- **Excel report export** with full analysis.
