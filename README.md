# Tavily Company Analysis Tool

A tool for generating detailed company analysis reports for job interviews, applications, and corporate research. Built with Python and Streamlit, leveraging the Tavily search API with optional LLM integration.

## Features

- **Comprehensive Company Analysis**: Analyze companies based on key aspects including executives, philosophy, business model, performance, and more
- **Multiple Languages**: Support for both Japanese and English analysis
- **LLM Integration**: Optional integration with OpenAI or SiliconFlow LLMs for enhanced content refinement
- **Multiple Output Formats**: Generate reports in Markdown, HTML, Text, or JSON
- **Interactive UI**: User-friendly Streamlit interface with real-time progress tracking
- **Customizable Sections**: Select specific sections to include in your analysis
- **Source Control**: Options to prefer official sources and exclude specific domains
- **Motivation Points**: Generates job application motivation points (希望動機) based on company analysis

## Installation

### Prerequisites

- Python 3.7+
- Tavily API key (get from [Tavily](https://tavily.com))
- Optional: OpenAI or SiliconFlow API key for LLM integration

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/bopeng-saitama/tavily-company-analysis.git
   cd tavily-company-analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

Or use the launch script:

```bash
python launch.py
```

### Configuration

1. Enter your API keys in the Settings page
2. Configure your preferences:
   - Select language (Japanese or English)
   - Choose output format (Markdown, HTML, Text, JSON)
   - Select LLM provider and model (optional)
   - Configure source control options
   - Select which sections to include in the analysis

### Generating Reports

1. Enter the company name
2. Configure analysis options
3. Click "Start Analysis"
4. Track progress in real-time
5. View and download the generated report

All reports are saved in the `reports` folder for easy access.

## Key Components

- **CompanyAnalyzer**: Core analysis engine that collects and processes company information
- **Streamlit UI**: Interactive interface for configuring and tracking analysis
- **Tavily API Integration**: Uses the Tavily search API to gather company information
- **LLM Integration**: Optional LLM processing through OpenAI or SiliconFlow for enhanced content

## Project Structure

```
tavily-company-analysis/
├── app.py                # Main Streamlit application
├── launch.py             # Launch script
├── reports/              # Generated reports directory
├── requirements.txt      # Project dependencies
├── tavily/               # Tavily API wrapper
│   ├── __init__.py
│   ├── company_analyzer.py  # Company analysis engine
│   ├── config.py
│   ├── errors.py
│   ├── tavily.py         # Core Tavily client
│   └── utils.py
└── LICENSE               # MIT License
```

## API Usage

The tool uses the following APIs:
- **Tavily API**: For web search and information gathering
- **OpenAI API** (optional): For content refinement and summarization
- **SiliconFlow API** (optional): Alternative LLM provider

## License

This project is licensed under the MIT License - see the LICENSE file for details.
