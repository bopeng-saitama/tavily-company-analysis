import os
import sys
import time
from tavily import CompanyAnalyzer

"""
This example demonstrates how to generate a comprehensive company analysis report
using the CompanyAnalyzer class with OpenAI integration for enhanced content quality.

The report is structured according to the 14 key business analysis areas:
1. Company Executives
2. Corporate Philosophy
3. Establishment, Capital, IPO, Business Locations
4. Detailed Business Content
5. Performance
6. Growth Potential
7. Economic Impact Sensitivity
8. Competitive Advantages
9. Corporate Culture
10. Career Development Environment
11. Job Positions
12. Working Conditions
13. CSR Activities/Diversity Initiatives
14. Related Companies
"""

def generate_company_report(
    company_name, 
    language="ja", 
    output_format="markdown", 
    use_openai=True,
    openai_model="gpt-4",
    verbose=True
):
    """
    Generate and save a company analysis report.
    
    Args:
        company_name: Name of the company to analyze
        language: Language for analysis ("ja" for Japanese, "en" for English)
        output_format: Format of the report ("text", "markdown", "html", "json")
        use_openai: Whether to use OpenAI for enhancing the report quality
        openai_model: OpenAI model to use (e.g., "gpt-4", "gpt-3.5-turbo")
        verbose: Whether to display detailed progress information
    
    Returns:
        Path to the saved report file
    """
    # Get API keys from environment variables
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY") if use_openai else None
    
    # Display configuration information
    print("\n" + "="*80)
    print("COMPANY ANALYSIS CONFIGURATION")
    print("="*80)
    print(f"Company: {company_name}")
    print(f"Language: {language}")
    print(f"Output format: {output_format}")
    print(f"OpenAI integration: {'Enabled' if use_openai else 'Disabled'}")
    if use_openai:
        print(f"OpenAI model: {openai_model}")
    print(f"Verbose output: {'Enabled' if verbose else 'Disabled'}")
    print(f"Tavily API key: {'Set' if tavily_api_key else 'Not set'}")
    print(f"OpenAI API key: {'Set' if openai_api_key else 'Not set'}")
    print("="*80 + "\n")
    
    if not tavily_api_key:
        print("ERROR: TAVILY_API_KEY environment variable is not set.")
        print("Please set your Tavily API key and try again.")
        sys.exit(1)
    
    if use_openai and not openai_api_key:
        print("WARNING: OpenAI integration is enabled but OPENAI_API_KEY is not set.")
        print("The analysis will proceed without OpenAI enhancement.")
        use_openai = False
    
    # Initialize the CompanyAnalyzer
    print("Initializing CompanyAnalyzer...")
    analyzer = CompanyAnalyzer(
        tavily_api_key=tavily_api_key,
        openai_api_key=openai_api_key,
        model=openai_model,
        use_openai=use_openai
    )
    
    # Start timing the analysis
    start_time = time.time()
    
    # Analyze the company
    analysis_data = analyzer.analyze_company(
        company_name, 
        language=language,
        verbose=verbose
    )
    
    # Generate a report
    report = analyzer.generate_report(
        analysis_data, 
        format=output_format,
        verbose=verbose
    )
    
    # Save the report to a file
    filename = analyzer.save_report(
        report, 
        company_name, 
        format=output_format,
        verbose=verbose
    )
    
    # Calculate total processing time
    total_time = time.time() - start_time
    
    # Display summary
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Company: {company_name}")
    print(f"Report file: {filename}")
    print(f"Total sections: {len(analysis_data['sections'])}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print("="*80)
    
    return filename

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a company analysis report")
    parser.add_argument("company_name", help="Name of the company to analyze")
    parser.add_argument("--language", choices=["ja", "en"], default="ja", 
                        help="Language for analysis (ja=Japanese, en=English)")
    parser.add_argument("--format", choices=["text", "markdown", "html", "json"], 
                        default="markdown", help="Output format for the report")
    parser.add_argument("--use-openai", action="store_true", default=True,
                        help="Use OpenAI for enhanced report quality")
    parser.add_argument("--no-openai", action="store_false", dest="use_openai",
                        help="Disable OpenAI integration")
    parser.add_argument("--openai-model", default="chatgpt-4o-latest",
                        help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Display detailed progress information")
    parser.add_argument("--quiet", action="store_false", dest="verbose",
                        help="Hide detailed progress information")
    
    args = parser.parse_args()
    
    generate_company_report(
        args.company_name, 
        args.language, 
        args.format, 
        args.use_openai,
        args.openai_model,
        args.verbose
    )