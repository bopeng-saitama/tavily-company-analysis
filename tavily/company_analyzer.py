import json
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
import os
import time
import sys
from urllib.parse import urlparse
import requests

from .tavily import TavilyClient

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class CompanyAnalyzer:
    """
    A class for analyzing companies and generating comprehensive reports.
    Uses Tavily's search capabilities to gather company information and LLM services
    for content refinement and synthesis.
    """

    def __init__(
        self, 
        tavily_client: Optional[TavilyClient] = None, 
        llm_client: Optional["OpenAI"] = None,
        tavily_api_key: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        model: str = "gpt-4",
        use_llm: bool = True,
        llm_provider: str = "openai",
        base_url: Optional[str] = None
    ):
        """
        Initialize the CompanyAnalyzer with TavilyClient and LLM clients.

        Args:
            tavily_client: An existing TavilyClient instance. If not provided, a new one will be created.
            llm_client: An existing OpenAI client instance. If not provided, a new one will be created.
            tavily_api_key: Tavily API key. Required if tavily_client is not provided.
            llm_api_key: LLM API key. Required if llm_client is not provided and use_llm is True.
            model: LLM model to use. Default is "gpt-4".
            use_llm: Whether to use LLM for content refinement. Default is True.
            llm_provider: Provider for LLM services ("openai" or "siliconflow"). Default is "openai".
            base_url: Base URL for API requests. Required for siliconflow, optional for openai.
        """
        # Initialize Tavily client
        if tavily_client is None:
            if tavily_api_key is None:
                tavily_api_key = os.getenv("TAVILY_API_KEY")
                if not tavily_api_key:
                    raise ValueError("Either tavily_client or tavily_api_key must be provided.")
            self.tavily_client = TavilyClient(api_key=tavily_api_key)
        else:
            self.tavily_client = tavily_client

        # Set LLM provider and availability
        self.llm_provider = llm_provider.lower()
        self.use_llm = use_llm and OPENAI_AVAILABLE
        
        if not self.use_llm:
            self.llm_client = None
            self.model = None
            return
            
        # Set provider-specific base URL
        if self.llm_provider == "siliconflow":
            self.base_url = base_url or "https://api.siliconflow.cn/v1"
        else:  # default to openai
            self.base_url = base_url  # None by default for OpenAI's default URL
            
        # Check for required imports
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package is not installed. Install it with 'pip install openai' "
                "or set use_llm=False."
            )
        
        # Initialize LLM client
        if llm_client is None:
            if llm_api_key is None:
                # Try to find API key from environment variables
                if self.llm_provider == "siliconflow":
                    llm_api_key = os.getenv("SILICONFLOW_API_KEY")
                else:
                    llm_api_key = os.getenv("OPENAI_API_KEY")
                    
                if not llm_api_key:
                    raise ValueError(
                        f"Either llm_client or llm_api_key must be provided when use_llm=True for provider {self.llm_provider}."
                    )
                    
            # Create client with appropriate base URL
            self.llm_client = OpenAI(
                api_key=llm_api_key,
                base_url=self.base_url
            )
        else:
            self.llm_client = llm_client
        
        # Validate and set model
        available_models = self.get_available_models()
        if available_models and model not in available_models:
            if available_models:
                # Use first available model
                self.model = available_models[0]
                print(f"Warning: Specified model '{model}' not available. Using '{self.model}' instead.")
            else:
                # No available models, but we'll try anyway with the specified model
                self.model = model
                print(f"Warning: Unable to validate model '{model}'. Attempting to use it anyway.")
        else:
            self.model = model

    def get_available_models(self) -> List[str]:
        """
        Get a list of available models from the specified LLM provider.
        
        Returns:
            A list of available model IDs
        """
        if not self.use_llm or not self.llm_client:
            return []
            
        try:
            if self.llm_provider == "siliconflow":
                return self._get_siliconflow_models()
            else:
                return self._get_openai_models()
        except Exception as e:
            print(f"Error fetching available models: {e}")
            return []
            
    def _get_openai_models(self) -> List[str]:
        """Get available models from OpenAI"""
        try:
            models = self.llm_client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            print(f"Error fetching OpenAI models: {e}")
            return []
            
    def _get_siliconflow_models(self) -> List[str]:
        """Get available models from Silicon Flow"""
        try:
            # Direct API request to Silicon Flow
            headers = {"Authorization": f"Bearer {self.llm_client.api_key}"}
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    return [model["id"] for model in data["data"]]
            
            # Fallback models if API call fails
            return [
                "Qwen/Qwen2.5-72B-Instruct", 
                "Qwen/Qwen1.5-110B-Chat",
                "Pro/deepseek-ai/DeepSeek-R1",
                "vllm-openai/llama3-8b-instruct"
            ]
        except Exception as e:
            print(f"Error fetching Silicon Flow models: {e}")
            # Return common Silicon Flow models as fallback
            return [
                "Qwen/Qwen2.5-72B-Instruct", 
                "Qwen/Qwen1.5-110B-Chat",
                "Pro/deepseek-ai/DeepSeek-R1",
                "vllm-openai/llama3-8b-instruct"
            ]

    def analyze_company(
        self, 
        company_name: str, 
        language: str = "ja", 
        verbose: bool = True,
        selected_sections: List[str] = None,
        exclude_domains: List[str] = None,
        prefer_official: bool = False,
        stream_handler = None
    ) -> Dict[str, Any]:
        """
        Analyze a company and generate a comprehensive report.

        Args:
            company_name: The name of the company to analyze.
            language: The language for the analysis. Default is Japanese ('ja').
            verbose: Whether to print detailed progress information.
            selected_sections: List of section IDs to include in the analysis.
                            If None, all sections will be included.
            exclude_domains: List of domains to exclude from search results.
            prefer_official: Whether to prioritize official company sources.
            stream_handler: Optional handler for streaming progress updates to a UI.
            
        Returns:
            A dictionary containing the company analysis report.
        """
        # Define the sections to search for
        all_sections = self._get_analysis_sections(language)
        
        # Filter sections if specified
        if selected_sections:
            sections = [s for s in all_sections if s["id"] in selected_sections]
        else:
            sections = all_sections
        
        # Set default exclude_domains if not provided
        if exclude_domains is None and language == "ja":
            exclude_domains = []
        elif exclude_domains is None:
            exclude_domains = []
        
        # Log the start of analysis
        self._log(f"\n{'='*60}", stream_handler)
        self._log(f"STARTING ANALYSIS: {company_name}", stream_handler)
        self._log(f"{'='*60}", stream_handler)
        self._log(f"Language: {language}", stream_handler)
        
        if self.use_llm:
            self._log(f"LLM Provider: {self.llm_provider}", stream_handler)
            self._log(f"LLM Model: {self.model}", stream_handler)
        else:
            self._log(f"LLM integration: Disabled", stream_handler)
            
        self._log(f"Total sections to analyze: {len(sections)}", stream_handler)
        self._log(f"Excluded domains: {exclude_domains if exclude_domains else 'None'}", stream_handler)
        self._log(f"{'='*60}\n", stream_handler)
        
        # Initialize the report
        report = {
            "company_name": company_name,
            "sections": [],
            "metadata": {
                "language": language,
                "use_llm": self.use_llm,
                "llm_provider": self.llm_provider if self.use_llm else None,
                "llm_model": self.model if self.use_llm else None,
                "excluded_domains": exclude_domains,
                "prefer_official": prefer_official,
                "analysis_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        # Track timing
        start_time = time.time()
        
        # Find company official domain if prefer_official is True
        official_domain = None
        if prefer_official:
            self._log(f"Searching for official company domain...", stream_handler)
            official_domain = self._get_company_domain(company_name)
            if official_domain:
                self._log(f"Found official domain: {official_domain}", stream_handler)
                report["metadata"]["official_domain"] = official_domain
            else:
                self._log(f"No official domain found. Using general search.", stream_handler)
        
        # Gather information for each section
        for i, section in enumerate(sections, 1):
            self._log(f"\n[{i}/{len(sections)}] ANALYZING SECTION: {section['title']}", stream_handler)
            self._log(f"{'-'*60}", stream_handler)
            
            section_start_time = time.time()
            section_data = self._gather_section_data(
                company_name=company_name, 
                section=section, 
                language=language, 
                verbose=verbose,
                exclude_domains=exclude_domains,
                official_domain=official_domain,
                stream_handler=stream_handler
            )
            report["sections"].append(section_data)
            
            section_time = time.time() - section_start_time
            completion_percentage = (i / len(sections)) * 100
            self._log(f"{'-'*60}", stream_handler)
            self._log(f"Section completed in {section_time:.2f} seconds", stream_handler)
            self._log(f"Overall progress: {completion_percentage:.1f}% complete", stream_handler)
            
            # Force a UI update if stream_handler is provided
            if stream_handler and hasattr(stream_handler, 'update_ui'):
                stream_handler.update_ui()
        
        # Generate motivation summary
        if self.use_llm:
            self._log(f"\nGENERATING MOTIVATION SUMMARY", stream_handler)
            self._log(f"{'-'*60}", stream_handler)
            motivation_start_time = time.time()
            
            motivation_summary = self._generate_motivation_summary(
                company_name=company_name,
                report=report,
                language=language,
                stream_handler=stream_handler
            )
            
            report["motivation_summary"] = motivation_summary
            
            motivation_time = time.time() - motivation_start_time
            self._log(f"Motivation summary generated in {motivation_time:.2f} seconds", stream_handler)
        
        total_time = time.time() - start_time
        report["metadata"]["total_time"] = total_time
        
        self._log(f"\n{'='*60}", stream_handler)
        self._log(f"ANALYSIS COMPLETE: {company_name}", stream_handler)
        self._log(f"{'='*60}", stream_handler)
        self._log(f"Total analysis time: {total_time:.2f} seconds", stream_handler)
        self._log(f"Sections analyzed: {len(sections)}", stream_handler)
        self._log(f"{'='*60}\n", stream_handler)
        
        return report

    def _get_analysis_sections(self, language: str) -> List[Dict[str, Any]]:
        """
        Get the sections for company analysis based on the language.

        Args:
            language: The language code ('en' for English, 'ja' for Japanese).

        Returns:
            A list of dictionaries containing section information.
        """
        if language == "ja":
            return [
                {
                    "id": "executives",
                    "title": "代表取締役",
                    "query_template": "{company_name} 代表取締役 社長 経歴 メッセージ",
                    "subsections": ["氏名", "経歴", "代表メッセージ"]
                },
                {
                    "id": "philosophy",
                    "title": "企業理念",
                    "query_template": "{company_name} 企業理念 経営理念 創業精神",
                    "subsections": ["企業理念", "創業以来の理念・精神"]
                },
                {
                    "id": "establishment",
                    "title": "設立年・資本金・株式公開・事業拠点",
                    "query_template": "{company_name} 設立 資本金 株式公開 上場 事業拠点 本社",
                    "subsections": ["設立年", "資本金", "株式公開", "事業拠点"]
                },
                {
                    "id": "business",
                    "title": "詳しい事業内容",
                    "query_template": "{company_name} 事業内容 製品 サービス ビジネスモデル 顧客",
                    "subsections": ["商品・サービスの詳しい内容", "商品・サービスの対象者", "業態"]
                },
                {
                    "id": "performance",
                    "title": "業績",
                    "query_template": "{company_name} 売上高 営業利益 財務 決算",
                    "subsections": ["売上高", "営業利益（率）"]
                },
                {
                    "id": "growth",
                    "title": "成長性",
                    "query_template": "{company_name} 成長率 伸び率 新規事業 事業拡大 展望",
                    "subsections": ["売上高・営業利益の伸び率", "新規事業・事業拡大の展望"]
                },
                {
                    "id": "economic_impact",
                    "title": "景況・経済動向による影響度",
                    "query_template": "{company_name} 売上推移 景気 経済動向 円高 円安 影響",
                    "subsections": ["売上高・営業利益の過去推移", "円高時・円安時の売上・営業利益の状況"]
                },
                {
                    "id": "competitiveness",
                    "title": "競争力",
                    "query_template": "{company_name} 競争力 強み 開発力 技術力 品質 提供ネットワーク",
                    "subsections": ["商品・サービスの開発力・技術力・品質", "商品・サービス提供のネットワーク"]
                },
                {
                    "id": "culture",
                    "title": "社風",
                    "query_template": "{company_name} 社風 企業文化 組織 雰囲気 意思決定 人員構成",
                    "subsections": ["年齢・男女別の人員構成", "意思決定の仕組み", "新しいことへの挑戦を重視／伝統的価値を提供し続けることを重視", "チーム力を重視／個人の裁量を重視", "職場の雰囲気", "社員教育・育成環境"]
                },
                {
                    "id": "career",
                    "title": "キャリア形成の環境",
                    "query_template": "{company_name} キャリア 昇進 昇給 評価制度 勤続年数",
                    "subsections": ["昇給・昇進の仕組み", "平均勤続年数、役職者の平均年齢"]
                },
                {
                    "id": "positions",
                    "title": "職種",
                    "query_template": "{company_name} 職種 部署 スキル 採用",
                    "subsections": ["職種の種類", "職種ごとに求められるスキル"]
                },
                {
                    "id": "conditions",
                    "title": "勤務条件",
                    "query_template": "{company_name} 給与 勤務地 勤務時間 休日 手当 福利厚生 保険",
                    "subsections": ["給与", "勤務地", "勤務時間", "休日", "手当", "福利厚生", "保険"]
                },
                {
                    "id": "csr_diversity",
                    "title": "CSR活動・ダイバーシティーの取り組み",
                    "query_template": "{company_name} CSR 社会貢献 ダイバーシティ 多様性",
                    "subsections": ["CSR活動の内容", "ダイバーシティーの内容"]
                },
                {
                    "id": "related_companies",
                    "title": "関連企業",
                    "query_template": "{company_name} 親会社 子会社 グループ会社 提携",
                    "subsections": ["親会社・子会社", "グループ会社", "資本提携会社・業務提携会社"]
                }
            ]
        else:  # Default to English
            return [
                {
                    "id": "executives",
                    "title": "Company Executives",
                    "query_template": "{company_name} CEO President executives leadership team background message",
                    "subsections": ["Name", "Career/Background", "Representative Message"]
                },
                {
                    "id": "philosophy",
                    "title": "Corporate Philosophy",
                    "query_template": "{company_name} corporate philosophy mission vision values founding principles",
                    "subsections": ["Corporate Philosophy", "Founding Spirit/Values"]
                },
                {
                    "id": "establishment",
                    "title": "Establishment, Capital, IPO, Business Locations",
                    "query_template": "{company_name} founded established capital IPO stock market listing headquarters locations offices",
                    "subsections": ["Year Established", "Capital", "IPO Status", "Business Locations"]
                },
                {
                    "id": "business",
                    "title": "Detailed Business Content",
                    "query_template": "{company_name} business products services offerings target customers business model",
                    "subsections": ["Products/Services Details", "Target Customers", "Business Model"]
                },
                {
                    "id": "performance",
                    "title": "Performance",
                    "query_template": "{company_name} revenue sales profit financial performance earnings",
                    "subsections": ["Revenue", "Operating Profit"]
                },
                {
                    "id": "growth",
                    "title": "Growth Potential",
                    "query_template": "{company_name} growth rate expansion new business future prospects",
                    "subsections": ["Revenue/Profit Growth Rate", "New Business/Expansion Prospects"]
                },
                {
                    "id": "economic_impact",
                    "title": "Economic Impact Sensitivity",
                    "query_template": "{company_name} economic impact currency exchange rate sensitivity historical performance",
                    "subsections": ["Historical Revenue/Profit Trends", "Performance During Currency Fluctuations"]
                },
                {
                    "id": "competitiveness",
                    "title": "Competitive Advantages",
                    "query_template": "{company_name} competitive advantages strengths development capabilities technical quality network",
                    "subsections": ["Product/Service Development/Technical Capabilities", "Product/Service Delivery Network"]
                },
                {
                    "id": "culture",
                    "title": "Corporate Culture",
                    "query_template": "{company_name} corporate culture workforce demographics decision making innovation tradition team individual atmosphere education development",
                    "subsections": ["Employee Demographics", "Decision-Making Structure", "Innovation vs. Tradition Focus", "Team vs. Individual Focus", "Workplace Atmosphere", "Employee Education/Development"]
                },
                {
                    "id": "career",
                    "title": "Career Development Environment",
                    "query_template": "{company_name} career development salary promotion system tenure average age managers",
                    "subsections": ["Salary/Promotion System", "Average Years of Service, Average Age of Managers"]
                },
                {
                    "id": "positions",
                    "title": "Job Positions",
                    "query_template": "{company_name} job positions departments roles skills required",
                    "subsections": ["Types of Positions", "Skills Required for Each Position"]
                },
                {
                    "id": "conditions",
                    "title": "Working Conditions",
                    "query_template": "{company_name} salary compensation work location hours holidays time off allowances benefits insurance",
                    "subsections": ["Salary", "Work Location", "Working Hours", "Holidays/Time Off", "Allowances", "Benefits", "Insurance"]
                },
                {
                    "id": "csr_diversity",
                    "title": "CSR Activities/Diversity Initiatives",
                    "query_template": "{company_name} CSR corporate social responsibility sustainability diversity inclusion initiatives",
                    "subsections": ["CSR Activities", "Diversity Initiatives"]
                },
                {
                    "id": "related_companies",
                    "title": "Related Companies",
                    "query_template": "{company_name} parent subsidiary related group companies partnerships alliances",
                    "subsections": ["Parent/Subsidiary Companies", "Group Companies", "Capital/Business Alliance Companies"]
                }
            ]

    def _gather_section_data(
        self, 
        company_name: str, 
        section: Dict[str, Any], 
        language: str, 
        verbose: bool = True,
        exclude_domains: List[str] = None,
        official_domain: Optional[str] = None,
        stream_handler = None
    ) -> Dict[str, Any]:
        """
        Gather data for a specific section of the company analysis.

        Args:
            company_name: The name of the company to analyze.
            section: The section definition containing ID, title, query template, and subsections.
            language: The language code.
            verbose: Whether to print detailed progress information.
            exclude_domains: List of domains to exclude from search results.
            official_domain: The official company domain if available.
            stream_handler: Optional handler for streaming progress updates to a UI.

        Returns:
            A dictionary containing the section data.
        """
        query = section["query_template"].format(company_name=company_name)
        
        # Adjust query to prefer official sources if specified
        include_domains = None
        if official_domain:
            include_domains = [official_domain]
            self._log(f"   Including official domain: {official_domain}", stream_handler)
        
        self._log(f"1. Sending initial search query:", stream_handler)
        self._log(f"   Query: {query}", stream_handler)
        if exclude_domains:
            self._log(f"   Excluding domains: {exclude_domains}", stream_handler)
        
        # Use the Tavily search API to get information
        search_start = time.time()
        try:
            search_results = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
                exclude_domains=exclude_domains,
                include_domains=include_domains
            )
            search_time = time.time() - search_start
            
            if verbose:
                self._log(f"   Query completed in {search_time:.2f} seconds", stream_handler)
                self._log(f"   Retrieved {len(search_results.get('results', []))} results", stream_handler)
                urls = [result.get("url", "") for result in search_results.get("results", [])]
                domains = [self._extract_domain(url) for url in urls]
                self._log(f"   Sources: {', '.join(domains[:3])}{'...' if len(domains) > 3 else ''}", stream_handler)
        except Exception as e:
            if verbose:
                self._log(f"   ❌ Error during search: {str(e)}", stream_handler)
            search_results = {"results": [], "answer": ""}
        
        # Extract the answer and content
        answer = search_results.get("answer", "")
        content = "\n\n".join([result.get("content", "") for result in search_results.get("results", [])])
        
        if verbose:
            answer_preview = answer[:100] + "..." if len(answer) > 100 else answer
            self._log(f"   Answer: {answer_preview}", stream_handler)
        
        # If using LLM, evaluate the quality of the information
        information_quality = "sufficient"
        if self.use_llm:
            if verbose:
                self._log(f"\n2. Evaluating information quality with LLM...", stream_handler)
            
            eval_start = time.time()
            information_quality = self._evaluate_information_quality(
                company_name, section, answer, content, language, verbose, stream_handler
            )
            eval_time = time.time() - eval_start
            
            if verbose:
                self._log(f"   Evaluation completed in {eval_time:.2f} seconds", stream_handler)
                self._log(f"   Assessment: {information_quality}", stream_handler)
        
        # If information is insufficient, perform additional searches
        if information_quality == "insufficient" and self.use_llm:
            if verbose:
                self._log(f"\n3. Generating additional search queries...", stream_handler)
            
            gen_start = time.time()
            additional_queries = self._generate_additional_queries(
                company_name, section, answer, content, language, verbose, stream_handler
            )
            gen_time = time.time() - gen_start
            
            if verbose:
                self._log(f"   Generated {len(additional_queries)} additional queries in {gen_time:.2f} seconds", stream_handler)
                for idx, q in enumerate(additional_queries, 1):
                    self._log(f"   Query {idx}: {q}", stream_handler)
            
            additional_content = []
            for idx, query in enumerate(additional_queries, 1):
                try:
                    if verbose:
                        self._log(f"\n   Executing additional query {idx}/{len(additional_queries)}:", stream_handler)
                        self._log(f"   Query: {query}", stream_handler)
                    
                    add_search_start = time.time()
                    additional_results = self.tavily_client.search(
                        query=query,
                        search_depth="advanced",
                        max_results=3,
                        include_answer=True,
                        exclude_domains=exclude_domains,
                        include_domains=include_domains
                    )
                    add_search_time = time.time() - add_search_start
                    
                    if verbose:
                        self._log(f"   Query completed in {add_search_time:.2f} seconds", stream_handler)
                        self._log(f"   Retrieved {len(additional_results.get('results', []))} results", stream_handler)
                    
                    additional_answer = additional_results.get("answer", "")
                    additional_texts = [
                        result.get("content", "") 
                        for result in additional_results.get("results", [])
                    ]
                    additional_content.append(additional_answer)
                    additional_content.extend(additional_texts)
                except Exception as e:
                    if verbose:
                        self._log(f"   ❌ Error in additional search: {str(e)}", stream_handler)
            
            # Append additional content
            if additional_content:
                content = content + "\n\n" + "\n\n".join(additional_content)
                if verbose:
                    self._log(f"\n   Added {len(additional_content)} pieces of supplementary content", stream_handler)
        
        # Process and structure the data for each subsection
        if verbose:
            self._log(f"\n4. Processing {len(section['subsections'])} subsections...", stream_handler)
        
        subsections_data = []
        for idx, subsection in enumerate(section['subsections'], 1):
            if verbose:
                self._log(f"   Processing subsection {idx}/{len(section['subsections'])}: {subsection}", stream_handler)
            
            sub_start = time.time()
            subsection_content = self._extract_subsection_content(
                answer, content, subsection, company_name, section, language, verbose, stream_handler
            )
            sub_time = time.time() - sub_start
            
            content_preview = subsection_content[:80].replace("\n", " ") + "..." if len(subsection_content) > 80 else subsection_content
            if verbose:
                self._log(f"   Processed in {sub_time:.2f} seconds", stream_handler)
                self._log(f"   Content preview: {content_preview}", stream_handler)
            
            subsection_data = {
                "title": subsection,
                "content": subsection_content
            }
            subsections_data.append(subsection_data)
        
        # Save the sources used
        sources = []
        for result in search_results.get("results", []):
            if "url" in result and "title" in result:
                sources.append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "domain": self._extract_domain(result.get("url", ""))
                })
        
        return {
            "id": section["id"],
            "title": section["title"],
            "answer": answer,
            "subsections": subsections_data,
            "sources": sources
        }

    def _evaluate_information_quality(
        self, company_name: str, section: Dict[str, Any], 
        answer: str, content: str, language: str, verbose: bool = False,
        stream_handler = None
    ) -> str:
        """
        Evaluate the quality of the information gathered.
        
        Args:
            company_name: The name of the company.
            section: The section definition.
            answer: The answer from Tavily search.
            content: The combined content from search results.
            language: The language code.
            verbose: Whether to print detailed progress information.
            stream_handler: Optional handler for streaming progress updates to a UI.
            
        Returns:
            A string indicating if the information is "sufficient" or "insufficient".
        """
        if not self.use_llm:
            return "sufficient"
        
        # Prepare the prompt based on language
        if language == "ja":
            prompt = f"""
            あなたは企業分析の専門家です。私は{company_name}に関する「{section['title']}」セクションの情報を収集しています。
            収集した情報が十分かどうか評価してください。

            セクションの内容: {section['title']}
            求められる情報: {', '.join(section['subsections'])}
            
            収集した情報:
            [回答]
            {answer}
            
            [コンテンツ]
            {content[:2000]}...
            
            この情報は「{section['title']}」セクションに関して十分ですか？「十分」または「不十分」で回答してください。
            不十分と判断する場合、何が不足しているかも簡潔に説明してください。
            """
        else:
            prompt = f"""
            You are a company analysis expert. I'm gathering information about {company_name} for the section "{section['title']}".
            Please evaluate whether the information collected is sufficient.

            Section content: {section['title']}
            Required information: {', '.join(section['subsections'])}
            
            Collected information:
            [Answer]
            {answer}
            
            [Content]
            {content[:2000]}...
            
            Is this information sufficient for the "{section['title']}" section? Please reply with "sufficient" or "insufficient".
            If insufficient, briefly explain what is missing.
            """
        
        if verbose:
            self._log(f"   Sending evaluation request to LLM ({self.model})...", stream_handler)
        
        # Call LLM API
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a company analysis expert evaluating the quality of information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            # Process the response
            result = response.choices[0].message.content.lower()
            if verbose:
                self._log(f"   LLM response: {result}", stream_handler)
            
            if "insufficient" in result or "不十分" in result:
                return "insufficient"
            return "sufficient"
        except Exception as e:
            if verbose:
                self._log(f"   ❌ Error during evaluation: {str(e)}", stream_handler)
            return "sufficient"  # Default to sufficient in case of error

    def _generate_additional_queries(
        self, company_name: str, section: Dict[str, Any], 
        answer: str, content: str, language: str, verbose: bool = False,
        stream_handler = None
    ) -> List[str]:
        """
        Generate additional search queries to fill information gaps.
        
        Args:
            company_name: The name of the company.
            section: The section definition.
            answer: The answer from Tavily search.
            content: The combined content from search results.
            language: The language code.
            verbose: Whether to print detailed progress information.
            stream_handler: Optional handler for streaming progress updates to a UI.
            
        Returns:
            A list of additional search queries.
        """
        if not self.use_llm:
            return []
        
        # Prepare the prompt based on language
        if language == "ja":
            prompt = f"""
            あなたは企業分析の専門家です。私は{company_name}に関する「{section['title']}」セクションの情報を収集しています。
            現在の情報が不十分なため、追加の検索クエリを生成してください。

            セクションの内容: {section['title']}
            求められる情報: {', '.join(section['subsections'])}
            
            現在の情報:
            [回答]
            {answer}
            
            [コンテンツ]
            {content[:1500]}...
            
            不足している情報を収集するために、3つの追加検索クエリを生成してください。
            各クエリは具体的で、不足している情報を特定するものにしてください。
            クエリの形式: "{company_name} [具体的な検索語句]"
            """
        else:
            prompt = f"""
            You are a company analysis expert. I'm gathering information about {company_name} for the section "{section['title']}".
            The current information is insufficient, so I need additional search queries.

            Section content: {section['title']}
            Required information: {', '.join(section['subsections'])}
            
            Current information:
            [Answer]
            {answer}
            
            [Content]
            {content[:1500]}...
            
            Please generate 3 additional search queries to collect the missing information.
            Each query should be specific and targeted to identify the missing information.
            Query format: "{company_name} [specific search terms]"
            """
        
        if verbose:
            self._log(f"   Sending query generation request to LLM ({self.model})...", stream_handler)
        
        # Call LLM API
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a company analysis expert generating search queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            # Process the response
            result = response.choices[0].message.content
            
            if verbose:
                self._log(f"   LLM generated queries response received", stream_handler)
            
            # Extract queries using simple parsing
            queries = []
            for line in result.split('\n'):
                line = line.strip()
                if line.startswith('"') and line.endswith('"'):
                    # Remove quotes
                    query = line.strip('"')
                    queries.append(query)
                elif line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
                    # Extract query from numbered list
                    parts = line.split(':', 1)
                    if len(parts) > 1:
                        query = parts[1].strip().strip('"')
                        queries.append(query)
                    else:
                        query = parts[0].split('.', 1)[1].strip().strip('"')
                        queries.append(query)
                elif company_name in line and len(line) > len(company_name) + 10:
                    # Looks like a query
                    queries.append(line)
            
            # Filter and clean queries
            filtered_queries = []
            for query in queries:
                if company_name in query and len(query) > len(company_name) + 5:
                    filtered_queries.append(query)
            
            # Ensure we have at least one query
            if not filtered_queries and company_name:
                section_keywords = " ".join(section['subsections'][:2])
                return [f"{company_name} {section_keywords}"]
            
            return filtered_queries[:3]  # Limit to 3 queries
        
        except Exception as e:
            if verbose:
                self._log(f"   ❌ Error generating queries: {str(e)}", stream_handler)
            
            # Fallback: create a simple query based on section title
            section_keywords = " ".join(section['subsections'][:2])
            return [f"{company_name} {section_keywords}"]

    def _extract_subsection_content(
        self, answer: str, content: str, subsection: str, 
        company_name: str, section: Dict[str, Any], language: str, verbose: bool = False,
        stream_handler = None
    ) -> str:
        """
        Extract relevant content for a specific subsection.
        
        Args:
            answer: The generated answer from Tavily search.
            content: The combined content from search results.
            subsection: The title of the subsection.
            company_name: The name of the company.
            section: The section definition.
            language: The language code.
            verbose: Whether to print detailed progress information.
            stream_handler: Optional handler for streaming progress updates to a UI.
            
        Returns:
            The extracted content for the subsection.
        """
        # Combine the answer and content for better context
        combined_content = f"{answer}\n\n{content}"
        
        # If not using LLM or content is too short, try a direct search or return the combined content
        if not self.use_llm or len(combined_content.strip()) < 50:
            # Fall back to a new search specifically for this subsection
            try:
                if verbose:
                    self._log(f"   LLM not available/enabled, using direct Tavily search for subsection", stream_handler)
                
                subsection_query = f"{company_name} {subsection}"
                
                if verbose:
                    self._log(f"   Subsection query: {subsection_query}", stream_handler)
                
                subsection_search_start = time.time()
                subsection_answer = self.tavily_client.qna_search(subsection_query)
                subsection_search_time = time.time() - subsection_search_start
                
                if verbose:
                    self._log(f"   Subsection search completed in {subsection_search_time:.2f} seconds", stream_handler)
                
                if subsection_answer and len(subsection_answer) > 50:
                    return subsection_answer
            except Exception as e:
                if verbose:
                    self._log(f"   ❌ Error in subsection search: {str(e)}", stream_handler)
            
            return combined_content
        
        # Prepare the prompt based on language
        if language == "ja":
            prompt = f"""
            あなたは企業分析の専門家です。私は{company_name}に関する「{section['title']}」セクションの「{subsection}」サブセクションの情報を抽出しています。
            
            次の情報から、「{subsection}」に関連する情報を抽出し、整理して要約してください。
            
            情報源:
            {combined_content[:4000]}
            
            出力形式:
            - 冒頭の挨拶や自己紹介は不要です。
            - 直接「{subsection}」に関連する情報から始めてください。
            - 明確で具体的な情報を提供してください。
            - 不明な場合は「この情報は入手できませんでした」と述べてください。
            - 箇条書きではなく、段落形式で回答してください。
            - 情報源からの事実のみを含めてください。推測は避けてください。
            """
        else:
            prompt = f"""
            You are a company analysis expert. I'm extracting information for the "{subsection}" subsection of the "{section['title']}" section about {company_name}.
            
            Please extract, organize, and summarize information related to "{subsection}" from the following sources:
            
            Information sources:
            {combined_content[:4000]}
            
            Output format:
            - Skip any introductory greetings or self-introductions.
            - Start directly with information related to "{subsection}".
            - Provide clear and specific information.
            - If information is not available, state "This information could not be obtained."
            - Answer in paragraph format, not bullet points.
            - Include only facts from the sources, avoid speculation.
            """
        
        if verbose:
            self._log(f"   Sending extraction request to LLM ({self.model})...", stream_handler)
        
        # Call LLM API
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a company analysis expert extracting specific information."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Process the response
            result = response.choices[0].message.content
            
            if verbose:
                self._log(f"   LLM extraction response received ({len(result)} chars)", stream_handler)
            
            return result
        
        except Exception as e:
            if verbose:
                self._log(f"   ❌ Error extracting content: {str(e)}", stream_handler)
            
            # Fall back to direct content if LLM fails
            try:
                subsection_query = f"{company_name} {subsection}"
                subsection_answer = self.tavily_client.qna_search(subsection_query)
                if subsection_answer and len(subsection_answer) > 50:
                    return subsection_answer
            except:
                pass
            
            return combined_content

    def _generate_motivation_summary(
        self, company_name: str, report: Dict[str, Any], language: str, stream_handler = None
    ) -> str:
        """
        Generate a summary to help with job application motivation (希望動機).
        
        Args:
            company_name: The name of the company.
            report: The complete company analysis report.
            language: The language code.
            stream_handler: Optional handler for streaming progress updates to a UI.
            
        Returns:
            A motivation summary text.
        """
        if not self.use_llm:
            return ""
        
        self._log(f"   Extracting key information for motivation summary...", stream_handler)
        
        # Extract key information from different sections
        key_data = {}
        important_sections = ["philosophy", "business", "competitiveness", "culture", "growth"]
        
        for section in report["sections"]:
            if section["id"] in important_sections:
                section_data = ""
                for subsection in section["subsections"]:
                    section_data += f"{subsection['title']}: {subsection['content']}\n\n"
                key_data[section["id"]] = section_data
        
        # Prepare the prompt based on language
        if language == "ja":
            prompt = f"""
            あなたは就職活動のエキスパートです。以下の{company_name}に関する情報を元に、この企業への「志望動機」を作成するための重要ポイントをまとめてください。

            ## 企業情報
            """
            
            for section_id, content in key_data.items():
                section_title = next((s["title"] for s in report["sections"] if s["id"] == section_id), section_id)
                prompt += f"\n### {section_title}\n{content}\n"
            
            prompt += f"""
            ## 出力指示
            上記の企業情報を踏まえて、{company_name}への志望動機を書く際に活用できる重要なポイントを以下の要素に分けてまとめてください：

            1. 企業の魅力：この企業の他社にない強みや特徴
            2. 企業理念や価値観：あなたの価値観と合致する点
            3. 事業・サービス：興味を持てる事業や製品・サービス
            4. 成長可能性：企業の将来性や自分の成長機会
            5. 企業文化：働く環境として魅力的な点

            それぞれ具体的な事実を基にまとめ、志望動機の例文ではなく、志望動機を書く際の「材料」として使えるように簡潔にポイントを挙げてください。
            """
        else:
            prompt = f"""
            You are a career advice expert. Based on the following information about {company_name}, please summarize key points that would help in creating a compelling "motivation statement" or "statement of purpose" for a job application.

            ## Company Information
            """
            
            for section_id, content in key_data.items():
                section_title = next((s["title"] for s in report["sections"] if s["id"] == section_id), section_id)
                prompt += f"\n### {section_title}\n{content}\n"
            
            prompt += f"""
            ## Output Instructions
            Based on the company information above, please summarize important points that could be used in writing a motivation statement for {company_name}, divided into the following elements:

            1. Company Appeal: Strengths and unique characteristics that set this company apart
            2. Corporate Philosophy and Values: Points that align with your personal values
            3. Business/Services: Business areas or products/services you can be passionate about
            4. Growth Potential: The company's future prospects and opportunities for your personal growth
            5. Corporate Culture: Attractive aspects of the working environment

            For each element, provide concise points based on specific facts, not as a sample motivation statement but as "material" that could be used to craft a personal statement.
            """
        
        self._log(f"   Generating motivation summary with LLM...", stream_handler)
        
        # Call LLM API
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a career advisor specializing in job applications."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000
            )
            
            # Process the response
            result = response.choices[0].message.content
            self._log(f"   Generated motivation summary ({len(result)} chars)", stream_handler)
            
            return result
        
        except Exception as e:
            self._log(f"   ❌ Error generating motivation summary: {str(e)}", stream_handler)
            
            # Create a simple fallback message
            if language == "ja":
                return "志望動機を作成するための情報を生成中にエラーが発生しました。分析レポートの内容を元に、企業の強み、理念、事業内容、成長性、企業文化について検討してください。"
            else:
                return "An error occurred while generating motivation information. Please review the analysis report to identify the company's strengths, philosophy, business areas, growth potential, and corporate culture to help craft your motivation statement."

    def generate_report(self, analysis_data: Dict[str, Any], format: str = "text", verbose: bool = True, stream_handler = None) -> str:
        """
        Generate a formatted report from the analysis data.
        
        Args:
            analysis_data: The company analysis data.
            format: The format of the report ("text", "markdown", "html", "json").
            verbose: Whether to print detailed progress information.
            stream_handler: Optional handler for streaming progress updates to a UI.
            
        Returns:
            The formatted report as a string.
        """
        if verbose:
            self._log(f"\n{'='*60}", stream_handler)
            self._log(f"GENERATING REPORT", stream_handler)
            self._log(f"{'='*60}", stream_handler)
            self._log(f"Company: {analysis_data['company_name']}", stream_handler)
            self._log(f"Format: {format}", stream_handler)
            self._log(f"Using LLM: {self.use_llm}", stream_handler)
            
        if format == "json":
            if verbose:
                self._log(f"Creating JSON report...", stream_handler)
            return json.dumps(analysis_data, indent=2, ensure_ascii=False)
        
        # Extract key data
        company_name = analysis_data["company_name"]
        sections = analysis_data["sections"]
        
        # If using LLM, generate a more refined report
        if self.use_llm and format != "json":
            if verbose:
                self._log(f"Generating enhanced report with LLM...", stream_handler)
            report_start = time.time()
            report = self._generate_report_with_llm(analysis_data, format, verbose, stream_handler)
            report_time = time.time() - report_start
            
            if verbose:
                self._log(f"Report generation completed in {report_time:.2f} seconds", stream_handler)
                self._log(f"Report size: {len(report)} characters", stream_handler)
            
            return report
        
        # Fall back to basic formatting
        if verbose:
            self._log(f"Generating basic report (without LLM enhancement)...", stream_handler)
        
        report_start = time.time()
        
        if format == "text":
            report = f"Company Analysis Report: {company_name}\n"
            report += "=" * (len(report) - 1) + "\n\n"
            
            for section in sections:
                report += f"{section['title']}\n"
                report += "-" * len(section['title']) + "\n\n"
                
                for subsection in section['subsections']:
                    report += f"{subsection['title']}:\n"
                    report += f"{subsection['content']}\n\n"
                
                report += "\n"
            
            # Add motivation summary if available
            if "motivation_summary" in analysis_data and analysis_data["motivation_summary"]:
                report += "希望動機のポイント (Job Application Motivation Points)\n"
                report += "-" * 50 + "\n\n"
                report += analysis_data["motivation_summary"] + "\n\n"
        
        elif format == "markdown":
            report = f"# Company Analysis Report: {company_name}\n\n"
            
            for section in sections:
                report += f"## {section['title']}\n\n"
                
                for subsection in section['subsections']:
                    report += f"### {subsection['title']}\n\n"
                    report += f"{subsection['content']}\n\n"
            
            # Add motivation summary if available
            if "motivation_summary" in analysis_data and analysis_data["motivation_summary"]:
                report += "## 希望動機のポイント (Job Application Motivation Points)\n\n"
                report += analysis_data["motivation_summary"] + "\n\n"
        
        elif format == "html":
            report = f"<h1>Company Analysis Report: {company_name}</h1>\n\n"
            
            for section in sections:
                report += f"<h2>{section['title']}</h2>\n\n"
                
                for subsection in section['subsections']:
                    report += f"<h3>{subsection['title']}</h3>\n\n"
                    report += f"<p>{subsection['content']}</p>\n\n"
            
            # Add motivation summary if available
            if "motivation_summary" in analysis_data and analysis_data["motivation_summary"]:
                report += "<h2>希望動機のポイント (Job Application Motivation Points)</h2>\n\n"
                # Fix the f-string with newline problem by using a temp variable
                content = analysis_data["motivation_summary"].replace('\n', '<br>')
                report += f"<div>{content}</div>\n\n"
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        report_time = time.time() - report_start
        
        if verbose:
            self._log(f"Basic report generation completed in {report_time:.2f} seconds", stream_handler)
            self._log(f"Report size: {len(report)} characters", stream_handler)
        
        return report

    def _generate_report_with_llm(
        self, analysis_data: Dict[str, Any], format: str, verbose: bool = False, stream_handler = None
    ) -> str:
        """
        Generate a refined report using LLM.
        
        Args:
            analysis_data: The complete company analysis data.
            format: The format of the report.
            verbose: Whether to print detailed progress information.
            stream_handler: Optional handler for streaming progress updates to a UI.
            
        Returns:
            The formatted report as a string.
        """
        # Extract key data
        company_name = analysis_data["company_name"]
        sections = analysis_data["sections"]
        motivation_summary = analysis_data.get("motivation_summary", "")
        
        # Prepare sections data
        if verbose:
            self._log(f"   Preparing section data for LLM report generation...", stream_handler)
        
        sections_data = []
        for section in sections:
            section_content = f"## {section['title']}\n\n"
            for subsection in section['subsections']:
                section_content += f"### {subsection['title']}\n{subsection['content']}\n\n"
            sections_data.append(section_content)
        
        sections_text = "\n".join(sections_data)
        
        # Prepare the motivation summary section
        motivation_text = ""
        if motivation_summary:
            if verbose:
                self._log(f"   Including motivation summary in report...", stream_handler)
            
            motivation_text = f"""
            ## 希望動機のポイント (Job Application Motivation Points)
            
            {motivation_summary}
            """
        
        # Prepare the prompt based on the format
        if verbose:
            self._log(f"   Creating prompt for {format} format...", stream_handler)
        
        if format == "markdown":
            prompt = f"""
            You are a professional business analyst. Please format the following company analysis report in Markdown.
            This is for a formal business report about {company_name}.
            
            Please maintain the section structure, but improve the formatting, flow, and readability of the content.
            Add a brief executive summary at the beginning that highlights the key points from across all sections.
            
            # Company Analysis: {company_name}
            
            {sections_text}
            
            {motivation_text}
            
            Please return the complete formatted report in Markdown.
            """
        elif format == "text":
            prompt = f"""
            You are a professional business analyst. Please format the following company analysis report in plain text.
            This is for a formal business report about {company_name}.
            
            Please maintain the section structure, but improve the formatting, flow, and readability of the content.
            Add a brief executive summary at the beginning that highlights the key points from across all sections.
            
            COMPANY ANALYSIS: {company_name}
            
            {sections_text}
            
            {motivation_text}
            
            Please return the complete formatted report in plain text, using appropriate line breaks and separators.
            """
        elif format == "html":
            prompt = f"""
            You are a professional business analyst. Please format the following company analysis report in HTML.
            This is for a formal business report about {company_name}.
            
            Please maintain the section structure, but improve the formatting, flow, and readability of the content.
            Add a brief executive summary at the beginning that highlights the key points from across all sections.
            
            <h1>Company Analysis: {company_name}</h1>
            
            {sections_text}
            
            {motivation_text}
            
            Please return the complete formatted report in HTML.
            Use appropriate HTML tags for headings, paragraphs, and structure.
            Apply basic HTML styling to make the report professional and readable.
            """
        
        if verbose:
            self._log(f"   Sending report generation request to LLM ({self.model})...", stream_handler)
        
        # Call LLM API
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a professional business analyst creating formal company reports."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=4000
            )
            
            # Process the response
            result = response.choices[0].message.content
            
            if verbose:
                self._log(f"   LLM report generation response received ({len(result)} chars)", stream_handler)
            
            return result
        
        except Exception as e:
            if verbose:
                self._log(f"   ❌ Error generating report with LLM: {str(e)}", stream_handler)
                self._log(f"   Falling back to basic report format...", stream_handler)
            
            # Fall back to basic formatting in case of error
            if format == "markdown":
                report = f"# Company Analysis Report: {company_name}\n\n"
                
                for section in sections:
                    report += f"## {section['title']}\n\n"
                    
                    for subsection in section['subsections']:
                        report += f"### {subsection['title']}\n\n"
                        report += f"{subsection['content']}\n\n"
                
                # Add motivation summary if available
                if motivation_summary:
                    report += "## 希望動機のポイント (Job Application Motivation Points)\n\n"
                    report += motivation_summary + "\n\n"
                
                return report
            
            elif format == "text":
                report = f"Company Analysis Report: {company_name}\n"
                report += "=" * (len(report) - 1) + "\n\n"
                
                for section in sections:
                    report += f"{section['title']}\n"
                    report += "-" * len(section['title']) + "\n\n"
                    
                    for subsection in section['subsections']:
                        report += f"{subsection['title']}:\n"
                        report += f"{subsection['content']}\n\n"
                    
                    report += "\n"
                
                # Add motivation summary if available
                if motivation_summary:
                    report += "希望動機のポイント (Job Application Motivation Points)\n"
                    report += "-" * 50 + "\n\n"
                    report += motivation_summary + "\n\n"
                
                return report
            
            elif format == "html":
                report = f"<h1>Company Analysis Report: {company_name}</h1>\n\n"
                
                for section in sections:
                    report += f"<h2>{section['title']}</h2>\n\n"
                    
                    for subsection in section['subsections']:
                        report += f"<h3>{subsection['title']}</h3>\n\n"
                        report += f"<p>{subsection['content']}</p>\n\n"
                
                # Add motivation summary if available
                if motivation_summary:
                    report += "<h2>希望動機のポイント (Job Application Motivation Points)</h2>\n\n"
                    # Fix the f-string with newline problem by using a temp variable
                    content = motivation_summary.replace('\n', '<br>')
                    report += f"<div>{content}</div>\n\n"
                
                return report

    def save_report(self, report: str, company_name: str, format: str = "markdown", verbose: bool = True, stream_handler = None) -> str:
        """
        Save the report to a file.
        
        Args:
            report: The report content.
            company_name: The name of the company.
            format: The format of the report ("text", "markdown", "html", "json").
            verbose: Whether to print detailed progress information.
            stream_handler: Optional handler for streaming progress updates to a UI.
            
        Returns:
            The path to the saved file.
        """
        extension = {
            "text": "txt",
            "markdown": "md",
            "html": "html",
            "json": "json"
        }.get(format, "txt")
        
        # Sanitize company name for filename
        safe_company_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in company_name)
        
        filename = f"{safe_company_name}_analysis.{extension}"
        
        if verbose:
            self._log(f"\n{'='*60}", stream_handler)
            self._log(f"SAVING REPORT", stream_handler)
            self._log(f"{'='*60}", stream_handler)
            self._log(f"Company: {company_name}", stream_handler)
            self._log(f"Format: {format}", stream_handler)
            self._log(f"Filename: {filename}", stream_handler)
            self._log(f"Report size: {len(report)} characters", stream_handler)
        
        save_start = time.time()
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        save_time = time.time() - save_start
        
        if verbose:
            self._log(f"Report successfully saved in {save_time:.2f} seconds", stream_handler)
            self._log(f"{'='*60}\n", stream_handler)
        
        return filename

    def _log(self, message: str, stream_handler = None):
        """
        Log a message and optionally send it to a stream handler.
        
        Args:
            message: The message to log.
            stream_handler: Optional handler for streaming progress updates to a UI.
        """
        print(message)
        if stream_handler:
            stream_handler.write(message)

    def _get_company_domain(self, company_name: str) -> Optional[str]:
        """
        Try to determine the official domain for a company.
        
        Args:
            company_name: The name of the company.
            
        Returns:
            The domain name if found, None otherwise.
        """
        # First, try a direct search for the company website
        try:
            results = self.tavily_client.search(
                query=f"{company_name} official website",
                search_depth="basic",
                max_results=3
            )
            
            # Extract domain names from results
            domains = []
            for result in results.get("results", []):
                url = result.get("url", "")
                domain = self._extract_domain(url)
                if domain and "wikipedia" not in domain and "linkedin" not in domain:
                    domains.append(domain)
            
            # Look for domains that might contain the company name
            company_words = company_name.lower().split()
            for domain in domains:
                domain_lower = domain.lower()
                if any(word in domain_lower for word in company_words if len(word) > 3):
                    return domain
            
            # If no match found, return the first non-generic domain
            if domains:
                return domains[0]
                
        except Exception:
            pass
        
        return None

    def _extract_domain(self, url: str) -> Optional[str]:
        """
        Extract the domain from a URL.
        
        Args:
            url: The URL to extract the domain from.
            
        Returns:
            The domain name if successful, None otherwise.
        """
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            return domain
        except:
            return None