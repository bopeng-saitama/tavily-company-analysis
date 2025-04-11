import streamlit as st
import os
import time
import pandas as pd
import sys
import requests
import threading
import json
import queue

# 创建全局消息队列，而不是使用session_state中的队列
global_message_queue = queue.Queue()

# 配置页面
st.set_page_config(
    page_title="Company Analyzer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 创建一个适用于不同Streamlit版本的重新运行函数
def rerun_app():
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except:
            print("WARNING: Unable to rerun the application automatically.")

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .section-header {
        margin-top: 1rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
    .output-container {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        max-height: 600px;
        overflow-y: auto;
        font-family: monospace;
    }
    .progress-text {
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.85rem;
    }
    .info-box {
        background-color: #e1f5fe;
        border-left: 5px solid #03a9f4;
        padding: 1rem;
        border-radius: 5px;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .status-running {
        color: #ff5722;
        font-weight: bold;
    }
    .status-complete {
        color: #4caf50;
        font-weight: bold;
    }
    .provider-logo {
        max-width: 100px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 导入CompanyAnalyzer类
try:
    from tavily import CompanyAnalyzer
    TAVILY_IMPORTED = True
except ImportError:
    TAVILY_IMPORTED = False
    st.error("Unable to import CompanyAnalyzer. Make sure the tavily package is installed correctly.")

# 初始化会话状态 - 所有状态变量必须在此处初始化
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'progress_output' not in st.session_state:
    st.session_state.progress_output = []
if 'current_section' not in st.session_state:
    st.session_state.current_section = ""
if 'sections_completed' not in st.session_state:
    st.session_state.sections_completed = 0
if 'total_sections' not in st.session_state:
    st.session_state.total_sections = 0
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'report_content' not in st.session_state:
    st.session_state.report_content = ""
if 'available_models' not in st.session_state:
    st.session_state.available_models = {}
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = time.time()

# 处理全局消息队列中的消息
def process_message_queue():
    while not global_message_queue.empty():
        try:
            message = global_message_queue.get_nowait()
            message_type = message.get("type", "log")
            
            if message_type == "log":
                st.session_state.progress_output.append(message["text"] + "\n")
                # 更新进度指标
                if "ANALYZING SECTION:" in message["text"]:
                    section_name = message["text"].split("ANALYZING SECTION:")[1].strip()
                    st.session_state.current_section = section_name
                    st.session_state.sections_completed += 1
            elif message_type == "complete":
                st.session_state.analysis_running = False
                st.session_state.analysis_complete = True
                st.session_state.report_content = message.get("report", "")
                st.session_state.analysis_results = message.get("results", None)
            elif message_type == "error":
                st.session_state.analysis_running = False
                st.session_state.progress_output.append(f"\n❌ ERROR: {message['text']}\n")
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")

# 定期处理消息
process_message_queue()

# 检查可用性并获取模型
def check_openai_models(api_key):
    if not api_key:
        return []
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception as e:
        st.error(f"Error fetching OpenAI models: {str(e)}")
        return []

def check_siliconflow_models(api_key, force_refresh=False):
    key = "siliconflow"
    
    # 如果可用且不强制刷新，返回缓存的模型
    if key in st.session_state.available_models and not force_refresh:
        return st.session_state.available_models[key]
    
    if not api_key:
        return []
    
    models = []
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get("https://api.siliconflow.cn/v1/models", headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                models = [model["id"] for model in data["data"]]
                # 缓存结果
                st.session_state.available_models[key] = models
        else:
            # 备用模型
            models = [
                "Qwen/Qwen2.5-72B-Instruct", 
                "Qwen/Qwen1.5-110B-Chat",
                "Pro/deepseek-ai/DeepSeek-R1",
                "vllm-openai/llama3-8b-instruct"
            ]
    except Exception as e:
        st.error(f"Error fetching Silicon Flow models: {str(e)}")
        # 备用模型
        models = [
            "Qwen/Qwen2.5-72B-Instruct", 
            "Qwen/Qwen1.5-110B-Chat",
            "Pro/deepseek-ai/DeepSeek-R1",
            "vllm-openai/llama3-8b-instruct"
        ]
    
    return models

# 标题和简介
st.markdown("<h1 class='main-header'>Company Analysis Tool</h1>", unsafe_allow_html=True)
st.markdown("""
This tool generates comprehensive company analysis reports to help you prepare for job interviews 
or applications. Includes motivation points (希望動機) for job applications.
""")

# 侧边栏配置
with st.sidebar:
    st.header("Configuration")
    
    company_name = st.text_input("Company Name", "Toyota Motor Corporation")
    
    language_options = {"Japanese (ja)": "ja", "English (en)": "en"}
    language_display = st.selectbox("Language", options=list(language_options.keys()), index=0)
    language = language_options[language_display]
    
    format_options = {"Markdown": "markdown", "HTML": "html", "Text": "text", "JSON": "json"}
    format_display = st.selectbox("Output Format", options=list(format_options.keys()), index=0)
    output_format = format_options[format_display]
    
    st.divider()
    
    st.header("API Keys")
    
    tavily_key = st.text_input("Tavily API Key", os.environ.get("TAVILY_API_KEY", ""), type="password",
                               help="Get a Tavily API key from https://tavily.com")
    
    # LLM提供商选择
    llm_provider = st.selectbox(
        "LLM Provider", 
        ["OpenAI", "Silicon Flow"], 
        index=0,
        help="Choose language model provider"
    )
    
    use_llm = st.checkbox("Use LLM for Enhanced Results", True,
                         help="LLM integration improves report quality but requires an API key")
    
    # 根据提供商确定正确的环境变量
    default_api_key = ""
    if llm_provider == "OpenAI":
        default_api_key = os.environ.get("OPENAI_API_KEY", "")
    else:  # Silicon Flow
        default_api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    
    llm_key = st.text_input(f"{llm_provider} API Key", default_api_key, 
                           type="password", disabled=not use_llm,
                           help=f"Get a {llm_provider} API key from their website")
    
    # 根据提供商和API密钥获取可用模型
    available_models = []
    if use_llm and llm_key:
        with st.spinner(f"Checking available {llm_provider} models..."):
            if llm_provider == "OpenAI":
                available_models = check_openai_models(llm_key)
            else:  # Silicon Flow
                available_models = check_siliconflow_models(llm_key)
        
        if available_models:
            st.success(f"Found {len(available_models)} available models")
        else:
            st.warning("No models found or API key issue")
    
    # 模型选择与适当的默认值
    if llm_provider == "OpenAI":
        default_models = ["gpt-4", "gpt-4-1106-preview", "gpt-3.5-turbo"]
    else:  # Silicon Flow
        default_models = [
            "Qwen/Qwen2.5-72B-Instruct", 
            "Qwen/Qwen1.5-110B-Chat",
            "Pro/deepseek-ai/DeepSeek-R1"
        ]
    
    # 合并可用和默认模型，移除重复项
    all_models = list(set(available_models + default_models))
    
    # 选择第一个可用模型作为默认值
    default_index = 0
    if available_models and available_models[0] in all_models:
        default_index = all_models.index(available_models[0])
    elif default_models and default_models[0] in all_models:
        default_index = all_models.index(default_models[0])
    
    llm_model = st.selectbox(
        f"{llm_provider} Model", 
        all_models, 
        index=min(default_index, len(all_models)-1) if all_models else 0,
        disabled=not use_llm or not llm_key,
        help="Select the model to use for analysis"
    )
    
    if llm_provider == "Silicon Flow":
        st.markdown("""
        <div class="info-box">
        <b>Silicon Flow Notes:</b><br>
        - Silicon Flow provides access to models like Qwen and DeepSeek<br>
        - Costs are generally lower than OpenAI<br>
        - Response time can be slower for complex prompts
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # 源控制选项
    st.header("Source Control")
    
    exclude_wikipedia = st.checkbox("Exclude Wikipedia", True,
                                   help="Exclude Wikipedia from search results as they may contain inaccurate information")
    
    prefer_official = st.checkbox("Prefer Official Sources", True,
                                 help="Try to find and prioritize the company's official website")
    
    st.divider()

    # 区域选择
    st.header("Report Sections")
    st.info("Select sections to include in the analysis")
    
    # 根据语言获取区域
    section_options = {}
    selected_sections = {}
    
    # 创建占位符
    section_options_ph = st.empty()
    
    # 使用最小配置初始化临时分析器
    if TAVILY_IMPORTED:
        try:
            temp_analyzer = CompanyAnalyzer(
                tavily_api_key="temp" if not tavily_key else tavily_key,
                use_llm=False  # 重要，避免需要LLM API密钥
            )
            all_sections = temp_analyzer._get_analysis_sections(language)
            
            # 将区域分组以便更好地组织
            section_options = {section["id"]: section["title"] for section in all_sections}
            selected_sections = {}
            
            # 创建两列以进行区域选择
            with section_options_ph.container():
                col1, col2 = st.columns(2)
                
                with col1:
                    for i, (section_id, title) in enumerate(list(section_options.items())[:7]):
                        selected_sections[section_id] = st.checkbox(title, True, key=f"section_{section_id}")
                
                with col2:
                    for i, (section_id, title) in enumerate(list(section_options.items())[7:]):
                        selected_sections[section_id] = st.checkbox(title, True, key=f"section_{section_id}")
                        
            selected_section_ids = [section_id for section_id, selected in selected_sections.items() if selected]
        
        except Exception as e:
            st.error(f"Error loading sections: {str(e)}")
            selected_section_ids = []
    else:
        st.error("Cannot load sections: CompanyAnalyzer not available")
        selected_section_ids = []

    # 开始分析按钮
    st.divider()
    start_button = st.button("Start Analysis", type="primary", disabled=st.session_state.analysis_running)

    if st.session_state.analysis_running:
        st.warning("Analysis is currently running...")
    
    # 重置按钮
    if st.session_state.analysis_complete or st.session_state.analysis_running:
        reset_button = st.button("Start New Analysis", type="secondary")
        if reset_button:
            # 重置所有相关的会话状态
            st.session_state.analysis_results = None
            st.session_state.progress_output = []
            st.session_state.current_section = ""
            st.session_state.sections_completed = 0
            st.session_state.total_sections = 0
            st.session_state.analysis_running = False
            st.session_state.analysis_complete = False
            st.session_state.report_content = ""
            # 清空全局消息队列
            while not global_message_queue.empty():
                global_message_queue.get()
            rerun_app()

# 主内容区域
col_left, col_right = st.columns([3, 2])

with col_left:
    # 进度输出区域
    st.markdown("<div class='section-header'>Analysis Progress:</div>", unsafe_allow_html=True)
    
    status_html = ""
    if st.session_state.analysis_running:
        status_html = "<span class='status-running'>⚙️ Analysis Running...</span>"
    elif st.session_state.analysis_complete:
        status_html = "<span class='status-complete'>✅ Analysis Complete</span>"
    
    st.markdown(status_html, unsafe_allow_html=True)
    
    progress_container = st.empty()
    
    with progress_container:
        if st.session_state.progress_output:
            st.markdown("<div class='output-container'>", unsafe_allow_html=True)
            st.markdown(f"<pre class='progress-text'>{''.join(st.session_state.progress_output)}</pre>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Progress output will appear here once analysis starts.")

with col_right:
    # 进度指标
    st.markdown("<div class='section-header'>Progress Metrics:</div>", unsafe_allow_html=True)
    
    # 以一种美观的格式显示指标
    if st.session_state.total_sections > 0:
        progress_pct = min(st.session_state.sections_completed / st.session_state.total_sections, 1.0)
        st.progress(progress_pct)
        
        metrics_col1, metrics_col2 = st.columns(2)
        metrics_col1.metric("Sections Completed", f"{st.session_state.sections_completed}/{st.session_state.total_sections}")
        metrics_col2.metric("Completion", f"{int(progress_pct * 100)}%")
        
        if st.session_state.current_section:
            st.info(f"Currently analyzing: **{st.session_state.current_section}**", icon="ℹ️")
    else:
        st.info("Metrics will be displayed once analysis starts.")
    
    # 环境信息
    st.markdown("<div class='section-header'>Environment:</div>", unsafe_allow_html=True)
    if use_llm:
        st.success(f"Using {llm_provider} / {llm_model}")
    else:
        st.warning("LLM integration disabled")
    
    # 显示被排除的域名
    if exclude_wikipedia:
        st.info("Wikipedia domains will be excluded from results")
    
    # 显示选定的区域
    if selected_section_ids:
        st.success(f"{len(selected_section_ids)} sections selected for analysis")

# 报告输出区域
st.markdown("<div class='section-header'>Analysis Report:</div>", unsafe_allow_html=True)
report_container = st.container()

with report_container:
    if st.session_state.report_content:
        # 创建用于查看和下载的选项卡
        view_tab, download_tab = st.tabs(["View Report", "Download"])
        
        with view_tab:
            # 根据格式适当地显示
            if output_format == "markdown":
                st.markdown(st.session_state.report_content)
            elif output_format == "html":
                st.components.v1.html(st.session_state.report_content, height=600, scrolling=True)
            elif output_format == "json":
                try:
                    st.json(json.loads(st.session_state.report_content))
                except:
                    st.text(st.session_state.report_content)
            else:  # text
                st.text(st.session_state.report_content)
        
        with download_tab:
            # 提供下载按钮
            st.download_button(
                "Download Report",
                st.session_state.report_content,
                file_name=f"{company_name.replace(' ', '_')}_analysis.{output_format}",
                mime="text/plain",
                key="download_button"
            )
            
            # 说明
            st.markdown("""
            ### Download Instructions
            1. Click the button above to download the report
            2. Save the file to your preferred location
            3. The file format matches your selected output format
            """)
            
    elif not st.session_state.analysis_running:
        st.markdown("<div class='info-box'>The analysis report will appear here once processing is complete.</div>", unsafe_allow_html=True)

# 线程安全的流处理器类，通过队列传递消息
class QueueStreamHandler:
    def write(self, text):
        # 将消息放入队列而不是直接修改会话状态
        global_message_queue.put({"type": "log", "text": text})

    def update_ui(self):
        # 线程中不直接更新UI
        pass

# 在单独的线程中运行分析的函数
def run_analysis_thread(
    company_name, 
    language, 
    output_format, 
    use_llm,
    llm_provider,
    llm_model,
    llm_key,
    tavily_key,
    selected_section_ids,
    exclude_domains,
    prefer_official,
    stream_handler
):
    try:
        # 根据提供商设置基本URL
        base_url = None
        if llm_provider.lower() == "siliconflow":
            base_url = "https://api.siliconflow.cn/v1"
        
        # 初始化分析器
        analyzer = CompanyAnalyzer(
            tavily_api_key=tavily_key,
            llm_api_key=llm_key if use_llm else None,
            model=llm_model,
            use_llm=use_llm,
            llm_provider=llm_provider.lower(),
            base_url=base_url
        )
        
        # 运行分析
        analysis_data = analyzer.analyze_company(
            company_name,
            language=language,
            verbose=True,
            selected_sections=selected_section_ids,
            exclude_domains=exclude_domains,
            prefer_official=prefer_official,
            stream_handler=stream_handler
        )
        
        # 生成报告
        report = analyzer.generate_report(
            analysis_data, 
            format=output_format,
            verbose=True,
            stream_handler=stream_handler
        )
        
        # 发送完成消息
        completion_message = {
            "type": "complete",
            "report": report,
            "results": analysis_data
        }
        global_message_queue.put(completion_message)
        
    except Exception as e:
        # 处理任何错误
        error_message = {"type": "error", "text": str(e)}
        global_message_queue.put(error_message)

# 设置自动刷新
if st.session_state.analysis_running:
    rerun_app()

# 当点击开始按钮时运行分析
if start_button:
    if not tavily_key:
        st.error("Please enter your Tavily API Key")
    elif use_llm and not llm_key:
        st.error(f"Please enter your {llm_provider} API Key or disable LLM integration")
    elif not selected_section_ids:
        st.error("Please select at least one section to analyze")
    else:
        # 清除任何先前的输出
        st.session_state.progress_output = []
        st.session_state.current_section = ""
        st.session_state.sections_completed = 0
        st.session_state.total_sections = len(selected_section_ids)
        st.session_state.analysis_running = True
        st.session_state.analysis_complete = False
        st.session_state.report_content = ""
        st.session_state.last_update_time = time.time()
        
        # 清空消息队列
        while not global_message_queue.empty():
            global_message_queue.get()
        
        # 设置域名排除
        exclude_domains = ["wikipedia.org", "wikimedia.org"] if exclude_wikipedia else None
        
        # 创建基于队列的流处理器
        stream_handler = QueueStreamHandler()
        
        # 在单独的线程中开始分析
        analysis_thread = threading.Thread(
            target=run_analysis_thread,
            args=(
                company_name, 
                language, 
                output_format, 
                use_llm,
                llm_provider,
                llm_model,
                llm_key,
                tavily_key,
                selected_section_ids,
                exclude_domains,
                prefer_official,
                stream_handler
            )
        )
        analysis_thread.daemon = True
        analysis_thread.start()
        
        # 强制立即更新UI以显示分析已开始
        rerun_app()