import streamlit as st
import os
import time
import datetime
import asyncio
import json
import pandas as pd
import threading
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure page
st.set_page_config(
    page_title="Company Analyzer",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin-top: 10px;
        max-height: 600px;
        overflow-y: auto;
        font-family: monospace;
        border: 1px solid #e0e0e0;
    }
    .progress-text {
        font-family: monospace;
        white-space: pre-wrap;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .user-log {
        color: #212121;
        margin-bottom: 8px;
        padding: 5px 0;
        border-bottom: 1px solid #eeeeee;
    }
    .settings-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    .phase-container {
        margin-bottom: 10px;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ccc;
    }
    .phase-pending { border-left-color: #9e9e9e; background-color: #f5f5f5; }
    .phase-running { border-left-color: #ff9800; background-color: #fff8e1; }
    .phase-complete { border-left-color: #4caf50; background-color: #f1f8e9; }
        .log-entry {
        padding: 3px 0;
        border-bottom: 1px solid #eee;
        font-family: monospace;
    }
    .log-technical {
        font-family: monospace;
        font-size: 0.85rem;
        padding: 3px 5px;
        background-color: #f5f5f5;
        border-left: 3px solid #ccc;
        margin-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Import CompanyAnalyzer class
try:
    from tavily import CompanyAnalyzer
    TAVILY_IMPORTED = True
except ImportError:
    TAVILY_IMPORTED = False
    st.error("Unable to import CompanyAnalyzer. Make sure the tavily package is installed correctly.")

# Create directory for settings
SETTINGS_DIR = Path.home() / ".company_analyzer"
SETTINGS_FILE = SETTINGS_DIR / "settings.json"
MODELS_FILE = SETTINGS_DIR / "models_cache.json"
SETTINGS_DIR.mkdir(exist_ok=True)

# Settings management functions
def load_settings():
    """Load settings from file or create with defaults"""
    default_settings = {
        "api_keys": {
            "tavily": "",
            "openai": "",
            "siliconflow": ""
        },
        "preferences": {
            "preferred_provider": "OpenAI",
            "preferred_model": "",
            "language": "ja",
            "output_format": "markdown",
            "exclude_wikipedia": True,
            "prefer_official": True,
            "auto_refresh": True,
            "refresh_interval": 1.0
        },
        "models_cache": {
            "openai": [],
            "siliconflow": []
        },
        "models_cache_timestamp": None
    }
    
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            
            # Merge with defaults to ensure all keys exist
            merged_settings = default_settings.copy()
            deep_update(merged_settings, settings)
            return merged_settings
        except Exception as e:
            st.error(f"Error loading settings: {e}")
            return default_settings
    else:
        return default_settings

def deep_update(target_dict, update_dict):
    """Recursively update nested dictionaries"""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict):
            deep_update(target_dict[key], value)
        else:
            target_dict[key] = value

def save_settings(settings):
    """Save settings to file"""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving settings: {e}")
        return False

def update_models_cache(provider, models):
    """Update the cached models for a provider"""
    settings = load_settings()
    settings["models_cache"][provider.lower()] = models
    settings["models_cache_timestamp"] = datetime.datetime.now().isoformat()
    save_settings(settings)

# 初始化会话状态
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'settings' not in st.session_state:
    st.session_state.settings = load_settings()

# 导航函数
def go_to_main():
    st.session_state.page = 'main'

def go_to_settings():
    st.session_state.page = 'settings'

# Function to fetch models
async def fetch_models(llm_provider, api_key):
    """Fetch available models from the specified provider"""
    if not api_key:
        return []
    
    # Check cache first
    settings = st.session_state.settings
    provider_key = llm_provider.lower()
    cached_models = settings["models_cache"].get(provider_key, [])
    
    # Use cached models if available and not expired (7 days)
    timestamp = settings.get("models_cache_timestamp")
    if cached_models and timestamp:
        try:
            cache_time = datetime.datetime.fromisoformat(timestamp)
            cache_age = datetime.datetime.now() - cache_time
            if cache_age.days < 7:
                return cached_models
        except:
            pass
    
    # Fetch new models
    if llm_provider == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            models = client.models.list()
            model_list = [model.id for model in models.data]
            update_models_cache(provider_key, model_list)
            return model_list
        except Exception as e:
            return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    else:  # Silicon Flow
        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get("https://api.siliconflow.cn/v1/models", headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    models = [model["id"] for model in data["data"]]
                    update_models_cache(provider_key, models)
                    return models
            return ["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen1.5-110B-Chat"]
        except Exception as e:
            return ["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen1.5-110B-Chat"]

# 分析工作线程类 - 使用与 app_test3.py 中相同的线程模型
class AnalysisWorker(threading.Thread):
    def __init__(self, 
                 company_name,
                 language,
                 output_format,
                 use_llm,
                 llm_provider,
                 llm_model,
                 llm_key,
                 tavily_key,
                 selected_section_ids,
                 section_titles,
                 exclude_domains,
                 prefer_official,
                 **kwargs):
        """初始化分析工作线程"""
        super().__init__(**kwargs)
        # 存储所有分析参数
        self.company_name = company_name
        self.language = language
        self.output_format = output_format
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_key = llm_key
        self.tavily_key = tavily_key
        self.selected_section_ids = selected_section_ids
        self.section_titles = section_titles
        self.exclude_domains = exclude_domains
        self.prefer_official = prefer_official
        
        # 进度信息
        self.should_stop = threading.Event()
        self.logs = []
        self.user_logs = []
        self.sections = []
        self.overall_progress = 0
        self.completed = False
        self.report_content = ""
        self.report_path = None
        self.current_section = ""
        
        # 初始化部分
        for section_id in selected_section_ids:
            title = section_titles.get(section_id, section_id)
            self.sections.append({
                "id": section_id,
                "title": title,
                "status": "pending",
                "start_time": None,
                "end_time": None,
                "completion_time": None
            })
    
    def write(self, text):
        """日志记录方法 - 模拟 AnalysisLogger 的接口"""
        # 添加技术日志 - 所有内容都进入技术日志
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {text}")
        
        # 处理特殊消息以创建用户友好日志
        # 只有特定类型的消息才会添加到用户日志中
        if "ANALYZING SECTION:" in text:
            section_name = text.split("ANALYZING SECTION:")[1].strip()
            self.current_section = section_name
            self._update_section_status(section_name, "running")
            self._add_user_friendly_log(f"✨ 開始: 「{section_name}」分析中...")
            
        elif "Section completed in" in text:
            self._update_section_status(self.current_section, "complete")
            self._add_user_friendly_log(f"✅ 完了: 「{self.current_section}」分析が完了しました")
            
        elif "Sending initial search query" in text:
            self._add_user_friendly_log("🔍 情報検索中...")
        elif "Evaluating information quality" in text:
            self._add_user_friendly_log("🧐 情報の質を評価しています...")
        elif "Generating additional search queries" in text:
            self._add_user_friendly_log("🔄 追加情報の検索中...")
        elif "Processing subsection" in text:
            try:
                subsection_info = text.split("Processing subsection")[1].split(":", 1)
                subsection_name = subsection_info[1].strip()
                self._add_user_friendly_log(f"📝 サブセクション処理中: {subsection_name}")
            except:
                self._add_user_friendly_log("📝 サブセクション処理中...")
        elif "GENERATING REPORT" in text:
            self._add_user_friendly_log("📊 レポート生成中...")
        elif "Report generation completed" in text:
            self._add_user_friendly_log("📑 レポート生成が完了しました")
        elif "Report size:" in text:
            try:
                size = text.split("Report size:")[1].strip()
                self._add_user_friendly_log(f"📏 レポートサイズ: {size}")
            except:
                pass
        elif "Report saved" in text and "to:" in text:
            try:
                file_path = text.split("to:")[1].strip()
                self._add_user_friendly_log(f"💾 レポートを保存しました: {file_path}")
            except:
                pass
        # 注意：其他类型的消息不会添加到用户日志，只存在于技术日志中
    
    def _add_user_friendly_log(self, message):
        """添加用户友好日志消息"""
        self.user_logs.append(message)
    
    def _update_section_status(self, section_name, status):
        """更新部分状态"""
        for section in self.sections:
            if section["title"] == section_name:
                section["status"] = status
                if status == "running":
                    section["start_time"] = datetime.datetime.now().isoformat()
                elif status == "complete":
                    section["end_time"] = datetime.datetime.now().isoformat()
                    # 计算完成时间
                    if section.get("start_time"):
                        start = datetime.datetime.fromisoformat(section["start_time"])
                        end = datetime.datetime.now()
                        section["completion_time"] = (end - start).total_seconds()
                break
        
        # 更新整体进度
        self._update_overall_progress()
    
    def _update_overall_progress(self):
        """更新整体进度百分比"""
        total_phases = len(self.sections)
        completed_phases = sum(1 for p in self.sections if p["status"] == "complete")
        running_phases = [p for p in self.sections if p["status"] == "running"]
        
        # 已完成阶段的基础进度
        progress = completed_phases / total_phases * 100
        
        # 添加运行中阶段的部分进度
        if running_phases:
            for phase in running_phases:
                # 使用更精确的进度贡献计算
                phase_contribution = 0
                if "start_time" in phase and phase["start_time"]:
                    # 估算当前阶段完成的百分比
                    start = datetime.datetime.fromisoformat(phase["start_time"])
                    now = datetime.datetime.now()
                    elapsed = (now - start).total_seconds()
                    # 假设每个阶段平均需要 10 秒
                    estimated_progress = min(elapsed / 10.0, 0.9) * 100  
                    phase_contribution = estimated_progress / 100 / total_phases
                else:
                    # 如果没有开始时间，假设完成了 5%
                    phase_contribution = 0.05 / total_phases
                
                progress += phase_contribution
        
        # 如果已经完成，确保进度为100%
        if self.completed:
            progress = 100
            
        self.overall_progress = progress
    
    def run(self):
        """线程运行方法 - 执行实际分析过程"""
        try:
            # 记录开始日志
            self.write(f"Starting analysis for {self.company_name}")
            self.write(f"Language: {self.language}")
            self.write(f"Output format: {self.output_format}")
            self.write(f"Using LLM: {self.use_llm} ({self.llm_provider}/{self.llm_model})")
            self.write(f"Selected sections: {len(self.selected_section_ids)}")
            
            if self.should_stop.is_set():
                return
            
            # 设置基本 URL
            base_url = None
            if self.llm_provider.lower() == "siliconflow":
                base_url = "https://api.siliconflow.cn/v1"
            
            # 初始化分析器
            try:
                self.write("Initializing CompanyAnalyzer...")
                analyzer = CompanyAnalyzer(
                    tavily_api_key=self.tavily_key,
                    llm_api_key=self.llm_key if self.use_llm else None,
                    model=self.llm_model,
                    use_llm=self.use_llm,
                    llm_provider=self.llm_provider.lower(),
                    base_url=base_url
                )
                self.write("CompanyAnalyzer initialized successfully")
            except Exception as e:
                self.write(f"ERROR: Failed to initialize CompanyAnalyzer: {str(e)}")
                return
            
            if self.should_stop.is_set():
                return
            
            # 运行分析
            try:
                self.write(f"\nStarting company analysis for {self.company_name}...")
                
                # 运行分析
                analysis_data = analyzer.analyze_company(
                    self.company_name,
                    language=self.language,
                    verbose=True,
                    selected_sections=self.selected_section_ids,
                    exclude_domains=self.exclude_domains,
                    prefer_official=self.prefer_official,
                    stream_handler=self
                )
                
                if self.should_stop.is_set():
                    return
                
                # 生成报告
                self.write(f"\nGenerating {self.output_format} report...")
                
                report = analyzer.generate_report(
                    analysis_data, 
                    format=self.output_format,
                    verbose=True,
                    stream_handler=self
                )
                self.report_content = report
                
                if self.should_stop.is_set():
                    return
                
                # 保存报告到文件
                try:
                    # 处理公司名称以用于文件名
                    safe_company_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in self.company_name)
                    file_name = f"{safe_company_name}_analysis.{self.output_format}"
                    
                    self.write(f"Saving report to file: {file_name}...")
                    with open(file_name, "w", encoding="utf-8") as f:
                        f.write(report)
                    
                    file_path = os.path.abspath(file_name)
                    self.write(f"Report saved successfully to: {file_path}")
                    self.report_path = file_path
                except Exception as e:
                    self.write(f"ERROR: File saving failed: {str(e)}")
                
                # 完成
                self.write("\nAnalysis process complete!")
                self.completed = True
                self.overall_progress = 100  # 确保进度为100%
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                self.write(f"ERROR: Analysis failed: {str(e)}")
                self.write(f"Error details: {error_details}")
                
        except Exception as e:
            import traceback
            self.write(f"THREAD ERROR: {str(e)}")
            self.write(f"Error details: {traceback.format_exc()}")
        finally:
            # 不在这里修改会话状态 - 而是在主循环中处理
            pass

# 线程管理器类
class ThreadManager:
    def __init__(self):
        self.worker = None
    
    def get_worker(self):
        return self.worker
    
    def is_running(self):
        return self.worker is not None and self.worker.is_alive() and not self.worker.completed
    
    def is_completed(self):
        return self.worker is not None and self.worker.completed
    
    def start_worker(self, 
                    company_name,
                    language,
                    output_format,
                    use_llm,
                    llm_provider,
                    llm_model,
                    llm_key,
                    tavily_key,
                    selected_section_ids,
                    section_titles,
                    exclude_domains,
                    prefer_official):
        """启动新的工作线程"""
        if self.worker is not None:
            self.stop_worker()
        
        self.worker = AnalysisWorker(
            company_name=company_name,
            language=language,
            output_format=output_format,
            use_llm=use_llm,
            llm_provider=llm_provider,
            llm_model=llm_model,
            llm_key=llm_key,
            tavily_key=tavily_key,
            selected_section_ids=selected_section_ids,
            section_titles=section_titles,
            exclude_domains=exclude_domains,
            prefer_official=prefer_official,
            daemon=True
        )
        self.worker.start()
        return self.worker
    
    def stop_worker(self):
        """停止工作线程"""
        if self.worker is not None:
            self.worker.should_stop.set()
            self.worker.join(timeout=2)  # 等待最多2秒
            self.worker = None

# 使用 st.cache_resource 缓存线程管理器
@st.cache_resource
def get_thread_manager():
    return ThreadManager()

# Settings page
def show_settings_page():
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    
    settings = st.session_state.settings
    
    # Create form for settings
    with st.form("settings_form"):
        st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
        st.subheader("API Keys")
        
        tavily_key = st.text_input("Tavily API Key", 
                                  value=settings["api_keys"].get("tavily", ""), 
                                  type="password",
                                  help="Get a Tavily API key from https://tavily.com")
        
        openai_key = st.text_input("OpenAI API Key", 
                                 value=settings["api_keys"].get("openai", ""), 
                                 type="password",
                                 help="Get an OpenAI API key from https://platform.openai.com")
        
        siliconflow_key = st.text_input("SiliconFlow API Key", 
                                      value=settings["api_keys"].get("siliconflow", ""), 
                                      type="password",
                                      help="Get a SiliconFlow API key from https://siliconflow.cn")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
        st.subheader("Default Preferences")
        
        # LLM provider preference
        preferred_provider = st.selectbox(
            "Preferred LLM Provider",
            ["OpenAI", "Silicon Flow"],
            index=0 if settings["preferences"].get("preferred_provider") == "OpenAI" else 1
        )
        
        # Preferred models section
        st.subheader("Preferred Models")
        
        # Check if we have saved models for OpenAI
        openai_models = settings["models_cache"].get("openai", ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"])
        preferred_openai_model = settings["preferences"].get("preferred_openai_model", openai_models[0] if openai_models else "")
        
        # OpenAI model selection
        openai_model = st.selectbox(
            "Preferred OpenAI Model",
            openai_models,
            index=openai_models.index(preferred_openai_model) if preferred_openai_model in openai_models else 0
        ) if openai_models else st.text_input("Preferred OpenAI Model", preferred_openai_model)
        
        # Check if we have saved models for SiliconFlow
        siliconflow_models = settings["models_cache"].get("siliconflow", ["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen1.5-110B-Chat"])
        preferred_siliconflow_model = settings["preferences"].get("preferred_siliconflow_model", siliconflow_models[0] if siliconflow_models else "")
        
        # SiliconFlow model selection
        siliconflow_model = st.selectbox(
            "Preferred SiliconFlow Model",
            siliconflow_models,
            index=siliconflow_models.index(preferred_siliconflow_model) if preferred_siliconflow_model in siliconflow_models else 0
        ) if siliconflow_models else st.text_input("Preferred SiliconFlow Model", preferred_siliconflow_model)
        
        # Language preference
        language_options = {"Japanese (ja)": "ja", "English (en)": "en"}
        language_keys = list(language_options.keys())
        language_values = list(language_options.values())
        
        try:
            default_language_index = language_values.index(settings["preferences"].get("language", "ja"))
        except ValueError:
            default_language_index = 0
            
        language_display = st.selectbox(
            "Default Language",
            language_keys,
            index=default_language_index
        )
        language = language_options[language_display]
        
        # Output format preference
        format_options = {"Markdown": "markdown", "HTML": "html", "Text": "text", "JSON": "json"}
        format_keys = list(format_options.keys())
        format_values = list(format_options.values())
        
        try:
            default_format_index = format_values.index(settings["preferences"].get("output_format", "markdown"))
        except ValueError:
            default_format_index = 0
            
        format_display = st.selectbox(
            "Default Output Format",
            format_keys,
            index=default_format_index
        )
        output_format = format_options[format_display]
        
        # Other preferences
        exclude_wikipedia = st.checkbox(
            "Exclude Wikipedia by default", 
            value=settings["preferences"].get("exclude_wikipedia", True)
        )
        
        prefer_official = st.checkbox(
            "Prefer Official Sources by default", 
            value=settings["preferences"].get("prefer_official", True)
        )
        
        # UI refresh settings
        auto_refresh = st.checkbox(
            "Enable automatic UI updates",
            value=settings["preferences"].get("auto_refresh", True)
        )
        
        refresh_interval = st.slider(
            "UI refresh interval (seconds)",
            min_value=0.1,
            max_value=3.0,
            value=settings["preferences"].get("refresh_interval", 1.0),
            step=0.1
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Save button
        save_button = st.form_submit_button("Save Settings", type="primary", use_container_width=True)
    
    # Handle form submission
    if save_button:
        # Update settings
        new_settings = settings.copy()
        
        # Update API keys
        new_settings["api_keys"]["tavily"] = tavily_key
        new_settings["api_keys"]["openai"] = openai_key
        new_settings["api_keys"]["siliconflow"] = siliconflow_key
        
        # Update preferences
        new_settings["preferences"]["preferred_provider"] = preferred_provider
        new_settings["preferences"]["preferred_openai_model"] = openai_model
        new_settings["preferences"]["preferred_siliconflow_model"] = siliconflow_model
        new_settings["preferences"]["language"] = language
        new_settings["preferences"]["output_format"] = output_format
        new_settings["preferences"]["exclude_wikipedia"] = exclude_wikipedia
        new_settings["preferences"]["prefer_official"] = prefer_official
        new_settings["preferences"]["auto_refresh"] = auto_refresh
        new_settings["preferences"]["refresh_interval"] = refresh_interval
        
        # Save to session state and file
        st.session_state.settings = new_settings
        if save_settings(new_settings):
            st.success("Settings saved successfully!")
        else:
            st.error("Failed to save settings.")
    
    # Return to main page button
    if st.button("Return to Main Page", use_container_width=True):
        go_to_main()
        st.rerun()

# Main app page
def show_main_page():
    # Title and introduction
    st.markdown("<h1 class='main-header'>Company Analysis Tool</h1>", unsafe_allow_html=True)
    st.markdown("""
    This tool generates comprehensive company analysis reports to help you prepare for job interviews 
    or applications. Includes motivation points (希望動機) for job applications.
    """)
    
    # 获取线程管理器
    thread_manager = get_thread_manager()
    worker = thread_manager.get_worker()
    
    # 检查是否正在运行或已完成
    is_running = thread_manager.is_running()
    is_completed = thread_manager.is_completed()
    
    settings = st.session_state.settings
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        company_name = st.text_input("Company Name", "Toyota Motor Corporation")
        
        language_options = {"Japanese (ja)": "ja", "English (en)": "en"}
        language_keys = list(language_options.keys())
        language_values = list(language_options.values())
        
        try:
            default_language_index = language_values.index(settings["preferences"].get("language", "ja"))
        except ValueError:
            default_language_index = 0
            
        language_display = st.selectbox("Language", options=language_keys, index=default_language_index)
        language = language_options[language_display]
        
        format_options = {"Markdown": "markdown", "HTML": "html", "Text": "text", "JSON": "json"}
        format_keys = list(format_options.keys())
        format_values = list(format_options.values())
        
        try:
            default_format_index = format_values.index(settings["preferences"].get("output_format", "markdown"))
        except ValueError:
            default_format_index = 0
            
        format_display = st.selectbox("Output Format", options=format_keys, index=default_format_index)
        output_format = format_options[format_display]
        
        st.divider()
        
        st.header("API Keys")
        
        tavily_key = settings["api_keys"].get("tavily", "")
        if not tavily_key:
            tavily_key = st.text_input("Tavily API Key", "", type="password")
        else:
            st.success("✅ Tavily API Key set")
        
        # LLM provider selection
        llm_provider = st.selectbox(
            "LLM Provider", 
            ["OpenAI", "Silicon Flow"], 
            index=0 if settings["preferences"].get("preferred_provider") == "OpenAI" else 1
        )
        
        use_llm = st.checkbox("Use LLM for Enhanced Results", True)
        
        # Get appropriate API key and preferred model based on provider
        if llm_provider == "OpenAI":
            llm_key = settings["api_keys"].get("openai", "")
            preferred_model = settings["preferences"].get("preferred_openai_model", "")
        else:  # Silicon Flow
            llm_key = settings["api_keys"].get("siliconflow", "")
            preferred_model = settings["preferences"].get("preferred_siliconflow_model", "")
        
        # Prompt for API key if not available
        if not llm_key and use_llm:
            llm_key = st.text_input(f"{llm_provider} API Key", "", type="password")
        elif llm_key:
            st.success(f"✅ {llm_provider} API Key set")
        
        # Get cached models
        provider_key = llm_provider.lower()
        cached_models = settings["models_cache"].get(provider_key, [])
        
        # Use asyncio to fetch models if needed and not already in cache
        if use_llm and llm_key and not cached_models:
            with st.spinner(f"Checking available {llm_provider} models..."):
                models = asyncio.run(fetch_models(llm_provider, llm_key))
                if models:
                    cached_models = models
        
        # Default models if none available
        if not cached_models:
            if llm_provider == "OpenAI":
                cached_models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
            else:
                cached_models = ["Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen1.5-110B-Chat"]
        
        # Determine selected model index
        selected_model_index = 0
        if preferred_model in cached_models:
            selected_model_index = cached_models.index(preferred_model)
        
        llm_model = st.selectbox(
            f"{llm_provider} Model", 
            cached_models,
            index=min(selected_model_index, len(cached_models)-1),
            disabled=not use_llm or not llm_key
        )
        
        st.divider()
        
        # Source control options
        st.header("Source Control")
        
        exclude_wikipedia = st.checkbox("Exclude Wikipedia", settings["preferences"].get("exclude_wikipedia", True))
        prefer_official = st.checkbox("Prefer Official Sources", settings["preferences"].get("prefer_official", True))
        
        st.divider()
    
        # Section selection
        st.header("Report Sections")
        selected_section_ids = []
        section_titles = {}
        
        if TAVILY_IMPORTED:
            try:
                temp_analyzer = CompanyAnalyzer(
                    tavily_api_key="temp" if not tavily_key else tavily_key,
                    use_llm=False
                )
                all_sections = temp_analyzer._get_analysis_sections(language)
                
                section_options = {section["id"]: section["title"] for section in all_sections}
                selected_sections = {}
                
                section_titles = section_options
                
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
        
        # Navigation buttons
        st.divider()
        settings_col, start_col = st.columns(2)
        
        with settings_col:
            if st.button("Settings", use_container_width=True):
                go_to_settings()
                st.rerun()
        
        with start_col:
            # 使用线程管理器状态控制按钮
            start_disabled = is_running
            if st.button("Start Analysis", type="primary", disabled=start_disabled, use_container_width=True):
                # 检验必要条件
                if not tavily_key:
                    st.error("Please enter your Tavily API Key")
                elif use_llm and not llm_key:
                    st.error(f"Please enter your {llm_provider} API Key or disable LLM integration")
                elif not selected_section_ids:
                    st.error("Please select at least one section to analyze")
                else:
                    # 启动分析工作线程
                    thread_manager.start_worker(
                        company_name=company_name,
                        language=language,
                        output_format=output_format,
                        use_llm=use_llm,
                        llm_provider=llm_provider,
                        llm_model=llm_model,
                        llm_key=llm_key,
                        tavily_key=tavily_key,
                        selected_section_ids=selected_section_ids,
                        section_titles=section_titles,
                        exclude_domains=["wikipedia.org", "wikimedia.org"] if exclude_wikipedia else None,
                        prefer_official=prefer_official
                    )
                    # 立即重新运行
                    st.rerun()
        
        # 显示停止按钮
        if is_running:
            if st.button("Stop Analysis", type="secondary", use_container_width=True):
                thread_manager.stop_worker()
                st.rerun()
        
        # 显示重置按钮
        if is_running or is_completed:
            if st.button("Start New Analysis", use_container_width=True):
                thread_manager.stop_worker()
                st.rerun()
    
    # 主内容区域 - 使用与 app_test3.py 相似的布局
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # 状态和日志区域
        st.markdown("<div class='section-header'>Analysis Status:</div>", unsafe_allow_html=True)
        status_container = st.empty()
        
        # 显示当前状态
        with status_container:
            if is_running:
                st.warning("Process running...", icon="⚙️")
            elif is_completed:
                st.success("Process completed successfully!", icon="✅")
            else:
                st.info("Click 'Start Analysis' to begin")
        
        # 日志区域
        st.markdown("<div class='section-header'>Analysis Logs:</div>", unsafe_allow_html=True)
        log_container = st.container()
        
        # 显示日志 
        with log_container:
            if worker and worker.user_logs:
                st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                log_html = ""
                for log in worker.user_logs[-10:]:  # 显示最近10条用户友好日志
                    log_html += f'<div class="log-entry">{log}</div>'
                st.markdown(log_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # 完整技术日志视图（可展开）
                with st.expander("View Full Technical Log"):
                    st.markdown("<div style='max-height: 400px; overflow-y: auto;'>", unsafe_allow_html=True)
                    for log in worker.logs:  # 使用完整的技术日志
                        st.code(log, language=None)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='output-container'>", unsafe_allow_html=True)
                st.text("Logs will appear here once the process starts...")
                st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # 进度区域
        st.markdown("<div class='section-header'>Analysis Progress:</div>", unsafe_allow_html=True)
        progress_container = st.container()
        
        # 显示进度
        with progress_container:
            if worker and (is_running or is_completed):
                # 显示总体进度
                st.header("Overall Progress")
                progress_value = worker.overall_progress / 100 if is_running else 1.0
                st.progress(progress_value)
                st.metric("Completion", f"{worker.overall_progress:.1f}%" if is_running else "100%")
                
                # 显示阶段进度
                st.header("Section Progress")
                for section in worker.sections:
                    status_class = f"phase-{section['status']}"
                    status_icon = "⏳" if section["status"] == "running" else "✅" if section["status"] == "complete" else "⏱️"
                    completion_time = f" ({section.get('completion_time', 0):.1f}s)" if section.get("completion_time") is not None else ""
                    
                    st.markdown(f"""
                    <div class="phase-container {status_class}">
                        <div>{status_icon} {section["title"]}{completion_time}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                # 显示空进度
                st.header("Overall Progress")
                st.progress(0)
                st.metric("Completion", "0%")
                
                # 显示空阶段进度
                st.header("Section Progress")
                if selected_section_ids:
                    for section_id in selected_section_ids:
                        title = section_titles.get(section_id, section_id)
                        st.markdown(f"""
                        <div class="phase-container phase-pending">
                            <div>⏱️ {title} - 0%</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Select sections to analyze in the sidebar")
    
    # 报告区域 - 放在主内容区域下方
    report_container = st.container()
    
    # 显示报告
    with report_container:
        st.header("Analysis Report")
        if worker and worker.completed and worker.report_content:
            # 创建标签页
            view_tab, download_tab = st.tabs(["View Report", "Download"])
            
            with view_tab:
                if output_format == "markdown":
                    st.markdown(worker.report_content)
                elif output_format == "html":
                    st.components.v1.html(worker.report_content, height=600, scrolling=True)
                elif output_format == "json":
                    try:
                        st.json(json.loads(worker.report_content))
                    except:
                        st.text(worker.report_content)
                else:  # text
                    st.text(worker.report_content)
            
            with download_tab:
                # 提供下载按钮
                st.download_button(
                    "Download Report",
                    worker.report_content,
                    file_name=f"{company_name.replace(' ', '_')}_analysis.{output_format}",
                    mime="text/plain",
                    key="download_button"
                )
                
                # 显示保存文件信息
                if worker.report_path:
                    st.success(f"Report saved to: {worker.report_path}")
                
        elif not is_running:
            st.info("The analysis report will appear here once processing is complete.")

# 主函数
def main():
    # 基于会话状态显示相应页面
    if st.session_state.page == 'settings':
        show_settings_page()
    else:
        show_main_page()
    
    # 检查是否需要强制刷新
    thread_manager = get_thread_manager()
    if thread_manager.is_running():
        # 使用固定的刷新间隔 - 0.5秒提供良好的响应性
        time.sleep(0.5)
        st.rerun()

# Run the app
if __name__ == "__main__":
    main()