

# File: config.py

# config file
import json
import os


class Config:
    """Config class for GPT Researcher."""

    def __init__(self, config_file: str = None):
        """Initialize the config class."""
        self.config_file = config_file if config_file else os.getenv('CONFIG_FILE')
        self.retriever = os.getenv('SEARCH_RETRIEVER', "tavily")
        self.embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'openai')
        self.llm_provider = os.getenv('LLM_PROVIDER', "openai")
        self.fast_llm_model = os.getenv('FAST_LLM_MODEL', "gpt-3.5-turbo-16k")
        self.smart_llm_model = os.getenv('SMART_LLM_MODEL', "gpt-4-turbo")
        self.fast_token_limit = int(os.getenv('FAST_TOKEN_LIMIT', 2000))
        self.smart_token_limit = int(os.getenv('SMART_TOKEN_LIMIT', 4000))
        self.browse_chunk_max_length = int(os.getenv('BROWSE_CHUNK_MAX_LENGTH', 8192))
        self.summary_token_limit = int(os.getenv('SUMMARY_TOKEN_LIMIT', 700))
        self.temperature = float(os.getenv('TEMPERATURE', 0.55))
        self.user_agent = os.getenv('USER_AGENT', "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                                   "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0")
        self.max_search_results_per_query = int(os.getenv('MAX_SEARCH_RESULTS_PER_QUERY', 5))
        self.memory_backend = os.getenv('MEMORY_BACKEND', "local")
        self.total_words = int(os.getenv('TOTAL_WORDS', 1000))
        self.report_format = os.getenv('REPORT_FORMAT', "APA")
        self.max_iterations = int(os.getenv('MAX_ITERATIONS', 3))
        self.agent_role = os.getenv('AGENT_ROLE', None)
        self.scraper = os.getenv("SCRAPER", "bs")
        self.max_subtopics = os.getenv("MAX_SUBTOPICS", 3)

        self.load_config_file()

    def load_config_file(self) -> None:
        """Load the config file."""
        if self.config_file is None:
            return None
        with open(self.config_file, "r") as f:
            config = json.load(f)
        for key, value in config.items():
            self.__dict__[key] = value



# File: __init__.py

from .config import Config

__all__ = ['Config']

# File: compression.py

from .retriever import SearchAPIRetriever
from langchain.retrievers import (
    ContextualCompressionRetriever,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ContextCompressor:
    def __init__(self, documents, embeddings, max_results=5, **kwargs):
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = 0.38

    def _get_contextual_retriever(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        relevance_filter = EmbeddingsFilter(embeddings=self.embeddings,
                                            similarity_threshold=self.similarity_threshold)
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, relevance_filter]
        )
        base_retriever = SearchAPIRetriever(
            pages=self.documents
        )
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor, base_retriever=base_retriever
        )
        return contextual_retriever

    def _pretty_print_docs(self, docs, top_n):
        return f"\n".join(f"Source: {d.metadata.get('source')}\n"
                          f"Title: {d.metadata.get('title')}\n"
                          f"Content: {d.page_content}\n"
                          for i, d in enumerate(docs) if i < top_n)

    def get_context(self, query, max_results=5):
        compressed_docs = self._get_contextual_retriever()
        relevant_docs = compressed_docs.get_relevant_documents(query)
        return self._pretty_print_docs(relevant_docs, max_results)

# File: retriever.py

import os
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever


class SearchAPIRetriever(BaseRetriever):
    """Search API retriever."""
    pages: List[Dict] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        docs = [
            Document(
                page_content=page.get("raw_content", ""),
                metadata={
                    "title": page.get("title", ""),
                    "source": page.get("url", ""),
                },
            )
            for page in self.pages
        ]

        return docs


# File: __init__.py

from .compression import ContextCompressor
from .retriever import SearchAPIRetriever

__all__ = ['ContextCompressor', 'SearchAPIRetriever']


# File: azureopenai.py

import os

from colorama import Fore, Style
from langchain_openai import AzureChatOpenAI

'''
Please note:
Needs additional env vars such as: 
    AZURE_OPENAI_ENDPOINT  e.g. https://xxxx.openai.azure.com/",
    AZURE_OPENAI_API_KEY e.g "xxxxxxxxxxxxxxxxxxxxx",
    OPENAI_API_VERSION, e.g. "2024-03-01-preview" but needs to updated over time as API verison updates,
    AZURE_EMBEDDING_MODEL e.g. "ada2" The Azure OpenAI embedding model deployment name.

config.py settings for Azure OpenAI should look like:
    self.embedding_provider = os.getenv('EMBEDDING_PROVIDER', 'azureopenai')
    self.llm_provider = os.getenv('LLM_PROVIDER', "azureopenai")
    self.fast_llm_model = os.getenv('FAST_LLM_MODEL', "gpt-3.5-turbo-16k") #Deployment name of your GPT3.5T model as per azure OpenAI studio deployment section
    self.smart_llm_model = os.getenv('SMART_LLM_MODEL', "gpt4")  #Deployment name of your GPT4 1106-Preview+ (GPT4T) model as per azure OpenAI studio deployment section
'''
class AzureOpenAIProvider:

    def __init__(
        self,
        deployment_name,
        temperature,
        max_tokens
    ):
        self.deployment_name = deployment_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = self.get_api_key()
        self.llm = self.get_llm_model()

    def get_api_key(self):
        """
        Gets the OpenAI API key
        Returns:

        """
        try:
            api_key = os.environ["AZURE_OPENAI_API_KEY"]
        except:
            raise Exception(
                "Azure OpenAI API key not found. Please set the AZURE_OPENAI_API_KEY environment variable.")
        return api_key

    def get_llm_model(self):
        # Initializing the chat model
        llm = AzureChatOpenAI(
            deployment_name=self.deployment_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )

        return llm

    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            # Getting output from the model chain using ainvoke for asynchronous invoking
            output = await self.llm.ainvoke(messages)

            return output.content

        else:
            return await self.stream_response(messages, websocket)

    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    if websocket is not None:
                        await websocket.send_json({"type": "report", "output": paragraph})
                    else:
                        print(f"{Fore.GREEN}{paragraph}{Style.RESET_ALL}")
                    paragraph = ""
                    
        return response


# File: __init__.py



# File: __init__.py



# File: google.py

import os

from colorama import Fore, Style
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


class GoogleProvider:

    def __init__(
        self,
        model,
        temperature,
        max_tokens
    ):
        # May be extended to support more google models in the future
        self.model = "gemini-pro"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = self.get_api_key()
        self.llm = self.get_llm_model()

    def get_api_key(self):
        """
        Gets the GEMINI_API_KEY
        Returns:

        """
        try:
            api_key = os.environ["GEMINI_API_KEY"]
        except:
            raise Exception(
                "GEMINI API key not found. Please set the GEMINI_API_KEY environment variable.")
        return api_key

    def get_llm_model(self):
        # Initializing the chat model
        llm = ChatGoogleGenerativeAI(
            convert_system_message_to_human=True,
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            google_api_key=self.api_key
        )

        return llm

    def convert_messages(self, messages):
        """
        The function `convert_messages` converts messages based on their role into either SystemMessage
        or HumanMessage objects.
        
        Args:
          messages: It looks like the code snippet you provided is a function called `convert_messages`
        that takes a list of messages as input and converts each message based on its role into either a
        `SystemMessage` or a `HumanMessage`.
        
        Returns:
          The `convert_messages` function is returning a list of converted messages based on the input
        `messages`. The function checks the role of each message in the input list and creates a new
        `SystemMessage` object if the role is "system" or a new `HumanMessage` object if the role is
        "user". The function then returns a list of these converted messages.
        """
        converted_messages = []
        for message in messages:
            if message["role"] == "system":
                converted_messages.append(
                    SystemMessage(content=message["content"]))
            elif message["role"] == "user":
                converted_messages.append(
                    HumanMessage(content=message["content"]))

        return converted_messages

    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            # Getting output from the model chain using ainvoke for asynchronous invoking
            converted_messages = self.convert_messages(messages)
            output = await self.llm.ainvoke(converted_messages)

            return output.content

        else:
            return await self.stream_response(messages, websocket)

    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    if websocket is not None:
                        await websocket.send_json({"type": "report", "output": paragraph})
                    else:
                        print(f"{Fore.GREEN}{paragraph}{Style.RESET_ALL}")
                    paragraph = ""

        return response


# File: __init__.py

from .google.google import GoogleProvider
from .openai.openai import OpenAIProvider
from .azureopenai.azureopenai import AzureOpenAIProvider

__all__ = [
    "GoogleProvider",
    "OpenAIProvider",
    "AzureOpenAIProvider"
]


# File: __init__.py



# File: openai.py

import os

from colorama import Fore, Style
from langchain_openai import ChatOpenAI


class OpenAIProvider:

    def __init__(
        self,
        model,
        temperature,
        max_tokens
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = self.get_api_key()
        self.llm = self.get_llm_model()

    def get_api_key(self):
        """
        Gets the OpenAI API key
        Returns:

        """
        try:
            api_key = os.environ["OPENAI_API_KEY"]
        except:
            raise Exception(
                "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return api_key

    def get_llm_model(self):
        # Initializing the chat model
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=self.api_key
        )

        return llm

    async def get_chat_response(self, messages, stream, websocket=None):
        if not stream:
            # Getting output from the model chain using ainvoke for asynchronous invoking
            output = await self.llm.ainvoke(messages)

            return output.content

        else:
            return await self.stream_response(messages, websocket)

    async def stream_response(self, messages, websocket=None):
        paragraph = ""
        response = ""

        # Streaming the response using the chain astream method from langchain
        async for chunk in self.llm.astream(messages):
            content = chunk.content
            if content is not None:
                response += content
                paragraph += content
                if "\n" in paragraph:
                    if websocket is not None:
                        await websocket.send_json({"type": "report", "output": paragraph})
                    else:
                        print(f"{Fore.GREEN}{paragraph}{Style.RESET_ALL}")
                    paragraph = ""
                    
        return response


# File: functions.py

import asyncio
import json

import markdown

from gpt_researcher.master.prompts import *
from gpt_researcher.scraper.scraper import Scraper
from gpt_researcher.utils.llm import *

def get_retriever(retriever):
    """
    Gets the retriever
    Args:
        retriever: retriever name

    Returns:
        retriever: Retriever class

    """
    match retriever:
        case "tavily":
            from gpt_researcher.retrievers import TavilySearch
            retriever = TavilySearch
        case "tavily_news":
            from gpt_researcher.retrievers import TavilyNews
            retriever = TavilyNews
        case "google":
            from gpt_researcher.retrievers import GoogleSearch
            retriever = GoogleSearch
        case "searx":
            from gpt_researcher.retrievers import SearxSearch
            retriever = SearxSearch
        case "serpapi":
            raise NotImplementedError(
                "SerpApiSearch is not fully implemented yet.")
            from gpt_researcher.retrievers import SerpApiSearch
            retriever = SerpApiSearch
        case "googleSerp":
            from gpt_researcher.retrievers import SerperSearch
            retriever = SerperSearch
        case "duckduckgo":
            from gpt_researcher.retrievers import Duckduckgo
            retriever = Duckduckgo
        case "BingSearch":
            from gpt_researcher.retrievers import BingSearch
            retriever = BingSearch

        case _:
            raise Exception("Retriever not found.")

    return retriever


async def choose_agent(query, cfg):
    """
    Chooses the agent automatically
    Args:
        query: original query
        cfg: Config

    Returns:
        agent: Agent name
        agent_role_prompt: Agent role prompt
    """
    try:
        response = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {query}"}],
            temperature=0,
            llm_provider=cfg.llm_provider
        )
        agent_dict = json.loads(response)
        return agent_dict["server"], agent_dict["agent_role_prompt"]
    except Exception as e:
        return "Default Agent", "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."


async def get_sub_queries(query: str, agent_role_prompt: str, cfg, parent_query: str, report_type:str):
    """
    Gets the sub queries
    Args:
        query: original query
        agent_role_prompt: agent role prompt
        cfg: Config

    Returns:
        sub_queries: List of sub queries

    """
    max_research_iterations = cfg.max_iterations if cfg.max_iterations else 1
    response = await create_chat_completion(
        model=cfg.smart_llm_model,
        messages=[
            {"role": "system", "content": f"{agent_role_prompt}"},
            {"role": "user", "content": generate_search_queries_prompt(query, parent_query, report_type, max_iterations=max_research_iterations)}],
        temperature=0,
        llm_provider=cfg.llm_provider
    )
    sub_queries = json.loads(response)
    return sub_queries


def scrape_urls(urls, cfg=None):
    """
    Scrapes the urls
    Args:
        urls: List of urls
        cfg: Config (optional)

    Returns:
        text: str

    """
    content = []
    user_agent = cfg.user_agent if cfg else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    try:
        content = Scraper(urls, user_agent, cfg.scraper).run()
    except Exception as e:
        print(f"{Fore.RED}Error in scrape_urls: {e}{Style.RESET_ALL}")
    return content


async def summarize(query, content, agent_role_prompt, cfg, websocket=None):
    """
    Asynchronously summarizes a list of URLs.

    Args:
        query (str): The search query.
        content (list): List of dictionaries with 'url' and 'raw_content'.
        agent_role_prompt (str): The role prompt for the agent.
        cfg (object): Configuration object.

    Returns:
        list: A list of dictionaries with 'url' and 'summary'.
    """

    # Function to handle each summarization task for a chunk
    async def handle_task(url, chunk):
        summary = await summarize_url(query, chunk, agent_role_prompt, cfg)
        if summary:
            await stream_output("logs", f"🌐 Summarizing url: {url}", websocket)
            await stream_output("logs", f"📃 {summary}", websocket)
        return url, summary

    # Function to split raw content into chunks of 10,000 words
    def chunk_content(raw_content, chunk_size=10000):
        words = raw_content.split()
        for i in range(0, len(words), chunk_size):
            yield ' '.join(words[i:i+chunk_size])

    # Process each item one by one, but process chunks in parallel
    concatenated_summaries = []
    for item in content:
        url = item['url']
        raw_content = item['raw_content']

        # Create tasks for all chunks of the current URL
        chunk_tasks = [handle_task(url, chunk)
                       for chunk in chunk_content(raw_content)]

        # Run chunk tasks concurrently
        chunk_summaries = await asyncio.gather(*chunk_tasks)

        # Aggregate and concatenate summaries for the current URL
        summaries = [summary for _, summary in chunk_summaries if summary]
        concatenated_summary = ' '.join(summaries)
        concatenated_summaries.append(
            {'url': url, 'summary': concatenated_summary})

    return concatenated_summaries


async def summarize_url(query, raw_data, agent_role_prompt, cfg):
    """
    Summarizes the text
    Args:
        query:
        raw_data:
        agent_role_prompt:
        cfg:

    Returns:
        summary: str

    """
    summary = ""
    try:
        summary = await create_chat_completion(
            model=cfg.fast_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": f"{generate_summary_prompt(query, raw_data)}"}],
            temperature=0,
            llm_provider=cfg.llm_provider
        )
    except Exception as e:
        print(f"{Fore.RED}Error in summarize: {e}{Style.RESET_ALL}")
    return summary


async def generate_report(
    query,
    context,
    agent_role_prompt,
    report_type,
    websocket,
    cfg,
    main_topic: str = "",
    existing_headers: list = []
):
    """
    generates the final report
    Args:
        query:
        context:
        agent_role_prompt:
        report_type:
        websocket:
        cfg:
        main_topic:
        existing_headers:

    Returns:
        report:

    """
    generate_prompt = get_prompt_by_report_type(report_type)
    report = ""

    if report_type == "subtopic_report":
        content = f"{generate_prompt(query, existing_headers, main_topic, context, cfg.report_format, cfg.total_words)}"
    else:
        content = (
            f"{generate_prompt(query, context, cfg.report_format, cfg.total_words)}")

    try:
        report = await create_chat_completion(
            model=cfg.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{agent_role_prompt}"},
                {"role": "user", "content": content}],
            temperature=0,
            llm_provider=cfg.llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=cfg.smart_token_limit
        )
    except Exception as e:
        print(f"{Fore.RED}Error in generate_report: {e}{Style.RESET_ALL}")

    return report


async def stream_output(type, output, websocket=None, logging=True):
    """
    Streams output to the websocket
    Args:
        type:
        output:

    Returns:
        None
    """
    if not websocket or logging:
        print(output)

    if websocket:
        await websocket.send_json({"type": type, "output": output})

async def get_report_introduction(query, context, role, config, websocket=None):
    try:
        introduction = await create_chat_completion(
            model=config.smart_llm_model,
            messages=[
                {"role": "system", "content": f"{role}"},
                {"role": "user", "content": generate_report_introduction(query, context)}],
            temperature=0,
            llm_provider=config.llm_provider,
            stream=True,
            websocket=websocket,
            max_tokens=config.smart_token_limit
        )
        
        return introduction
    except Exception as e:
        print(f"{Fore.RED}Error in generating report introduction: {e}{Style.RESET_ALL}")

    return ""
    
def extract_headers(markdown_text: str):
    # Function to extract headers from markdown text

    headers = []
    parsed_md = markdown.markdown(markdown_text)  # Parse markdown text
    lines = parsed_md.split("\n")  # Split text into lines

    stack = []  # Initialize stack to keep track of nested headers
    for line in lines:
        if line.startswith("<h") and len(line) > 1:  # Check if the line starts with an HTML header tag
            level = int(line[2])  # Extract header level
            header_text = line[
                line.index(">") + 1: line.rindex("<")
            ]  # Extract header text

            # Pop headers from the stack with higher or equal level
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            header = {
                "level": level,
                "text": header_text,
            }  # Create header dictionary
            if stack:
                stack[-1].setdefault("children", []).append(
                    header
                )  # Append as child if parent exists
            else:
                # Append as top-level header if no parent exists
                headers.append(header)

            stack.append(header)  # Push header onto the stack

    return headers  # Return the list of headers


def table_of_contents(markdown_text: str):
    try:
        # Function to generate table of contents recursively
        def generate_table_of_contents(headers, indent_level=0):
            toc = ""  # Initialize table of contents string
            for header in headers:
                toc += (
                    " " * (indent_level * 4) + "- " + header["text"] + "\n"
                )  # Add header text with indentation
                if "children" in header:  # If header has children
                    toc += generate_table_of_contents(
                        header["children"], indent_level + 1
                    )  # Generate TOC for children
            return toc  # Return the generated table of contents

        # Extract headers from markdown text
        headers = extract_headers(markdown_text)
        toc = "## Table of Contents\n\n" + generate_table_of_contents(
            headers
        )  # Generate table of contents

        return toc  # Return the generated table of contents

    except Exception as e:
        print("table_of_contents Exception : ", e)  # Print exception if any
        return markdown_text  # Return original markdown text if an exception occurs

def add_source_urls(report_markdown: str, visited_urls: set):
    """
    This function takes a Markdown report and a set of visited URLs as input parameters.
    
    Args:
      report_markdown (str): The `add_source_urls` function takes in two parameters:
      visited_urls (set): Visited_urls is a set that contains URLs that have already been visited. This
    parameter is used to keep track of which URLs have already been included in the report_markdown to
    avoid duplication.
    """
    try:
        url_markdown = "\n\n\n## References\n\n"

        url_markdown += "".join(f"- [{url}]({url})\n" for url in visited_urls)

        updated_markdown_report = report_markdown + url_markdown

        return updated_markdown_report

    except Exception as e:
        print(f"Encountered exception in adding source urls : {e}")
        return report_markdown

# File: __init__.py

from .agent import GPTResearcher

__all__ = ['GPTResearcher']

# File: prompts.py

from datetime import datetime, timezone
import warnings
from gpt_researcher.utils.enum import ReportType


def generate_search_queries_prompt(question: str, parent_query: str, report_type: str, max_iterations: int=3,):
    """ Generates the search queries prompt for the given question.
    Args: 
        question (str): The question to generate the search queries prompt for
        parent_query (str): The main question (only relevant for detailed reports)
        report_type (str): The report type
        max_iterations (int): The maximum number of search queries to generate
    
    Returns: str: The search queries prompt for the given question
    """
    
    if report_type == ReportType.DetailedReport.value or report_type == ReportType.SubtopicReport.value:
        task = f"{parent_query} - {question}"
    else:
        task = question

    return f'Write {max_iterations} google search queries to search online that form an objective opinion from the following task: "{task}"' \
           f'Use the current date if needed: {datetime.now().strftime("%B %d, %Y")}.\n' \
           f'Also include in the queries specified task details such as locations, names, etc.\n' \
           f'You must respond with a list of strings in the following format: ["query 1", "query 2", "query 3"].'


def generate_report_prompt(question, context, report_format="apa", total_words=1000):
    """ Generates the report prompt for the given question and research summary.
    Args: question (str): The question to generate the report prompt for
            research_summary (str): The research summary to generate the report prompt for
    Returns: str: The report prompt for the given question and research summary
    """

    return f'Information: """{context}"""\n\n' \
           f'Using the above information, answer the following' \
           f' query or task: "{question}" in a detailed report --' \
           " The report should focus on the answer to the query, should be well structured, informative," \
           f" in depth and comprehensive, with facts and numbers if available and a minimum of {total_words} words.\n" \
           "You should strive to write the report as long as you can using all relevant and necessary information provided.\n" \
           "You must write the report with markdown syntax.\n " \
           f"Use an unbiased and journalistic tone. \n" \
           "You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.\n" \
           f"You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.\n" \
           "Every url should be hyperlinked: [url website](url)"\
           """
            Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report : 
        
            eg:    
                # Report Header
                
                This is a sample text. ([url website](url))
            """\
            f"You MUST write the report in {report_format} format.\n " \
            f"Cite search results using inline notations. Only cite the most \
            relevant results that answer the query accurately. Place these citations at the end \
            of the sentence or paragraph that reference them.\n"\
            f"Please do your best, this is very important to my career. " \
            f"Assume that the current date is {datetime.now().strftime('%B %d, %Y')}"


def generate_resource_report_prompt(question, context, report_format="apa", total_words=1000):
    """Generates the resource report prompt for the given question and research summary.

    Args:
        question (str): The question to generate the resource report prompt for.
        context (str): The research summary to generate the resource report prompt for.

    Returns:
        str: The resource report prompt for the given question and research summary.
    """
    return f'"""{context}"""\n\nBased on the above information, generate a bibliography recommendation report for the following' \
           f' question or topic: "{question}". The report should provide a detailed analysis of each recommended resource,' \
           ' explaining how each source can contribute to finding answers to the research question.\n' \
           'Focus on the relevance, reliability, and significance of each source.\n' \
           'Ensure that the report is well-structured, informative, in-depth, and follows Markdown syntax.\n' \
           'Include relevant facts, figures, and numbers whenever available.\n' \
           'The report should have a minimum length of 700 words.\n' \
        'You MUST include all relevant source urls.'\
        'Every url should be hyperlinked: [url website](url)'


def generate_custom_report_prompt(query_prompt, context, report_format="apa", total_words=1000):
    return f'"{context}"\n\n{query_prompt}'


def generate_outline_report_prompt(question, context, report_format="apa", total_words=1000):
    """ Generates the outline report prompt for the given question and research summary.
    Args: question (str): The question to generate the outline report prompt for
            research_summary (str): The research summary to generate the outline report prompt for
    Returns: str: The outline report prompt for the given question and research summary
    """

    return f'"""{context}""" Using the above information, generate an outline for a research report in Markdown syntax' \
           f' for the following question or topic: "{question}". The outline should provide a well-structured framework' \
           ' for the research report, including the main sections, subsections, and key points to be covered.' \
           ' The research report should be detailed, informative, in-depth, and a minimum of 1,200 words.' \
           ' Use appropriate Markdown syntax to format the outline and ensure readability.'


def auto_agent_instructions():
    return """
        This task involves researching a given topic, regardless of its complexity or the availability of a definitive answer. The research is conducted by a specific server, defined by its type and role, with each server requiring distinct instructions.
        Agent
        The server is determined by the field of the topic and the specific name of the server that could be utilized to research the topic provided. Agents are categorized by their area of expertise, and each server type is associated with a corresponding emoji.

        examples:
        task: "should I invest in apple stocks?"
        response: 
        {
            "server": "💰 Finance Agent",
            "agent_role_prompt: "You are a seasoned finance analyst AI assistant. Your primary goal is to compose comprehensive, astute, impartial, and methodically arranged financial reports based on provided data and trends."
        }
        task: "could reselling sneakers become profitable?"
        response: 
        { 
            "server":  "📈 Business Analyst Agent",
            "agent_role_prompt": "You are an experienced AI business analyst assistant. Your main objective is to produce comprehensive, insightful, impartial, and systematically structured business reports based on provided business data, market trends, and strategic analysis."
        }
        task: "what are the most interesting sites in Tel Aviv?"
        response:
        {
            "server:  "🌍 Travel Agent",
            "agent_role_prompt": "You are a world-travelled AI tour guide assistant. Your main purpose is to draft engaging, insightful, unbiased, and well-structured travel reports on given locations, including history, attractions, and cultural insights."
        }
    """


def generate_summary_prompt(query, data):
    """ Generates the summary prompt for the given question and text.
    Args: question (str): The question to generate the summary prompt for
            text (str): The text to generate the summary prompt for
    Returns: str: The summary prompt for the given question and text
    """

    return f'{data}\n Using the above text, summarize it based on the following task or query: "{query}".\n If the ' \
           f'query cannot be answered using the text, YOU MUST summarize the text in short.\n Include all factual ' \
           f'information such as numbers, stats, quotes, etc if available. '


################################################################################################

# DETAILED REPORT PROMPTS

def generate_subtopics_prompt() -> str:
    return """
                Provided the main topic:
                
                {task}
                
                and research data:
                
                {data}
                
                - Construct a list of subtopics which indicate the headers of a report document to be generated on the task. 
                - These are a possible list of subtopics : {subtopics}.
                - There should NOT be any duplicate subtopics.
                - Limit the number of subtopics to a maximum of {max_subtopics}
                - Finally order the subtopics by their tasks, in a relevant and meaningful order which is presentable in a detailed report
                
                "IMPORTANT!":
                - Every subtopic MUST be relevant to the main topic and provided research data ONLY!
                
                {format_instructions}
            """


def generate_subtopic_report_prompt(
    current_subtopic,
    existing_headers,
    main_topic,
    context,
    report_format="apa",
    total_words=1000,
    max_subsections=5,
) -> str:

    return f"""
    "Context":
    "{context}"
    
    "Main Topic and Subtopic":
    Using the latest information available, construct a detailed report on the subtopic: {current_subtopic} under the main topic: {main_topic}.
    You must limit the number of subsections to a maximum of {max_subsections}.
    
    "Content Focus":
    - The report should focus on answering the question, be well-structured, informative, in-depth, and include facts and numbers if available.
    - Use markdown syntax and follow the {report_format.upper()} format.
    
    "Structure and Formatting":
    - As this sub-report will be part of a larger report, include only the main body divided into suitable subtopics without any introduction or conclusion section.
    
    - You MUST include markdown hyperlinks to relevant source URLs wherever referenced in the report, for example:
    
        # Report Header
        
        This is a sample text. ([url website](url))
    
    "Existing Subtopic Reports":
    - This is a list of existing subtopic reports and their section headers:
    
        {existing_headers}.
    
    - Do not use any of the above headers or related details to avoid duplicates. Use smaller Markdown headers (e.g., H2 or H3) for content structure, avoiding the largest header (H1) as it will be used for the larger report's heading.
    
    "Date":
    Assume the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.
    
    "IMPORTANT!":
    - The focus MUST be on the main topic! You MUST Leave out any information un-related to it!
    - Must NOT have any introduction, conclusion, summary or reference section.
    - You MUST include hyperlinks with markdown syntax ([url website](url)) related to the sentences wherever necessary.
    """


def generate_report_introduction(question: str, research_summary: str = "") -> str:
    return f"""{research_summary}\n 
        Using the above latest information, Prepare a detailed report introduction on the topic -- {question}.
        - The introduction should be succinct, well-structured, informative with markdown syntax.
        - As this introduction will be part of a larger report, do NOT include any other sections, which are generally present in a report.
        - The introduction should be preceded by an H1 heading with a suitable topic for the entire report.
        - You must include hyperlinks with markdown syntax ([url website](url)) related to the sentences wherever necessary.
        Assume that the current date is {datetime.now(timezone.utc).strftime('%B %d, %Y')} if required.
    """


report_type_mapping = {
    ReportType.ResearchReport.value: generate_report_prompt,
    ReportType.ResourceReport.value: generate_resource_report_prompt,
    ReportType.OutlineReport.value: generate_outline_report_prompt,
    ReportType.CustomReport.value: generate_custom_report_prompt,
    ReportType.SubtopicReport.value: generate_subtopic_report_prompt
}


def get_prompt_by_report_type(report_type):
    prompt_by_type = report_type_mapping.get(report_type)
    default_report_type = ReportType.ResearchReport.value
    if not prompt_by_type:
        warnings.warn(f"Invalid report type: {report_type}.\n"
                        f"Please use one of the following: {', '.join([enum_value for enum_value in report_type_mapping.keys()])}\n"
                        f"Using default report type: {default_report_type} prompt.",
                        UserWarning)
        prompt_by_type = report_type_mapping.get(default_report_type)
    return prompt_by_type


# File: agent.py

import asyncio
import time

from gpt_researcher.config import Config
from gpt_researcher.context.compression import ContextCompressor
from gpt_researcher.master.functions import *
from gpt_researcher.memory import Memory
from gpt_researcher.utils.enum import ReportType


class GPTResearcher:
    """
    GPT Researcher
    """

    def __init__(
        self,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        source_urls=None,
        config_path=None,
        websocket=None,
        agent=None,
        role=None,
        parent_query: str = "",
        subtopics: list = [],
        visited_urls: set = set()
    ):
        """
        Initialize the GPT Researcher class.
        Args:
            query: str,
            report_type: str
            source_urls
            config_path
            websocket
            agent
            role
            parent_query: str
            subtopics: list
            visited_urls: set
        """
        self.query = query
        self.agent = agent
        self.role = role
        self.report_type = report_type
        self.report_prompt = get_prompt_by_report_type(self.report_type)  # this validates the report type
        self.websocket = websocket
        self.cfg = Config(config_path)
        self.retriever = get_retriever(self.cfg.retriever)
        self.context = []
        self.source_urls = source_urls
        self.memory = Memory(self.cfg.embedding_provider)
        self.visited_urls = visited_urls

        # Only relevant for DETAILED REPORTS
        # --------------------------------------

        # Stores the main query of the detailed report
        self.parent_query = parent_query

        # Stores all the user provided subtopics
        self.subtopics = subtopics

    async def conduct_research(self):
        """
        Runs the GPT Researcher to conduct research
        """
        print(f"🔎 Running research for '{self.query}'...")
        
        # Generate Agent
        if not (self.agent and self.role):
            self.agent, self.role = await choose_agent(self.query, self.cfg)
        await stream_output("logs", self.agent, self.websocket)

        # If specified, the researcher will use the given urls as the context for the research.
        if self.source_urls:
            self.context = await self.get_context_by_urls(self.source_urls)
        else:
            self.context = await self.get_context_by_search(self.query)

        time.sleep(2)

    async def write_report(self, existing_headers: list = []):
        """
        Writes the report based on research conducted

        Returns:
            str: The report
        """

        await stream_output("logs", f"✍️ Writing summary for research task: {self.query}...", self.websocket)

        if self.report_type == "custom_report":
            self.role = self.cfg.agent_role if self.cfg.agent_role else self.role
        elif self.report_type == "subtopic_report":
            report = await generate_report(
                query=self.query,
                context=self.context,
                agent_role_prompt=self.role,
                report_type=self.report_type,
                websocket=self.websocket,
                cfg=self.cfg,
                main_topic=self.parent_query,
                existing_headers=existing_headers
            )
        else:
            report = await generate_report(
                query=self.query,
                context=self.context,
                agent_role_prompt=self.role,
                report_type=self.report_type,
                websocket=self.websocket,
                cfg=self.cfg
            )

        return report

    async def get_context_by_urls(self, urls):
        """
            Scrapes and compresses the context from the given urls
        """
        new_search_urls = await self.get_new_urls(urls)
        await stream_output("logs",
                            f"🧠 I will conduct my research based on the following urls: {new_search_urls}...",
                            self.websocket)
        scraped_sites = scrape_urls(new_search_urls, self.cfg)
        return await self.get_similar_content_by_query(self.query, scraped_sites)

    async def get_context_by_search(self, query):
        """
           Generates the context for the research task by searching the query and scraping the results
        Returns:
            context: List of context
        """
        context = []
        # Generate Sub-Queries including original query
        sub_queries = await get_sub_queries(query, self.role, self.cfg, self.parent_query, self.report_type)

        # If this is not part of a sub researcher, add original query to research for better results
        if self.report_type != "subtopic_report":
            sub_queries.append(query)

        await stream_output("logs",
                            f"🧠 I will conduct my research based on the following queries: {sub_queries}...",
                            self.websocket)

        # Using asyncio.gather to process the sub_queries asynchronously
        context = await asyncio.gather(*[self.process_sub_query(sub_query) for sub_query in sub_queries])
        return context

    async def process_sub_query(self, sub_query: str):
        """Takes in a sub query and scrapes urls based on it and gathers context.

        Args:
            sub_query (str): The sub-query generated from the original query

        Returns:
            str: The context gathered from search
        """
        await stream_output("logs", f"\n🔎 Running research for '{sub_query}'...", self.websocket)

        scraped_sites = await self.scrape_sites_by_query(sub_query)
        content = await self.get_similar_content_by_query(sub_query, scraped_sites)

        if content:
            await stream_output("logs", f"📃 {content}", self.websocket)
        else:
            await stream_output("logs", f"🤷 No content found for '{sub_query}'...", self.websocket)
        return content

    async def get_new_urls(self, url_set_input):
        """ Gets the new urls from the given url set.
        Args: url_set_input (set[str]): The url set to get the new urls from
        Returns: list[str]: The new urls from the given url set
        """

        new_urls = []
        for url in url_set_input:
            if url not in self.visited_urls:
                await stream_output("logs", f"✅ Adding source url to research: {url}\n", self.websocket)

                self.visited_urls.add(url)
                new_urls.append(url)

        return new_urls

    async def scrape_sites_by_query(self, sub_query):
        """
        Runs a sub-query
        Args:
            sub_query:

        Returns:
            Summary
        """
        # Get Urls
        retriever = self.retriever(sub_query)
        search_results = retriever.search(
            max_results=self.cfg.max_search_results_per_query)
        new_search_urls = await self.get_new_urls([url.get("href") for url in search_results])

        # Scrape Urls
        # await stream_output("logs", f"📝Scraping urls {new_search_urls}...\n", self.websocket)
        await stream_output("logs", f"🤔 Researching for relevant information...\n", self.websocket)
        scraped_content_results = scrape_urls(new_search_urls, self.cfg)
        return scraped_content_results

    async def get_similar_content_by_query(self, query, pages):
        await stream_output("logs", f"📝 Getting relevant content based on query: {query}...", self.websocket)
        # Summarize Raw Data
        context_compressor = ContextCompressor(
            documents=pages, embeddings=self.memory.get_embeddings())
        # Run Tasks
        return context_compressor.get_context(query, max_results=8)

    ########################################################################################

    # DETAILED REPORT

    async def write_introduction(self):
        # Construct Report Introduction from main topic research
        introduction = await get_report_introduction(self.query, self.context, self.role, self.cfg, self.websocket)

        return introduction

    async def get_subtopics(self):
        """
        This async function generates subtopics based on user input and other parameters.

        Returns:
          The `get_subtopics` function is returning the `subtopics` that are generated by the
        `construct_subtopics` function.
        """
        await stream_output("logs", f"🤔 Generating subtopics...", self.websocket)

        subtopics = await construct_subtopics(
            task=self.query,
            data=self.context,
            config=self.cfg,
            # This is a list of user provided subtopics
            subtopics=self.subtopics,
        )

        await stream_output("logs", f"📋Subtopics: {subtopics}", self.websocket)

        return subtopics


# File: __init__.py

from .embeddings import Memory


# File: embeddings.py

from langchain_community.vectorstores import FAISS
import os


class Memory:
    def __init__(self, embedding_provider, **kwargs):

        _embeddings = None
        match embedding_provider:
            case "ollama":
                from langchain.embeddings import OllamaEmbeddings
                _embeddings = OllamaEmbeddings(model="llama2")
            case "openai":
                from langchain_openai import OpenAIEmbeddings
                _embeddings = OpenAIEmbeddings()
            case "azureopenai":
                from langchain_openai import AzureOpenAIEmbeddings
                _embeddings = AzureOpenAIEmbeddings(deployment=os.environ["AZURE_EMBEDDING_MODEL"], chunk_size=16)
            case "huggingface":
                from langchain.embeddings import HuggingFaceEmbeddings
                _embeddings = HuggingFaceEmbeddings()

            case _:
                raise Exception("Embedding provider not found.")

        self._embeddings = _embeddings

    def get_embeddings(self):
        return self._embeddings


# File: __init__.py



# File: tavily_news.py

# Tavily API Retriever

# libraries
import os
from tavily import TavilyClient


class TavilyNews():
    """
    Tavily News API Retriever
    Retrieve news articles from the Tavily News API
    """
    def __init__(self, query):
        """
        Initializes the TavilySearch object
        Args:
            query:
        """
        self.query = query
        self.api_key = self.get_api_key()
        self.client = TavilyClient(self.api_key)

    def get_api_key(self):
        """
        Gets the Tavily API key
        Returns:

        """
        # Get the API key
        try:
            api_key = os.environ["TAVILY_API_KEY"]
        except:
            raise Exception("Tavily API key not found. Please set the TAVILY_API_KEY environment variable. "
                            "You can get a key at https://app.tavily.com")
        return api_key

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        # Search the query
        results = self.client.search(self.query, search_depth="advanced", topic="news", max_results=max_results)
        # Return the results
        search_response = [{"href": obj["url"], "body": obj["content"]} for obj in results.get("results", [])]
        return search_response


# File: __init__.py



# File: duckduckgo.py

from itertools import islice
from duckduckgo_search import DDGS


class Duckduckgo:
    """
    Duckduckgo API Retriever
    """
    def __init__(self, query):
        self.ddg = DDGS()
        self.query = query

    def search(self, max_results=5):
        """
        Performs the search
        :param query:
        :param max_results:
        :return:
        """
        ddgs_gen = self.ddg.text(self.query, region='wt-wt', max_results=max_results)
        return ddgs_gen

# File: __init__.py



# File: google.py

# Tavily API Retriever

# libraries
import os
import requests
import json
from tavily import TavilyClient


class GoogleSearch:
    """
    Tavily API Retriever
    """
    def __init__(self, query):
        """
        Initializes the TavilySearch object
        Args:
            query:
        """
        self.query = query
        self.api_key = self.get_api_key() #GOOGLE_API_KEY
        self.cx_key = self.get_cx_key() #GOOGLE_CX_KEY
        self.client = TavilyClient(self.api_key)

    def get_api_key(self):
        """
        Gets the Tavily API key
        Returns:

        """
        # Get the API key
        try:
            api_key = os.environ["GOOGLE_API_KEY"]
        except:
            raise Exception("Google API key not found. Please set the GOOGLE_API_KEY environment variable. "
                            "You can get a key at https://developers.google.com/custom-search/v1/overview")
        return api_key

    def get_cx_key(self):
        """
        Gets the Tavily API key
        Returns:

        """
        # Get the API key
        try:
            api_key = os.environ["GOOGLE_CX_KEY"]
        except:
            raise Exception("Google CX key not found. Please set the GOOGLE_CX_KEY environment variable. "
                            "You can get a key at https://developers.google.com/custom-search/v1/overview")
        return api_key

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        """Useful for general internet search queries using the Google API."""
        print("Searching with query {0}...".format(self.query))
        url = f"https://www.googleapis.com/customsearch/v1?key={self.api_key}&cx={self.cx_key}&q={self.query}&start=1"
        resp = requests.get(url)

        if resp is None:
            return
        try:
            search_results = json.loads(resp.text)
        except Exception:
            return
        if search_results is None:
            return

        results = search_results.get("items", [])
        search_results = []

        # Normalizing results to match the format of the other search APIs
        for result in results:
            # skip youtube results
            if "youtube.com" in result["link"]:
                continue
            search_result = {
                "title": result["title"],
                "href": result["link"],
                "body": result["snippet"],
            }
            search_results.append(search_result)

        return search_results


# File: bing.py

# Bing Search Retriever

# libraries
import os
import requests
import json


class BingSearch():
    """
    Bing Search Retriever
    """
    def __init__(self, query):
        """
        Initializes the BingSearch object
        Args:
            query:
        """
        self.query = query
        self.api_key = self.get_api_key()

    def get_api_key(self):
        """
        Gets the Bing API key
        Returns:

        """
        try:
            api_key = os.environ["BING_API_KEY"]
        except:
            raise Exception("Bing API key not found. Please set the BING_API_KEY environment variable.")
        return api_key

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        print("Searching with query {0}...".format(self.query))
        """Useful for general internet search queries using the Bing API."""


        # Search the query
        url = "https://api.bing.microsoft.com/v7.0/search"

        headers = {
        'Ocp-Apim-Subscription-Key': self.api_key,
        'Content-Type': 'application/json'
        }
        params = {
            "responseFilter" : "Webpages",
            "q": self.query,
            "count": max_results,
            "setLang": "en-GB",
            "textDecorations": False,
            "textFormat": "HTML",
            "safeSearch": "Strict"
        }
        
        resp = requests.get(url, headers=headers, params=params)

        # Preprocess the results
        if resp is None:
            return
        try:
            search_results = json.loads(resp.text)
        except Exception:
            return
        if search_results is None:
            return

        results = search_results["webPages"]["value"]
        search_results = []

        # Normalize the results to match the format of the other search APIs
        for result in results:
            # skip youtube results
            if "youtube.com" in result["url"]:
                continue
            search_result = {
                "title": result["name"],
                "href": result["url"],
                "body": result["snippet"],
            }
            search_results.append(search_result)

        return search_results


# File: __init__.py



# File: searx.py

# Tavily API Retriever

# libraries
import os
from tavily import TavilyClient
from langchain_community.utilities import SearxSearchWrapper


class SearxSearch():
    """
    Tavily API Retriever
    """
    def __init__(self, query):
        """
        Initializes the TavilySearch object
        Args:
            query:
        """
        self.query = query
        self.api_key = self.get_api_key()
        self.client = TavilyClient(self.api_key)

    def get_api_key(self):
        """
        Gets the Tavily API key
        Returns:

        """
        # Get the API key
        try:
            api_key = os.environ["SEARX_URL"]
        except:
            raise Exception("Searx URL key not found. Please set the SEARX_URL environment variable. "
                            "You can get your key from https://searx.space/")
        return api_key

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        searx = SearxSearchWrapper(searx_host=os.environ["SEARX_URL"])
        results = searx.results(self.query, max_results)
        # Normalizing results to match the format of the other search APIs
        search_response = [{"href": obj["link"], "body": obj["snippet"]} for obj in results]
        return search_response


# File: __init__.py



# File: __init__.py

from .tavily_search.tavily_search import TavilySearch
from .tavily_news.tavily_news import TavilyNews
from .duckduckgo.duckduckgo import Duckduckgo
from .google.google import GoogleSearch
from .serper.serper import SerperSearch
from .serpapi.serpapi import SerpApiSearch
from .searx.searx import SearxSearch
from .bing.bing import BingSearch

__all__ = [
    "TavilySearch",
    "TavilyNews",
    "Duckduckgo",
    "SerperSearch",
    "SerpApiSearch",
    "GoogleSearch",
    "SearxSearch",
    "BingSearch"
]


# File: __init__.py



# File: tavily_search.py

# Tavily API Retriever

# libraries
import os
from tavily import TavilyClient
from duckduckgo_search import DDGS


class TavilySearch():
    """
    Tavily API Retriever
    """
    def __init__(self, query):
        """
        Initializes the TavilySearch object
        Args:
            query:
        """
        self.query = query
        self.api_key = self.get_api_key()
        self.client = TavilyClient(self.api_key)

    def get_api_key(self):
        """
        Gets the Tavily API key
        Returns:

        """
        # Get the API key
        try:
            api_key = os.environ["TAVILY_API_KEY"]
        except:
            raise Exception("Tavily API key not found. Please set the TAVILY_API_KEY environment variable. "
                            "You can get a key at https://app.tavily.com")
        return api_key

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        try:
            # Search the query
            results = self.client.search(self.query, search_depth="advanced", max_results=max_results)
            # Return the results
            search_response = [{"href": obj["url"], "body": obj["content"]} for obj in results.get("results", [])]
        except Exception as e: # Fallback in case overload on Tavily Search API
            print(f"Error: {e}")
            ddg = DDGS()
            search_response = ddg.text(self.query, region='wt-wt', max_results=max_results)
        return search_response


# File: serpapi.py

# SerpApi Retriever

# libraries
import os
import requests
import json


class SerpApiSearch():
    """
    SerpApi Retriever
    """
    def __init__(self, query):
        """
        Initializes the SerpApiSearch object
        Args:
            query:
        """
        raise NotImplementedError("SerpApiSearch is not fully implemented yet.")
        self.query = query
        self.api_key = self.get_api_key()

    def get_api_key(self):
        """
        Gets the SerpApi API key
        Returns:

        """
        try:
            api_key = os.environ["SERPAPI_API_KEY"]
        except:
            raise Exception("SerpApi API key not found. Please set the SERPAPI_API_KEY environment variable. "
                            "You can get a key at https://serpapi.com/")
        return api_key

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        print("Searching with query {0}...".format(self.query))
        """Useful for general internet search queries using SerpApi."""


        # Perform the search
        # TODO: query needs to be url encoded, so the code won't work as is.
        # Encoding should look something like this (but this is untested):
        # url_encoded_query = self.query.replace(" ", "+")
        url = "https://serpapi.com/search.json?engine=google&q=" + self.query + "&api_key=" + self.api_key
        resp = requests.request("GET", url)

        # Preprocess the results
        if resp is None:
            return
        try:
            search_results = json.loads(resp.text)
        except Exception:
            return
        if search_results is None:
            return

        results = search_results["organic_results"]
        search_results = []

        # Normalize the results to match the format of the other search APIs
        for result in results:
            # skip youtube results
            if "youtube.com" in result["link"]:
                continue
            search_result = {
                "title": result["title"],
                "href": result["link"],
                "body": result["snippet"],
            }
            search_results.append(search_result)

        return search_results


# File: __init__.py



# File: serper.py

# Google Serper Retriever

# libraries
import os
import requests
import json


class SerperSearch():
    """
    Google Serper Retriever
    """
    def __init__(self, query):
        """
        Initializes the SerperSearch object
        Args:
            query:
        """
        self.query = query
        self.api_key = self.get_api_key()

    def get_api_key(self):
        """
        Gets the Serper API key
        Returns:

        """
        try:
            api_key = os.environ["SERPER_API_KEY"]
        except:
            raise Exception("Serper API key not found. Please set the SERPER_API_KEY environment variable. "
                            "You can get a key at https://serper.dev/")
        return api_key

    def search(self, max_results=7):
        """
        Searches the query
        Returns:

        """
        print("Searching with query {0}...".format(self.query))
        """Useful for general internet search queries using the Serp API."""


        # Search the query (see https://serper.dev/playground for the format)
        url = "https://google.serper.dev/search"

        headers = {
        'X-API-KEY': self.api_key,
        'Content-Type': 'application/json'
        }
        data = json.dumps({"q": self.query})

        resp = requests.request("POST", url, headers=headers, data=data)

        # Preprocess the results
        if resp is None:
            return
        try:
            search_results = json.loads(resp.text)
        except Exception:
            return
        if search_results is None:
            return

        results = search_results["organic"]
        search_results = []

        # Normalize the results to match the format of the other search APIs
        for result in results:
            # skip youtube results
            if "youtube.com" in result["link"]:
                continue
            search_result = {
                "title": result["title"],
                "href": result["link"],
                "body": result["snippet"],
            }
            search_results.append(search_result)

        return search_results


# File: __init__.py



# File: __init__.py



# File: pymupdf.py

from langchain_community.document_loaders import PyMuPDFLoader


class PyMuPDFScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def scrape(self) -> str:
        """
        The `scrape` function uses PyMuPDFLoader to load a document from a given link and returns it as
        a string.
        
        Returns:
          The `scrape` method is returning a string representation of the `doc` object, which is loaded
        using PyMuPDFLoader from the provided link.
        """
        loader = PyMuPDFLoader(self.link)
        doc = loader.load()
        return str(doc)


# File: scraper.py

from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial

import requests

from gpt_researcher.scraper import (
    ArxivScraper,
    BeautifulSoupScraper,
    NewspaperScraper,
    PyMuPDFScraper,
    WebBaseLoaderScraper,
)


class Scraper:
    """
    Scraper class to extract the content from the links
    """

    def __init__(self, urls, user_agent, scraper):
        """
        Initialize the Scraper class.
        Args:
            urls:
        """
        self.urls = urls
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.scraper = scraper

    def run(self):
        """
        Extracts the content from the links
        """
        partial_extract = partial(self.extract_data_from_link, session=self.session)
        with ThreadPoolExecutor(max_workers=20) as executor:
            contents = executor.map(partial_extract, self.urls)
        res = [content for content in contents if content["raw_content"] is not None]
        return res

    def extract_data_from_link(self, link, session):
        """
        Extracts the data from the link
        """
        content = ""
        try:
            Scraper = self.get_scraper(link)
            scraper = Scraper(link, session)
            content = scraper.scrape()

            if len(content) < 100:
                return {"url": link, "raw_content": None}
            return {"url": link, "raw_content": content}
        except Exception as e:
            return {"url": link, "raw_content": None}

    def get_scraper(self, link):
        """
        The function `get_scraper` determines the appropriate scraper class based on the provided link
        or a default scraper if none matches.

        Args:
          link: The `get_scraper` method takes a `link` parameter which is a URL link to a webpage or a
        PDF file. Based on the type of content the link points to, the method determines the appropriate
        scraper class to use for extracting data from that content.

        Returns:
          The `get_scraper` method returns the scraper class based on the provided link. The method
        checks the link to determine the appropriate scraper class to use based on predefined mappings
        in the `SCRAPER_CLASSES` dictionary. If the link ends with ".pdf", it selects the
        `PyMuPDFScraper` class. If the link contains "arxiv.org", it selects the `ArxivScraper
        """

        SCRAPER_CLASSES = {
            "pdf": PyMuPDFScraper,
            "arxiv": ArxivScraper,
            "newspaper": NewspaperScraper,
            "bs": BeautifulSoupScraper,
            "web_base_loader": WebBaseLoaderScraper,
        }

        scraper_key = None

        if link.endswith(".pdf"):
            scraper_key = "pdf"
        elif "arxiv.org" in link:
            scraper_key = "arxiv"
        else:
            scraper_key = self.scraper

        scraper_class = SCRAPER_CLASSES.get(scraper_key)
        if scraper_class is None:
            raise Exception("Scraper not found.")

        return scraper_class


# File: __init__.py



# File: beautiful_soup.py

from bs4 import BeautifulSoup


class BeautifulSoupScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def scrape(self):
        """
        This function scrapes content from a webpage by making a GET request, parsing the HTML using
        BeautifulSoup, and extracting script and style elements before returning the cleaned content.
        
        Returns:
          The `scrape` method is returning the cleaned and extracted content from the webpage specified
        by the `self.link` attribute. The method fetches the webpage content, removes script and style
        tags, extracts the text content, and returns the cleaned content as a string. If any exception
        occurs during the process, an error message is printed and an empty string is returned.
        """
        try:
            response = self.session.get(self.link, timeout=4)
            soup = BeautifulSoup(
                response.content, "lxml", from_encoding=response.encoding
            )

            for script_or_style in soup(["script", "style"]):
                script_or_style.extract()

            raw_content = self.get_content_from_url(soup)
            lines = (line.strip() for line in raw_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            content = "\n".join(chunk for chunk in chunks if chunk)
            return content

        except Exception as e:
            print("Error! : " + str(e))
            return ""
        
    def get_content_from_url(self, soup):
        """Get the text from the soup

        Args:
            soup (BeautifulSoup): The soup to get the text from

        Returns:
            str: The text from the soup
        """
        text = ""
        tags = ["p", "h1", "h2", "h3", "h4", "h5"]
        for element in soup.find_all(tags):  # Find all the <p> elements
            text += element.text + "\n"
        return text


# File: __init__.py


from .beautiful_soup.beautiful_soup import BeautifulSoupScraper
from .newspaper.newspaper import NewspaperScraper
from .web_base_loader.web_base_loader import WebBaseLoaderScraper
from .arxiv.arxiv import ArxivScraper
from .pymupdf.pymupdf import PyMuPDFScraper

__all__ = [
    "BeautifulSoupScraper",
    "NewspaperScraper",
    "WebBaseLoaderScraper",
    "ArxivScraper",
    "PyMuPDFScraper"
]

# File: web_base_loader.py

from langchain_community.document_loaders import WebBaseLoader


class WebBaseLoaderScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def scrape(self) -> str:
        """
        This Python function scrapes content from a webpage using a WebBaseLoader object and returns the
        concatenated page content.
        
        Returns:
          The `scrape` method is returning a string variable named `content` which contains the
        concatenated page content from the documents loaded by the `WebBaseLoader`. If an exception
        occurs during the process, an error message is printed and an empty string is returned.
        """
        try:
            loader = WebBaseLoader(self.link)
            loader.requests_kwargs = {"verify": False}
            docs = loader.load()
            content = ""

            for doc in docs:
                content += doc.page_content

            return content

        except Exception as e:
            print("Error! : " + str(e))
            return ""


# File: __init__.py



# File: newspaper.py

from newspaper import Article


class NewspaperScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def scrape(self) -> str:
        """
        This Python function scrapes an article from a given link, extracts the title and text content,
        and returns them concatenated with a colon.
        
        Returns:
          The `scrape` method returns a string that contains the title of the article followed by a
        colon and the text of the article. If the title or text is not present, an empty string is
        returned. If an exception occurs during the scraping process, an error message is printed and an
        empty string is returned.
        """
        try:
            article = Article(
                self.link,
                language="en",
                memoize_articles=False,
                fetch_images=False,
            )
            article.download()
            article.parse()

            title = article.title
            text = article.text

            # If title, summary are not present then return None
            if not (title and text):
                return ""

            return f"{title} : {text}"

        except Exception as e:
            print("Error! : " + str(e))
            return ""


# File: __init__.py



# File: arxiv.py

from langchain_community.retrievers import ArxivRetriever


class ArxivScraper:

    def __init__(self, link, session=None):
        self.link = link
        self.session = session

    def scrape(self):
        """
        The function scrapes relevant documents from Arxiv based on a given link and returns the content
        of the first document.
        
        Returns:
          The code is returning the page content of the first document retrieved by the ArxivRetriever
        for a given query extracted from the link.
        """
        query = self.link.split("/")[-1]
        retriever = ArxivRetriever(load_max_docs=2, doc_content_chars_max=None)
        docs = retriever.get_relevant_documents(query=query)
        return docs[0].page_content


# File: __init__.py



# File: validators.py

from typing import List

from pydantic import BaseModel, Field

class Subtopic(BaseModel):
    task: str = Field(description="Task name", min_length=1)

class Subtopics(BaseModel):
    subtopics: List[Subtopic] = []


# File: __init__.py



# File: llm.py

# libraries
from __future__ import annotations

import json
import logging
from typing import Optional

from colorama import Fore, Style
from fastapi import WebSocket
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from gpt_researcher.master.prompts import auto_agent_instructions, generate_subtopics_prompt

from .validators import Subtopics


def get_provider(llm_provider):
    match llm_provider:
        case "openai":
            from ..llm_provider import OpenAIProvider
            llm_provider = OpenAIProvider
        case "azureopenai":
            from ..llm_provider import AzureOpenAIProvider
            llm_provider = AzureOpenAIProvider
        case "google":
            from ..llm_provider import GoogleProvider
            llm_provider = GoogleProvider

        case _:
            raise Exception("LLM provider not found.")

    return llm_provider


async def create_chat_completion(
        messages: list,  # type: ignore
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        llm_provider: Optional[str] = None,
        stream: Optional[bool] = False,
        websocket: WebSocket | None = None,
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.9.
        max_tokens (int, optional): The max tokens to use. Defaults to None.
        stream (bool, optional): Whether to stream the response. Defaults to False.
        llm_provider (str, optional): The LLM Provider to use.
        webocket (WebSocket): The websocket used in the currect request
    Returns:
        str: The response from the chat completion
    """

    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 8001:
        raise ValueError(
            f"Max tokens cannot be more than 8001, but got {max_tokens}")

    # Get the provider from supported providers
    ProviderClass = get_provider(llm_provider)
    provider = ProviderClass(
        model,
        temperature,
        max_tokens
    )

    # create response
    for _ in range(10):  # maximum of 10 attempts
        response = await provider.get_chat_response(
            messages, stream, websocket
        )
        return response

    logging.error("Failed to get response from OpenAI API")
    raise RuntimeError("Failed to get response from OpenAI API")


def choose_agent(smart_llm_model: str, llm_provider: str, task: str) -> dict:
    """Determines what server should be used
    Args:
        task (str): The research question the user asked
        smart_llm_model (str): the llm model to be used
        llm_provider (str): the llm provider used
    Returns:
        server - The server that will be used
        agent_role_prompt (str): The prompt for the server
    """
    try:
        response = create_chat_completion(
            model=smart_llm_model,
            messages=[
                {"role": "system", "content": f"{auto_agent_instructions()}"},
                {"role": "user", "content": f"task: {task}"}],
            temperature=0,
            llm_provider=llm_provider
        )
        agent_dict = json.loads(response)
        print(f"Agent: {agent_dict.get('server')}")
        return agent_dict
    except Exception as e:
        print(f"{Fore.RED}Error in choose_agent: {e}{Style.RESET_ALL}")
        return {"server": "Default Agent",
                "agent_role_prompt": "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."}


async def construct_subtopics(task: str, data: str, config, subtopics: list = []) -> list:
    try:
        parser = PydanticOutputParser(pydantic_object=Subtopics)

        prompt = PromptTemplate(
            template=generate_subtopics_prompt(),
            input_variables=["task", "data", "subtopics", "max_subtopics"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()},
        )

        print(f"\n🤖 Calling {config.smart_llm_model}...\n")

        if config.llm_provider == "openai":
            model = ChatOpenAI(model=config.smart_llm_model)
        elif config.llm_provider == "azureopenai":
            from langchain_openai import AzureChatOpenAI
            model = AzureChatOpenAI(model=config.smart_llm_model)
        else:
            return []

        chain = prompt | model | parser

        output = chain.invoke({
            "task": task,
            "data": data,
            "subtopics": subtopics,
            "max_subtopics": config.max_subtopics
        })

        return output

    except Exception as e:
        print("Exception in parsing subtopics : ", e)
        return subtopics

# File: enum.py

from enum import Enum
class ReportType(Enum):
    ResearchReport = 'research_report'
    ResourceReport = 'resource_report'
    OutlineReport = 'outline_report'
    CustomReport = 'custom_report'
    DetailedReport = 'detailed_report'
    SubtopicReport = 'subtopic_report'
