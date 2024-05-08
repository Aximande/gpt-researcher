

# File: pydoc-markdown.yml

loaders:
   - type: python
     search_path: [../docs]
processors:
  - type: filter
    skip_empty_modules: true
  - type: smart
  - type: crossref
renderer:
  type: docusaurus
  docs_base_path: docs
  relative_output_path: reference
  relative_sidebar_path: sidebar.json
  sidebar_top_level_label: Reference
  markdown:
    escape_html_in_docstring: false


# File: agent_frameworks.md

# Multi Agent Frameworks

We are strong advocates for the future of AI agents, envisioning a world where autonomous agents communicate and collaborate as a cohesive team to undertake and complete complex tasks.

We hold the belief that research is a pivotal element in successfully tackling these complex tasks, ensuring superior outcomes.

Consider the scenario of developing a coding agent responsible for coding tasks using the latest API documentation and best practices. It would be wise to integrate an agent specializing in research to curate the most recent and relevant documentation, before crafting a technical design that would subsequently be handed off to the coding assistant tasked with generating the code. This approach is applicable across various sectors, including finance, business analysis, healthcare, marketing, and legal, among others.

One multi-agent framework that we're excited about is [LangGraph](https://python.langchain.com/docs/langgraph/), built by the team at [Langchain](https://www.langchain.com/).
LangGraph is a Python library for building stateful, multi-actor applications with LLMs. It extends the [LangChain Expression Language](https://python.langchain.com/docs/expression_language/) with the ability to coordinate multiple chains (or actors) across multiple steps of computation.

What's great about LangGraph is that it follows a DAG architecture, enabling each specialized agent to communicate with one another, and subsequently trigger actions among other agents within the graph. 

We've added an example for leveraging [GPT Researcher with LangGraph](https://github.com/assafelovic/gpt-researcher/tree/master/multi_agents) which can be found in `/multi_agents`.

The example demonstrates a generic use case for an editorial agent team that works together to complete a research report on a given task.

## The Multi Agent Team
The research team is made up of 7 AI agents:
- **Chief Editor** - Oversees the research process and manages the team. This is the "master" agent that coordinates the other agents using Langgraph.
- **Researcher** (gpt-researcher) - A specialized autonomous agent that conducts in depth research on a given topic.
- **Editor** - Responsible for planning the research outline and structure.
- **Reviewer** - Validates the correctness of the research results given a set of criteria.
- **Revisor** - Revises the research results based on the feedback from the reviewer.
- **Writer** - Responsible for compiling and writing the final report.
- **Publisher** - Responsible for publishing the final report in various formats.

## How it works
Generally, the process is based on the following stages: 
1. Planning stage
2. Data collection and analysis
3. Writing and submission
4. Review and revision
5. Publication

### Architecture
<div align="center">
<img align="center" height="600" src="https://cowriter-images.s3.amazonaws.com/gptr-langgraph-architecture.png"></img>
</div>
<br clear="all"/>

### Steps
More specifically (as seen in the architecture diagram) the process is as follows:
- Browser (gpt-researcher) - Browses the internet for initial research based on the given research task.
- Editor - Plans the report outline and structure based on the initial research.
- For each outline topic (in parallel):
  - Researcher (gpt-researcher) - Runs an in depth research on the subtopics and writes a draft.
  - Reviewer - Validates the correctness of the draft given a set of criteria and provides feedback.
  - Revisor - Revises the draft until it is satisfactory based on the reviewer feedback.
- Writer - Compiles and writes the final report including an introduction, conclusion and references section from the given research findings.
- Publisher - Publishes the final report to multi formats such as PDF, Docx, Markdown, etc.

## How to run
1. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the application:
    ```bash
    python main.py
    ```

## Usage
To change the research query and customize the report, edit the `task.json` file in the main directory.


# File: example.md

# Agent Example

If you're interested in using GPT Researcher as a standalone agent, you can easily import it into any existing Python project. Below, is an example of calling the agent to generate a research report:

```python
from gpt_researcher import GPTResearcher
import asyncio

# It is best to define global constants at the top of your script
QUERY = "What happened in the latest burning man floods?"
REPORT_TYPE = "research_report"

async def fetch_report(query, report_type):
    """
    Fetch a research report based on the provided query and report type.
    """
    researcher = GPTResearcher(query=query, report_type=report_type, config_path=None)
    await researcher.conduct_research()
    report = await researcher.write_report()
    return report

async def generate_research_report():
    """
    This is a sample script that executes an async main function to run a research report.
    """
    report = await fetch_report(QUERY, REPORT_TYPE)
    print(report)

if __name__ == "__main__":
    asyncio.run(generate_research_report())
```

You can further enhance this example to use the returned report as context for generating valuable content such as news article, marketing content, email templates, newsletters, etc.

You can also use GPT Researcher to gather information about code documentation, business analysis, financial information and more. All of which can be used to complete much more complex tasks that require factual and high quality realtime information.


# File: troubleshooting.md

# Troubleshooting
We're constantly working to provide a more stable version. If you're running into any issues, please first check out the resolved issues or ask us via our [Discord community](https://discord.gg/2pFkc83fRq).

**model: gpt-4 does not exist**
This relates to not having permission to use gpt-4 yet. Based on OpenAI, it will be [widely available for all by end of July](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4).

**cannot load library 'gobject-2.0-0'**

The issue relates to the library WeasyPrint (which is used to generate PDFs from the research report). Please follow this guide to resolve it: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html

**Error processing the url**

We're using [Selenium](https://www.selenium.dev) for site scraping. Some sites fail to be scraped. In these cases, restart and try running again.


**Chrome version issues**

Many users have an issue with their chromedriver because the latest chrome browser version doesn't have a compatible chrome driver yet.

To downgrade your Chrome web browser using [slimjet](https://www.slimjet.com/chrome/google-chrome-old-version.php), follow these steps. First, visit the website and scroll down to find the list of available older Chrome versions. Choose the version you wish to install
making sure it's compatible with your operating system.
Once you've selected the desired version, click on the corresponding link to download the installer. Before proceeding with the installation, it's crucial to uninstall your current version of Chrome to avoid conflicts.

It's important to check if the version you downgrade to, has a chromedriver available in the official [chrome driver website](https://chromedriver.chromium.org/downloads)

**If none of the above work, you can [try out our hosted beta](https://app.tavily.com)**

# File: pip-package.md

# PIP Package

🌟 **Exciting News!** Now, you can integrate `gpt-researcher` with your apps seamlessly!

## Steps to Install GPT Researcher 🛠️

Follow these easy steps to get started:

0. **Pre-requisite**: Ensure Python 3.10+ is installed on your machine 💻
1. **Install gpt-researcher**: Grab the official package from [PyPi](https://pypi.org/project/gpt-researcher/).
```bash
pip install gpt-researcher
```
2. **Environment Variables:** Create a .env file with your OpenAI API key or simply export it
```bash
export OPENAI_API_KEY={Your OpenAI API Key here}
```
```bash
export TAVILY_API_KEY={Your Tavily API Key here}
```
3. **Start using GPT Researcher in your own codebase**

## Example Usage 📝
```python
from gpt_researcher import GPTResearcher
import asyncio


async def get_report(query: str, report_type: str) -> str:
    researcher = GPTResearcher(query, report_type)
    report = await researcher.run()
    return report

if __name__ == "__main__":
    query = "what team may win the NBA finals?"
    report_type = "research_report"

    report = asyncio.run(get_report(query, report_type))
    print(report)
```

## Specific Examples 🌐
### Example 1: Research Report 📚
```python
query = "Latest developments in renewable energy technologies"
report_type = "research_report"
```
### Example 2: Resource Report 📋
```python
query = "List of top AI conferences in 2023"
report_type = "resource_report"
```

### Example 3: Outline Report 📝
```python
query = "Outline for an article on the impact of AI in education"
report_type = "outline_report"
```

## Integration with Web Frameworks 🌍
### FastAPI Example
```python
from fastapi import FastAPI
from gpt_researcher import GPTResearcher
import asyncio

app = FastAPI()

@app.get("/report/{report_type}")
async def get_report(report_type: str, query: str):
    researcher = GPTResearcher(query, report_type)
    report = await researcher.run()
    return {"report": report}

# Run the server
# uvicorn main:app --reload
```

### Flask Example
```python
from flask import Flask, request
from gpt_researcher import GPTResearcher
import asyncio

app = Flask(__name__)

@app.route('/report/<report_type>', methods=['GET'])
def get_report(report_type):
    query = request.args.get('query')
    report = asyncio.run(GPTResearcher(query, report_type).run())
    return report

# Run the server
# flask run
```



# File: introduction.md

# Introduction

**[GPT Researcher](https://gptr.dev) is an autonomous agent designed for comprehensive online research on a variety of tasks.** 

The agent can produce detailed, factual and unbiased research reports, with customization options for focusing on relevant resources, outlines, and lessons. Inspired by the recent [Plan-and-Solve](https://arxiv.org/abs/2305.04091) and [RAG](https://arxiv.org/abs/2005.11401) papers, GPT Researcher addresses issues of speed, determinism and reliability, offering a more stable performance and increased speed through parallelized agent work, as opposed to synchronous operations.

## Why GPT Researcher?

- To form objective conclusions for manual research tasks can take time, sometimes weeks to find the right resources and information.
- Current LLMs are trained on past and outdated information, with heavy risks of hallucinations, making them almost irrelevant for research tasks.
- Solutions that enable web search (such as ChatGPT + Web Plugin), only consider limited resources and content that in some cases result in superficial conclusions or biased answers.
- Using only a selection of resources can create bias in determining the right conclusions for research questions or tasks. 

## Architecture
The main idea is to run "planner" and "execution" agents, whereas the planner generates questions to research, and the execution agents seek the most related information based on each generated research question. Finally, the planner filters and aggregates all related information and creates a research report. <br /> <br /> 
The agents leverage both gpt3.5-turbo and gpt-4-turbo (128K context) to complete a research task. We optimize for costs using each only when necessary. **The average research task takes around 3 minutes to complete, and costs ~$0.1.**

<div align="center">
<img align="center" height="500" src="https://cowriter-images.s3.amazonaws.com/architecture.png" />
</div>


More specifically:
* Create a domain specific agent based on research query or task.
* Generate a set of research questions that together form an objective opinion on any given task. 
* For each research question, trigger a crawler agent that scrapes online resources for information relevant to the given task.
* For each scraped resources, summarize based on relevant information and keep track of its sources.
* Finally, filter and aggregate all summarized sources and generate a final research report.

## Demo
<iframe height="400" width="700" src="https://github.com/assafelovic/gpt-researcher/assets/13554167/a00c89a6-a295-4dd0-b58d-098a31c40fda" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

## Tutorials
 - [How it Works](https://medium.com/better-programming/how-i-built-an-autonomous-ai-agent-for-online-research-93435a97c6c)
 - [How to Install](https://www.loom.com/share/04ebffb6ed2a4520a27c3e3addcdde20?sid=da1848e8-b1f1-42d1-93c3-5b0b9c3b24ea)
 - [Live Demo](https://www.loom.com/share/6a3385db4e8747a1913dd85a7834846f?sid=a740fd5b-2aa3-457e-8fb7-86976f59f9b8)
 - [Homepage](https://gptr.dev)

## Features
- 📝 Generate research, outlines, resources and lessons reports
- 📜 Can generate long and detailed research reports (over 2K words)
- 🌐 Aggregates over 20 web sources per research to form objective and factual conclusions
- 🖥️ Includes an easy-to-use web interface (HTML/CSS/JS)
- 🔍 Scrapes web sources with javascript support
- 📂 Keeps track and context of visited and used web sources
- 📄 Export research reports to PDF, Word and more...


## Disclaimer

This project, GPT Researcher, is an experimental application and is provided "as-is" without any warranty, express or implied. We are sharing codes for academic purposes under the MIT license. Nothing herein is academic advice, and NOT a recommendation to use in academic or research papers.

Our view on unbiased research claims:
1. The whole point of our scraping system is to reduce incorrect fact. How? The more sites we scrape the less chances of incorrect data. We are scraping 20 per research, the chances that they are all wrong is extremely low.
2. We do not aim to eliminate biases; we aim to reduce it as much as possible. **We are here as a community to figure out the most effective human/llm interactions.**
3. In research, people also tend towards biases as most have already opinions on the topics they research about. This tool scrapes many opinions and will evenly explain diverse views that a biased person would never have read.

**Please note that the use of the GPT-4 language model can be expensive due to its token usage.** By utilizing this project, you acknowledge that you are responsible for monitoring and managing your own token usage and the associated costs. It is highly recommended to check your OpenAI API usage regularly and set up any necessary limits or alerts to prevent unexpected charges.


# File: getting-started.md

# Getting Started
> **Step 0** - Install Python 3.11 or later. [See here](https://www.tutorialsteacher.com/python/install-python) for a step-by-step guide.

> **Step 1** - Download the project and navigate to its directory

```bash
$ git clone https://github.com/assafelovic/gpt-researcher.git
$ cd gpt-researcher
```

> **Step 3** - Set up API keys using two methods: exporting them directly or storing them in a `.env` file.

For Linux/Temporary Windows Setup, use the export method:

```bash
export OPENAI_API_KEY={Your OpenAI API Key here}
export TAVILY_API_KEY={Your Tavily API Key here}
```

For a more permanent setup, create a `.env` file in the current `gpt-researcher` folder and input the keys as follows:

```bash
OPENAI_API_KEY={Your OpenAI API Key here}
TAVILY_API_KEY={Your Tavily API Key here}
```

- **For LLM, we recommend [OpenAI GPT](https://platform.openai.com/docs/guides/gpt)**, but you can use any other LLM model (including open sources) supported by [Langchain Adapter](https://python.langchain.com/docs/guides/adapters/openai), simply change the llm model and provider in config/config.py. 
- **For search engine, we recommend [Tavily Search API](https://app.tavily.com)**, but you can also refer to other search engines of your choice by changing the search provider in config/config.py to `"duckduckgo"`, `"googleAPI"`, `"bing"`, `"googleSerp"`, or `"searx"`. Then add the corresponding env API key as seen in the config.py file.

## Quickstart

> **Step 1** - Install dependencies

```bash
$ pip install -r requirements.txt
```

> **Step 2** - Run the agent with FastAPI

```bash
$ uvicorn main:app --reload
```

> **Step 3** - Go to http://localhost:8000 on any browser and enjoy researching!

## Using Virtual Environment or Poetry
Select either based on your familiarity with each:

### Virtual Environment

#### *Establishing the Virtual Environment with Activate/Deactivate configuration*

Create a virtual environment using the `venv` package with the environment name `<your_name>`, for example, `env`. Execute the following command in the PowerShell/CMD terminal:

```bash
python -m venv env
```

To activate the virtual environment, use the following activation script in PowerShell/CMD terminal:

```bash
.\env\Scripts\activate
```

To deactivate the virtual environment, run the following deactivation script in PowerShell/CMD terminal:

```bash
deactivate
```

#### *Install the dependencies for a Virtual environment*

After activating the `env` environment, install dependencies using the `requirements.txt` file with the following command:

```bash
python -m pip install -r requirements.txt
```

<br />

### Poetry

#### *Establishing the Poetry dependencies and virtual environment with Poetry version `~1.7.1`*

Install project dependencies and simultaneously create a virtual environment for the specified project. By executing this command, Poetry reads the project's "pyproject.toml" file to determine the required dependencies and their versions, ensuring a consistent and isolated development environment. The virtual environment allows for a clean separation of project-specific dependencies, preventing conflicts with system-wide packages and enabling more straightforward dependency management throughout the project's lifecycle.

```bash
poetry install
```

#### *Activate the virtual environment associated with a Poetry project*

By running this command, the user enters a shell session within the isolated environment associated with the project, providing a dedicated space for development and execution. This virtual environment ensures that the project dependencies are encapsulated, avoiding conflicts with system-wide packages. Activating the Poetry shell is essential for seamlessly working on a project, as it ensures that the correct versions of dependencies are used and provides a controlled environment conducive to efficient development and testing.

```bash
poetry shell
```

### *Run the app*
> Launch the FastAPI application agent on a *Virtual Environment or Poetry* setup by executing the following command:
```bash
python -m uvicorn main:app --reload
```
> Visit http://localhost:8000 in any web browser and explore your research!

<br />


## Try it with Docker

> **Step 1** - Install Docker

Follow instructions at https://docs.docker.com/engine/install/

> **Step 2** - Create .env file with your OpenAI Key or simply export it

```bash
$ export OPENAI_API_KEY={Your API Key here}
$ export TAVILY_API_KEY={Your Tavily API Key here}
```

> **Step 3** - Run the application

```bash
$ docker-compose up
```

> **Step 4** - Go to http://localhost:8000 on any browser and enjoy researching!


# File: config.md

# Customization

The config.py enables you to customize GPT Researcher to your specific needs and preferences.

Thanks to our amazing community and contributions, GPT Researcher supports multiple LLMs and Retrievers.
In addition, GPT Researcher can be tailored to various report formats (such as APA), word count, research iterations depth, etc.

GPT Researcher defaults to our recommended suite of integrations: [OpenAI](https://platform.openai.com/docs/overview) for LLM calls and [Tavily API](https://app.tavily.com) for retrieving realtime online information.

As seen below, OpenAI still stands as the superior LLM. We assume it will stay this way for some time, and that prices will only continue to decrease, while performance and speed increase over time.

<div style={{ marginBottom: '10px' }}>
<img align="center" height="350" src="/img/leaderboard.png" />
</div>

It may not come as a surprise that our default search engine is [Tavily](https://app.tavily.com). We're aimed at building our search engine to tailor the exact needs of searching and aggregating for the most factual and unbiased information for research tasks.
We highly recommend using it with GPT Researcher, and more generally with LLM applications that are built with RAG. To learn more about our search API [see here](/docs/tavily-api/introduction)

Here is an example of the default config.py file found in `/gpt_researcher/config/`:

```python
def __init__(self, config_file: str = None):
    self.config_file = config_file
    self.retriever = "tavily"
    self.llm_provider = "openai"
    self.fast_llm_model = "gpt-3.5-turbo-16k"
    self.smart_llm_model = "gpt-4-turbo"
    self.fast_token_limit = 2000
    self.smart_token_limit = 4000
    self.browse_chunk_max_length = 8192
    self.summary_token_limit = 700
    self.temperature = 0.6
    self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)" \
                      " Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"
    self.memory_backend = "local"
    self.total_words = 1000
    self.report_format = "apa"
    self.max_iterations = 1

    self.load_config_file()
```

Please note that you can also include your own external JSON file by adding the path in the `config_file` param.

To learn more about additional LLM support you can check out the [Langchain Adapter](https://python.langchain.com/docs/guides/adapters/openai) and [Langchain supported LLMs](https://python.langchain.com/docs/integrations/llms/) documentation. Simply pass different provider names in the `llm_provider` config param.

You can also change the search engine by modifying the `retriever` param to others such as `duckduckgo`, `googleAPI`, `googleSerp`, `searx` and more. 

Please note that you might need to sign up and obtain an API key for any of the other supported retrievers and LLM providers.

# File: roadmap.md

# Roadmap

We're constantly working on additional features and improvements to our products and services. We're also working on new products and services to help you build better AI applications using [GPT Researcher](https://gptr.dev).

Our vision is to build the #1 autonomous research agent for AI developers and researchers, and we're excited to have you join us on this journey!

The roadmap is prioritized based on the following goals: Performance, Quality, Modularity and Conversational flexibility. The roadmap is public and can be found [here](https://trello.com/b/3O7KBePw/gpt-researcher-roadmap). 

Interested in collaborating or contributing? Check out our [contributing page](/docs/contribute) for more information.

# File: contribute.md

# Contribute

We highly welcome contributions! Please check out [contributing](https://github.com/assafelovic/gpt-researcher/blob/master/CONTRIBUTING.md) if you're interested.

Please check out our [roadmap](https://trello.com/b/3O7KBePw/gpt-researcher-roadmap) page and reach out to us via our [Discord community](https://discord.gg/2pFkc83fRq) if you're interested in joining our mission.

# File: faq.md

# Frequently Asked Questions

### How do I get started?
It really depends on what you're aiming for. 

If you're looking to connect your AI application to the internet with our tailored API, check out the [Tavily API](/docs/tavily-api/introduction) documentation. 
If you're looking to build and deploy our open source autonomous research agent GPT Researcher, please see [GPT Researcher](/docs/gpt-researcher/introduction) documentation.
You can also check out demos and examples for inspiration [here](/docs/examples/examples).
### What is GPT Researcher?
GPT Researcher is a popular open source autonomous research agent that takes care of the tedious task of research for you, by scraping, filtering and aggregating over 20+ web sources per a single research task.

GPT Researcher is built with best practices for leveraging LLMs (prompt engineering, RAG, chains, embeddings, etc), and is optimized for quick and efficient research. It is also fully customizable and can be tailored to your specific needs.

To learn more about GPT Researcher, check out the [documentation page](/docs/gpt-researcher/introduction).
### How much does each research run cost?
A research task using GPT Researcher costs around $0.01 per a single run (for GPT-4 usage). We're constantly optimizing LLM calls to reduce costs and improve performance. 
### How do you ensure the report is factual and accurate?
we do our best to ensure that the information we provide is factual and accurate. We do this by using multiple sources, and by using proprietary AI to score and rank the most relevant and accurate information. We also use proprietary AI to filter out irrelevant information and sources.

Lastly, by using RAG and other techniques, we ensure that the information is relevant to the context of the research task, leading to more accurate generative AI content and reduced hallucinations.

### What is Tavily API?
Tavily search API is a search engine optimized for LLMs, aimed at efficient, quick and persistent search results. Unlike other search APIs such as Serp or Google, Tavily focuses on optimizing search for AI developers and autonomous AI agents. We take care of all the burden in searching, scraping, filtering and extracting the most relevant information from online sources. All in a single API call!

The search API can also be used return answers to questions (for use cases such as multi-agent frameworks like autogen) and can complete comprehensive research tasks in seconds. Moreover, Tavily leverages proprietary financial, code, news, and other data internal data sources to complement online information.

To learn more about Tavily search API, check out the [documentation page](/docs/tavily-api/introduction).

To try the API in action, you can now use our hosted version [here](https://app.tavily.com/chat) or on our [API Playground](https://app.tavily.com/playground).
### How is Tavily different from other search APIs?
Current search APIs such as Google, Serp and Bing retrieve search results based on user query. However, the results are sometimes irrelevant to the goal of the search, and return simple site URLs and snippets of content which are not always relevant. Because of this, any developer would need to then scrape the sites for relevant content, filter irrelevant information, optimize the content to fit LLM context limits, and more. This tasks is a burden and requires skills to get right.

Tavily Search API aggregates over 20+ sites per a single API call, and uses  AI to score, filter and rank the top most relevant sources and content to your task, query or goal. In addition, Tavily allows developers to add custom fields such as context and limit response tokens to enable the optimal search experience for LLMs.
proprietary
Lastly, Tavily indexes and ranks search results based on factors such as trusted sources, content quality, and more. This allows for a more accurate and relevant search experience for AI agents.

Remember: With LLM hallucinations, it's crucial to optimize for RAG with the right context and information.
### What is Tavily API pricing?
Tavily is free to use for up to 1,000 API calls. Check out our [pricing page](https://tavily.com/#pricing) for more information.

At the moment we don't have a pricing model, since we're still in beta and focused on building the best product for our users. We're always open to feedback and suggestions, so please reach out if you have any ideas!
### What are your plans for the future?
We're constantly working on improving our products and services. We're currently working on improving our search API together with design partners, and adding more data sources to our search engine. We're also working on improving our research agent GPT Researcher, and adding more features to it while growing our amazing open source community.

If you're interested in our roadmap or looking to collaborate, check out our [roadmap page](https://trello.com/b/3O7KBePw/gpt-researcher-roadmap). 

Feel free to [contact us](mailto:support@tavily.com) if you have any further questions or suggestions!

# File: welcome.md

# Introduction

Hey there! 👋

We're a team of AI researchers and developers who are passionate about building the next generation of AI assistants. 
Our mission is to empower individuals and organizations with accurate, unbiased, and factual information.

### GPT Researcher
In this digital age, quickly accessing relevant and trustworthy information is more crucial than ever. However, we've learned that none of today's search engines provide a suitable tool that provides factual, explicit and objective answers without the need to continuously click and explore multiple sites for a given research task. 

This is why we've built the trending open source **[GPT Researcher](https://github.com/assafelovic/gpt-researcher)**. GPT Researcher is an autonomous agent that takes care of the tedious task of research for you, by scraping, filtering and aggregating over 20+ web sources per a single research task. 

To learn more about GPT Researcher, check out the [documentation page](/docs/gpt-researcher/introduction).

### Tavily Search API
Building an AI agent that leverages realtime online information is not a simple task. Scraping doesn't scale and requires expertise to refine, current search engine APIs don't provide explicit information to queries but simply potential related articles (which are not always related), and are not very customziable for AI agent needs. This is why we're excited to introduce the first search engine for AI agents - **Tavily Search API**.

Tavily Search API is a search engine optimized for LLMs, aimed at efficient, quick and persistent search results. Unlike other search APIs such as Serp or Google, Tavily focuses on optimizing search for AI developers and autonomous AI agents. We take care of all the burden in searching, scraping, filtering and extracting the most relevant information from online sources. All in a single API call! 

To learn how to build your AI application with Tavily Search API, check out the [documentation page](/docs/tavily-api/introduction).

To try our API in action, you can now use GPT Researcher on our hosted version [here](https://app.tavily.com/chat) or on our [API Playground](https://app.tavily.com/playground).

If you're an AI developer looking to integrate your application with our API or seek increased API limits, **[please reach out!](mailto:support@tavily.com)**


# File: python-sdk.md


# Python SDK
The [Python library](https://github.com/assafelovic/tavily-python) allows for easy interaction with the Tavily API, offering both basic and advanced search functionalities directly from your Python programs. Easily integrate smart search capabilities into your applications, harnessing Tavily's powerful search features.

## Installing 📦

```bash
pip install tavily-python
```
## Usage 🛠️
The search API has two search depth options: **basic** and **advanced**. The basic search is optimized for performance leading to faster response time. The advanced may take longer (around 5-10 seconds response time) but optimizes for quality. 

Look out for the response **content** field. Using the 'advanced' search depth will highly improve the retrieved content to be only the most related content from each site based on a relevance score. The main search method can be used as seen below:
##
```python
from tavily import TavilyClient
tavily = TavilyClient(api_key="YOUR_API_KEY")
# For basic search:
response = tavily.search(query="Should I invest in Apple in 2024?")
# For advanced search:
response = tavily.search(query="Should I invest in Apple in 2024?", search_depth="advanced")
# Get the search results as context to pass an LLM:
context = [{"url": obj["url"], "content": obj["content"]} for obj in response.results]
```
In addition, you can use other powerful methods based on your application use case as seen below:

```python
# You can easily get search result context based on any max tokens straight into your RAG.
# The response is a string of the context within the max_token limit.
tavily.get_search_context(query="What happened in the burning man floods?", search_depth="advanced", max_tokens=1500)

# You can also get a simple answer to a question including relevant sources all with a simple function call:
tavily.qna_search(query="Where does Messi play right now?")
```

## API Methods 📚

### Client
The Client class is the entry point to interacting with the Tavily API. Kickstart your journey by instantiating it with your API key.

### Methods
* **search**(query, **kwargs)
  * The **search_depth** can be either **basic** or **advanced**. The **basic** type offers a quick response, while the **advanced** type gives in-depth, quality results.
  * Additional parameters can be provided as keyword arguments. See below for a list of all available parameters.
  * Returns a JSON with all related response fields.
* **get_search_context**(query, search_depth [Optional], max_tokens [Optional], **kwargs): 
  * Performs a search and returns a string of content and sources within token limit. 
  * Useful for getting only related content from retrieved websites without having to deal with context extraction and token management.
  * Max tokens defaults to 4,000. Search Depth defaults to basic.
  * Returns a string of the most relevant content including sources that fit within the defined token limit.
* **qna_search**(query, search_depth [Optional], **kwargs): 
  * Performs a search and returns a string containing an answer to the original query including relevant sources
  * Optimal to be used as a tool for AI agents.
  * Search depth defaults to advanced for best answer results.
  * Returns a string of a short answer and related sources.

### Keyword Arguments 🖊️

* **search_depth (str)**: The depth of the search. It can be "basic" or "advanced". Default is "basic" for basic_search and "advanced" for advanced_search.

* **max_results (int)**: The number of maximum search results to return. Default is 5.

* **include_images (bool)**: Include a list of query related images in the response. Default is False.

* **include_answer (bool)**: Include a short answer to original query in the search results. Default is False.

* **include_raw_content (bool)**: Include cleaned and parsed HTML of each site search results. Default is False.

* **include_domains (list)**: A list of domains to specifically include in the search results. Default is None, which includes all domains.

* **exclude_domains (list)**: A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains.

### Response Example
To learn more see [REST API](https://app.tavily.com/documentation/api) documentation.
## Error Handling ⚠️

In case of an unsuccessful HTTP request, a HTTPError will be raised.

## License 📝

This project is licensed under the terms of the MIT license.

## Contact 💌

For questions, support, or to learn more, please visit [Tavily](http://tavily.com) 🌍.



# File: rest_api.md

# Rest API

## Overview

Tavily Search is a robust search API tailored specifically for LLM Agents. It seamlessly integrates with diverse data sources to ensure a superior, relevant search experience.

## Features

* **Curated Results**: Provides top-tier results sorted by relevance across multiple sources.
* **Speed & Efficiency**: Optimized for performance, delivering real-time results.
* **Customizable**: Easily refine search results based on various criteria.
* **Easy Integration**: Simple to integrate with existing applications.

## Base URL

`https://api.tavily.com/`


## Endpoints

### POST `/search`

Search for data based on a query.

#### Parameters
- **api_key** (required): Your unique API key.
- **query** (required): The search query string.
- **search_depth** (optional): The depth of the search. It can be **basic** or **advanced**. Default is **basic** for quick results and **advanced** for indepth high quality results but longer response time. Advanced calls equals 2 requests.
- **include_images** (optional): Include a list of query related images in the response. Default is False.
- **include_answer** (optional): Include answers in the search results. Default is False.
- **include_raw_content** (optional): Include raw content in the search results. Default is False.
- **max_results** (optional): The number of maximum search results to return. Default is 5.
- **include_domains** (optional): A list of domains to specifically include in the search results. Default is None, which includes all domains.
- **exclude_domains** (optional): A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains.

## Example Request

```json
{
  "api_key": "your api key",
  "query": "your search query",
  "search_depth": "basic",
  "include_answer": false,
  "include_images": true,
  "include_raw_content": false,
  "max_results": 5,
  "include_domains": [],
  "exclude_domains": []
}
```

### Response

- **answer**: The answer to your search query.
- **query**: Your search query.
- **response_time**: Your search result response time.
- **images**: A list of query related image urls.
- **follow_up_questions**: A list of suggested research follow up questions related to original query.
- **results**: A list of sorted search results ranked by relevancy. 
  - **title**: The title of the search result url.
  - **url**: The url of the search result.
  - **content**: The most query related content from the scraped url. We use proprietary AI and algorithms to extract only the most relevant content from each url, to optimize for context quality and size.
  - **raw_content**: The parsed and cleaned HTML of the site. For now includes parsed text only.
  - **score**: The relevance score of the search result.

## Example Response

```json
{
    "answer": "Your search result answer",
    "query": "Your search query",
    "response_time": "Your search result response time",
    "follow_up_questions": [
        "follow up question 1",
        "follow up question 2",
        "..."
    ],
    "images": [
      "image url 1",
      "..."
    ]
    "results": [
        {
            "title": "website's title",
            "url": "https://your-search-result-url.com",
            "content": "website's content",
            "raw_content": "website's parsed raw content",
            "score": "tavily's smart relevance score"
        },{},{},{}
    ]
}
```

### Error Codes

- **400**: Bad Request — Your request is invalid.
- **401**: Unauthorized — Your API key is wrong.
- **403**: Forbidden — The endpoint requested is hidden for administrators only.
- **404**: Not Found — The specified endpoint could not be found.
- **405**: Method Not Allowed — You tried to access an endpoint with an invalid method.
- **429**: Too Many Requests — You're requesting too many results! Slow down!
- **500**: Internal Server Error — We had a problem with our server. Try again later.
- **503**: Service Unavailable — We're temporarily offline for maintenance. Please try again later.
- **504**: Gateway Timeout — We're temporarily offline for maintenance. Please try again later.

## Authentication

Tavily Search uses API keys to allow access to the API. You can register a new API key at [https://tavily.com](https://tavily.com).

## Rate Limiting

Tavily Search API has a rate limit of 20 requests per minute.

## Support

For questions, support, or to learn more, please visit [https://tavily.com](https://tavily.com).



# File: langchain.md

# Langchain

We're excited to partner with Langchain as their recommended search tool! 🚀
See the [Langchain blog](https://blog.langchain.dev/weblangchain/) for more details.

Tavily API can now empower your Langchain application with real time online information optimized for RAG.

### How to use Tavily API with Langchain
```python
import os
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain.tools.tavily_search import TavilySearchResults

# set up API key
os.environ["TAVILY_API_KEY"] = "..."

# set up the agent
llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

# initialize the agent
agent_chain = initialize_agent(
    [tavily_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

# run the agent
agent_chain.run(
    "What happened in the latest burning man floods?",
)
```

#### Result:
```commandline
    
    
    > Entering new AgentExecutor chain...
    Thought: I'm not aware of the current situation regarding the Burning Man event. I'll need to search for recent news about any flooding that might have affected it.
    Action:
    ```
    {
      "action": "tavily_search_results_json",
      "action_input": {"query": "Burning Man floods latest news"}
    }
    ```
    Observation: [{'url': 'https://www.theguardian.com/culture/2023/sep/03/burning-man-nevada-festival-floods', 'content': 'More on this story\nMore on this story\nBurning Man revelers begin exodus from festival after road reopens\nBurning Man festival-goers trapped in desert as rain turns site to mud\n\nOfficials investigate death at Burning Man as thousands stranded by floods\n\nBurning Man festivalgoers surrounded by mud in Nevada desert – video\nBurning Man attendees roadblocked by climate activists: ‘They have a privileged mindset’\n\nin our favor. We will let you know. It could be sooner, and it could be later,” said an update on the Burning Man website on Saturday evening.'}, {'url': 'https://www.npr.org/2023/09/03/1197497458/the-latest-on-the-burning-man-flooding', 'content': "National\nThe latest on the Burning Man flooding\nClaudia Peschiutta\n\nClaudia Peschiutta\nAuthorities are investigating a death at the Burning Man festival in the Nevada desert after tens of thousands of people are stuck in camps because of rain.\nSCOTT DETROW, HOST:\n\nDETROW: Well, that's NPR's Claudia Peschiutta covered and caked in a lot of mud at Burning Man. Thanks for talking to us.\nPESCHIUTTA: Confirmed.\nDETROW: Stay dry as much as you can.\n\nwith NPR's Claudia Peschiutta, who's at her first burn, and she told me it's muddy where she is, but that she and her camp family have been making the best of things."}, {'url': 'https://www.npr.org/2023/09/03/1197497458/the-latest-on-the-burning-man-flooding', 'content': "National\nThe latest on the Burning Man flooding\nClaudia Peschiutta\n\nClaudia Peschiutta\nAuthorities are investigating a death at the Burning Man festival in the Nevada desert after tens of thousands of people are stuck in camps because of rain.\nSCOTT DETROW, HOST:\n\nDETROW: Well, that's NPR's Claudia Peschiutta covered and caked in a lot of mud at Burning Man. Thanks for talking to us.\nPESCHIUTTA: Confirmed.\nDETROW: Stay dry as much as you can.\n\nwith NPR's Claudia Peschiutta, who's at her first burn, and she told me it's muddy where she is, but that she and her camp family have been making the best of things."}, {'url': 'https://abcnews.go.com/US/burning-man-flooding-happened-stranded-festivalgoers/story?id=102908331', 'content': 'Tens of thousands of Burning Man attendees are now able to leave the festival after a downpour and massive flooding left them stranded over the weekend.\n\nIn 2013, according to a blog post in the "Burning Man Journal," a rainstorm similarly rolled in, unexpectedly "trapping 160 people on the playa overnight."\n\nABC News\nVideo\nLive\nShows\nElection 2024\n538\nStream on\nBurning Man flooding: What happened to stranded festivalgoers?\nSome 64,000 people were still on site Monday as the exodus began.\n\nBurning Man has been hosted for over 30 years, according to a statement from the organizers.'}, {'url': 'https://www.today.com/news/what-is-burning-man-flood-death-rcna103231', 'content': 'Tens of thousands of Burning Man festivalgoers are slowly making their way home from the Nevada desert after muddy conditions from heavy rains made it nearly impossible to leave over the weekend.\n\naccording to burningman.org.\n\nPresident Biden was notified of the situation and, according to a spokesperson, administration officials monitored and received updates on the latest details.\nWhy are people stranded at Burning Man?\n\n"Thank goodness this community knows how to take care of each other," the Instagram page for Burning Man Information Radio wrote on a post predicting more rain.'}]
    Thought:The latest Burning Man event was severely affected by heavy rainfall that led to flooding. This resulted in tens of thousands of festival attendees getting stuck in their camps due to the muddy conditions. As a result, the exodus from the festival was delayed. An unfortunate incident also occurred, with a death being investigated at the festival. The situation was severe enough that President Biden was informed about it and administration officials were monitoring it. However, it seems that the festival goers were able to handle the situation well, as the Burning Man community is known for looking out for each other. This is not the first time a rainstorm has disrupted the Burning Man event; a similar incident occurred in 2013 where a sudden storm trapped people overnight. 
    Action:
    ```
    {
      "action": "Final Answer",
      "action_input": "The latest Burning Man event was severely affected by heavy rainfall that led to flooding. This resulted in tens of thousands of festival attendees getting stuck in their camps due to the muddy conditions, delaying their exit from the festival. An unfortunate incident also occurred, with a death being investigated at the festival. The situation was severe enough that President Biden was informed about it and administration officials were monitoring it. However, the festival goers were able to handle the situation well, as the Burning Man community is known for looking out for each other. This is not the first time a rainstorm has disrupted the Burning Man event; a similar incident occurred in 2013 when a sudden storm trapped people overnight."
    }
    ```
    
    > Finished chain.





    'The latest Burning Man event was severely affected by heavy rainfall that led to flooding. This resulted in tens of thousands of festival attendees getting stuck in their camps due to the muddy conditions, delaying their exit from the festival. An unfortunate incident also occurred, with a death being investigated at the festival. The situation was severe enough that President Biden was informed about it and administration officials were monitoring it. However, the festival goers were able to handle the situation well, as the Burning Man community is known for looking out for each other. This is not the first time a rainstorm has disrupted the Burning Man event; a similar incident occurred in 2013 when a sudden storm trapped people overnight.'
```



# File: introduction.md

# Introduction

Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, quick and persistent search results. Unlike other search APIs such as Serp or Google, Tavily focuses on optimizing search for AI developers and autonomous AI agents. We take care of all the burden in searching, scraping, filtering and extracting the most relevant information from online sources. All in a single API call! 

The search API can also be used return answers to questions (for use cases such as multi-agent frameworks like autogen) and can complete comprehensive research tasks in seconds. Moreover, Tavily leverages proprietary financial, code, news, and other data internal data sources to complement online information. 

To try our API in action, you can now use GPT Researcher on our hosted version [here](https://app.tavily.com/chat) or on our [API Playground](https://app.tavily.com/playground).

## Why Choose Tavily Search API?

1. **Purpose-Built**: Tailored just for LLM Agents, we ensure the search results are optimized for [RAG](https://towardsdatascience.com/retrieval-augmented-generation-intuitively-and-exhaustively-explain-6a39d6fe6fc9). We take care of all the burden in searching, scraping, filtering and extracting information from online sources. All in a single API call! Simply pass the returned search results as context to your LLM.
2. **Versatility**: Beyond just fetching results, Tavily Search API offers precision. With customizable search depths, domain management, and parsing html content controls, you're in the driver's seat.
3. **Performance**: Committed to rapidity and efficiency, our API guarantees real-time and trusted information. Please note that we're just getting started, so performance may vary and improve over time.
4. **Integration-friendly**: We appreciate the essence of adaptability. That's why integrating our API with your existing setup is a breeze. You can choose our Python library or a simple API call or any of our supported partners such as [Langchain](https://python.langchain.com/docs/integrations/tools/tavily_search) and [LLamaIndex](https://llamahub.ai/l/tools-tavily).
5. **Transparent & Informative**: Our detailed documentation ensures you're never left in the dark. From setup basics to nuanced features, we've got you covered.

## How does the Search API work?
Current search APIs such as Google, Serp and Bing retrieve search results based on user query. However, the results are sometimes irrelevant to the goal of the search, and return simple site URLs and snippets of content which are not always relevant. Because of this, any developer would need to then scrape the sites for relevant content, filter irrelevant information, optimize the content to fit LLM context limits, and more. This tasks is a burden and requires skills to get right.

Tavily Search API aggregates over 20+ sites per a single API call, and uses proprietary AI to score, filter and rank the top most relevant sources and content to your task, query or goal. 
In addition, Tavily allows developers to add custom fields such as context and limit response tokens to enable the optimal search experience for LLMs.

Tavily can also help your AI agent make better decisions such as suggesting follow up search queries or including a short answer for cross agent communication.

Remember: With LLM hallucinations, it's crucial to optimize for RAG with the right context and information.

## Getting Started
1. **Sign Up**: Begin by [signing up](https://app.tavily.com) on our platform.
2. **Obtain Your Unique Key**: Once registered, a unique Tavily API key is generated, ensuring you a seamless connection with our services.
3. **Test Drive in the API Playground**: Before diving in, familiarize yourself by testing out endpoints in our interactive [API playground](https://app.tavily.com/playground). 
4. **Explore & Learn**: Dive into our [Python SDK](/docs/tavily-api/python-sdk) or [REST API](/docs/tavily-api/rest_api) documentation to get familiar with the various features. The documentation offers a comprehensive rundown of functionalities, supplemented with practical sample inputs and outputs.
5. **Sample Use - Research Assistant**: Want a real-world application? Check out our [Research Assistant](https://app.tavily.com/chat) — a prime example that showcases how the API can optimize your AI content generation with factual and unbiased results.
6. **Stay up to date**: Join our [Community](https://discord.gg/rkYFaa8yHy) to get latest updates on our continuous improvements and development

🙋‍♂️ Got questions? Stumbled upon an issue? Or simply intrigued? Don't hesitate! Our support team is always on standby, eager to assist. Join us, dive deep, and redefine your search experience! **[Contact us](mailto:support@tavily.com)**


# File: llamaindex.md

# LlamaIndex

This tool has a more extensive example usage documented in a Jupyter notebook [here](https://github.com/emptycrown/llama-hub/tree/main/llama_hub/tools/notebooks/tavily.ipynb)

Here's an example usage of the TavilyToolSpec.

```python
from llama_hub.tools.tavily_research import TavilyToolSpec
from llama_index.agent import OpenAIAgent

tavily_tool = TavilyToolSpec(
    api_key='your-key',
)
agent = OpenAIAgent.from_tools(tavily_tool.to_tool_list())

agent.chat('What happened in the latest Burning Man festival?')
```

`search`: Search for relevant dynamic data based on a query. Returns a list of urls and their relevant content.


This loader is designed to be used as a way to load data as a Tool in an Agent. See [here](https://github.com/emptycrown/llama-hub/tree/main) for examples.

# File: examples.md

# Examples

### Getting Started
```python
# install tavily
!pip install tavily-python
```

```python  
# import and connect
from tavily import TavilyClient
client = TavilyClient(api_key="")
```
```python  
# simple query using tavily's advanced search
client.search("What happened in the latest burning man floods?", search_depth="advanced")
```
### Response
```commandline
{'query': 'What happened in the latest burning man floods?',
 'follow_up_questions': ['How severe were the floods at Burning Man?',
  'What were the impacts of the floods?',
  'How did the organizers handle the floods at Burning Man?'],
 'answer': None,
 'images': None,
 'results': [{'content': "This year’s rains opened the floodgates for Burning Man criticism  Give Newsletters Site search Vox main menu Filed under: The Burning Man flameout, explained Climate change — and schadenfreude\xa0— finally caught up to the survivalist cosplayers. Share this story Share  Has Burning Man finally lost its glamour?  September 1, after most of the scheduled events and live performances were canceled due to the weather, Burning Man organizers closed routes in and out of the area, forcing attendees to stay behindShare Attendees look at a rainbow over flooding on a desert plain on September 1, 2023, after heavy rains turned the annual Burning Man festival site in Nevada's Black Rock desert into a mud...",
   'url': 'https://www.vox.com/culture/2023/9/6/23861675/burning-man-2023-mud-stranded-climate-change-playa-foot',
   'score': 0.9797,
   'raw_content': None},
  {'content': 'Tens of thousands of Burning Man festivalgoers are slowly making their way home from the Nevada desert after muddy conditions from heavy rains made it nearly impossible to leave over the weekend.  according to burningman.org.  Though the death at this year\'s Burning Man is still being investigated, a social media hoax was blamed for spreading rumors that it\'s due to a breakout of Ebola.  "Thank goodness this community knows how to take care of each other," the Instagram page for Burning Man Information Radio wrote on a post predicting more rain.News Burning Man attendees make mass exodus after being stranded in the mud at festival A caravan of festivalgoers were backed up as much as eight hours when they were finally allowed to leave...',
   'url': 'https://www.today.com/news/what-is-burning-man-flood-death-rcna103231',
   'score': 0.9691,
   'raw_content': None},
  {'content': '“It was a perfect, typical Burning Man weather until Friday — then the rain started coming down hard," said Phillip Martin, 37. "Then it turned into Mud Fest."  After more than a half-inch (1.3 centimeters) of rain fell Friday, flooding turned the playa to foot-deep mud — closing roads and forcing burners to lean on each other for help.  ABC News Video Live Shows Election 2024 538 Stream on No longer stranded, tens of thousands clean up and head home after Burning Man floods  Mark Fromson, 54, who goes by the name “Stuffy” on the playa, had been staying in an RV, but the rains forced him to find shelter at another camp, where fellow burners provided him food and cover.RENO, Nev. -- The traffic jam leaving the Burning Man festival eased up considerably Tuesday as the exodus from the mud-caked Nevada desert entered another day following massive rain that left tens of thousands of partygoers stranded for days.',
   'url': 'https://abcnews.go.com/US/wireStory/wait-times-exit-burning-man-drop-after-flooding-102936473',
   'score': 0.9648,
   'raw_content': None},
  {'content': 'Burning Man hit by heavy rains, now mud soaked.People there told to conserve food and water as they shelter in place.(Video: Josh Keppel) pic.twitter.com/DuBj0Ejtb8  More on this story Burning Man revelers begin exodus from festival after road reopens Officials investigate death at Burning Man as thousands stranded by floods  Burning Man festival-goers trapped in desert as rain turns site to mud Tens of thousands of ‘burners’ urged to conserve food and water as rain and flash floods sweep Nevada  Burning Man festivalgoers surrounded by mud in Nevada desert – video Burning Man attendees roadblocked by climate activists: ‘They have a privileged mindset’Last year, Burning Man drew approximately 80,000 people. This year, only about 60,000 were expected - with many citing the usual heat and dust and eight-hour traffic jams when they tried to leave.',
   'url': 'https://www.theguardian.com/culture/2023/sep/02/burning-man-festival-mud-trapped-shelter-in-place',
   'score': 0.9618,
   'raw_content': None},
  {'content': 'Skip links Live Navigation menu Live Death at Burning Man investigated in US, thousands stranded by flooding  Attendees trudged through mud, many barefoot or wearing plastic bags on their feet. The revellers were urged to shelter in place and conserve food, water and other supplies.  Thousands of festivalgoers remain stranded as organisers close vehicular traffic to the festival site following storm flooding in Nevada’s desert.  Authorities in Nevada are investigating a death at the site of the Burning Man festival, where thousands of attendees remained stranded after flooding from storms swept through the Nevada desert in3 Sep 2023. Authorities in Nevada are investigating a death at the site of the Burning Man festival, where thousands of attendees remained stranded after flooding from storms swept through the ...',
   'url': 'https://www.aljazeera.com/news/2023/9/3/death-under-investigation-after-storm-flooding-at-burning-man-festival',
   'score': 0.9612,
   'raw_content': None}],
 'response_time': 6.23}
```

### Sample 1: Research Report using Tavily and GPT-4 with Langchain
```python
# install lanchain
!pip install langchain
```

```python
# set up openai api key
openai_api_key = ""
```
```python
# libraries
from langchain.adapters.openai import convert_openai_messages
from langchain_community.chat_models import ChatOpenAI

# setup query
query = "What happened in the latest burning man floods?"

# run tavily search
content = client.search(query, search_depth="advanced")["results"]

# setup prompt
prompt = [{
    "role": "system",
    "content":  f'You are an AI critical thinker research assistant. '\
                f'Your sole purpose is to write well written, critically acclaimed,'\
                f'objective and structured reports on given text.'
}, {
    "role": "user",
    "content": f'Information: """{content}"""\n\n' \
               f'Using the above information, answer the following'\
               f'query: "{query}" in a detailed report --'\
               f'Please use MLA format and markdown syntax.'
}]

# run gpt-4
lc_messages = convert_openai_messages(prompt)
report = ChatOpenAI(model='gpt-4',openai_api_key=openai_api_key).invoke(lc_messages).content

# print report
print(report)
```
### Response
```commandline
# The Burning Man Festival 2023: A Festival Turned Mud Fest

**Abstract:** The Burning Man Festival of 2023 in Nevada’s Black Rock desert will be remembered for a significant event: a heavy rainfall that turned the festival site into a muddy mess, testing the community spirit of the annual event attendees and stranding tens of thousands of festival-goers. 

**Keywords:** Burning Man Festival, flooding, rainfall, mud, community spirit, Nevada, Black Rock desert, stranded attendees, shelter

---
## 1. Introduction

The Burning Man Festival, an annual event known for its art installations, free spirit, and community ethos, faced an unprecedented challenge in 2023 due to heavy rains that flooded the festival site, turning it into a foot-deep mud pit[^1^][^2^]. The festival, held in Nevada's Black Rock desert, is known for its harsh weather conditions, including heat and dust, but this was the first time the event was affected to such an extent by rainfall[^4^].

## 2. Impact of the Rain

The heavy rains started on Friday, and more than a half-inch of rain fell, leading to flooding that turned the playa into a foot-deep mud pit[^2^]. The roads were closed due to the muddy conditions, stranding tens of thousands of festival-goers[^2^][^5^]. The burners, as the attendees are known, were forced to lean on each other for help[^2^].

## 3. Community Spirit Tested

The unexpected weather conditions put the Burning Man community spirit to the test[^1^]. Festival-goers found themselves sheltering in place, conserving food and water, and helping each other out[^3^]. For instance, Mark Fromson, who had been staying in an RV, was forced to find shelter at another camp due to the rains, where fellow burners provided him with food and cover[^2^].

## 4. Exodus After Rain

Despite the challenges, the festival-goers made the best of the situation. Once the rain stopped and things dried up a bit, the party quickly resumed[^3^]. A day later than scheduled, the massive wooden effigy known as the Man was set ablaze[^5^]. As the situation improved, thousands of Burning Man attendees began their mass exodus from the festival site[^5^].

## 5. Conclusion

The Burning Man Festival of 2023 will be remembered for the community spirit shown by the attendees in the face of heavy rainfall and flooding. Although the event was marred by the weather, the festival-goers managed to make the best of the situation, demonstrating the resilience and camaraderie that the Burning Man Festival is known for.

---
**References**

[^1^]: "Attendees walk through a muddy desert plain..." NPR. 2023. https://www.npr.org/2023/09/02/1197441202/burning-man-festival-rains-floods-stranded-nevada.

[^2^]: “'It was a perfect, typical Burning Man weather until Friday...'" ABC News. 2023. https://abcnews.go.com/US/wireStory/wait-times-exit-burning-man-drop-after-flooding-102936473.

[^3^]: "The latest on the Burning Man flooding..." WUNC. 2023. https://www.wunc.org/2023-09-03/the-latest-on-the-burning-man-flooding.

[^4^]: "Burning Man hit by heavy rains, now mud soaked..." The Guardian. 2023. https://www.theguardian.com/culture/2023/sep/02/burning-man-festival-mud-trapped-shelter-in-place.

[^5^]: "One day later than scheduled, the massive wooden effigy known as the Man was set ablaze..." CNN. 2023. https://www.cnn.com/2023/09/05/us/burning-man-storms-shelter-exodus-tuesday/index.html.
```


# File: sidebar.json

{
  "items": [
    {
      "items": [
        "reference/config/config",
        "reference/config/singleton"
      ],
      "label": "config",
      "type": "category"
    },
    {
      "items": [
        "reference/processing/html",
        "reference/processing/text"
      ],
      "label": "processing",
      "type": "category"
    }
  ],
  "label": "Reference",
  "type": "category"
}

# File: singleton.md

---
sidebar_label: singleton
title: config.singleton
---

The singleton metaclass for ensuring only one instance of a class.

## Singleton Objects

```python
class Singleton(abc.ABCMeta, type)
```

Singleton metaclass for ensuring only one instance of a class.

#### \_\_call\_\_

```python
def __call__(cls, *args, **kwargs)
```

Call method for the singleton metaclass.

## AbstractSingleton Objects

```python
class AbstractSingleton(abc.ABC, metaclass=Singleton)
```

Abstract singleton class for ensuring only one instance of a class.



# File: config.md

---
sidebar_label: config
title: config.config
---

Configuration class to store the state of bools for different scripts access.

## Config Objects

```python
class Config(metaclass=Singleton)
```

Configuration class to store the state of bools for different scripts access.

#### \_\_init\_\_

```python
def __init__() -> None
```

Initialize the Config class

#### set\_fast\_llm\_model

```python
def set_fast_llm_model(value: str) -> None
```

Set the fast LLM model value.

#### set\_smart\_llm\_model

```python
def set_smart_llm_model(value: str) -> None
```

Set the smart LLM model value.

#### set\_fast\_token\_limit

```python
def set_fast_token_limit(value: int) -> None
```

Set the fast token limit value.

#### set\_smart\_token\_limit

```python
def set_smart_token_limit(value: int) -> None
```

Set the smart token limit value.

#### set\_browse\_chunk\_max\_length

```python
def set_browse_chunk_max_length(value: int) -> None
```

Set the browse_website command chunk max length value.

#### set\_openai\_api\_key

```python
def set_openai_api_key(value: str) -> None
```

Set the OpenAI API key value.

#### set\_debug\_mode

```python
def set_debug_mode(value: bool) -> None
```

Set the debug mode value.

## APIKeyError Objects

```python
class APIKeyError(Exception)
```

Exception raised when an API key is not set in config.py or as an environment variable.

#### check\_openai\_api\_key

```python
def check_openai_api_key(cfg) -> None
```

Check if the OpenAI API key is set in config.py or as an environment variable.

#### check\_tavily\_api\_key

```python
def check_tavily_api_key(cfg) -> None
```

Check if the Tavily Search API key is set in config.py or as an environment variable.

#### check\_google\_api\_key

```python
def check_google_api_key(cfg) -> None
```

Check if the Google API key is set in config.py or as an environment variable.

#### check\_serp\_api\_key

```python
def check_serp_api_key(cfg) -> None
```

Check if the SERP API key is set in config.py or as an environment variable.

#### check\_searx\_url

```python
def check_searx_url(cfg) -> None
```

Check if the Searx URL is set in config.py or as an environment variable.



# File: text.md

---
sidebar_label: text
title: processing.text
---

Text processing functions

#### split\_text

```python
def split_text(text: str,
               max_length: int = 8192) -> Generator[str, None, None]
```

Split text into chunks of a maximum length

**Arguments**:

- `text` _str_ - The text to split
- `max_length` _int, optional_ - The maximum length of each chunk. Defaults to 8192.
  

**Yields**:

- `str` - The next chunk of text
  

**Raises**:

- `ValueError` - If the text is longer than the maximum length

#### summarize\_text

```python
def summarize_text(url: str,
                   text: str,
                   question: str,
                   driver: Optional[WebDriver] = None) -> str
```

Summarize text using the OpenAI API

**Arguments**:

- `url` _str_ - The url of the text
- `text` _str_ - The text to summarize
- `question` _str_ - The question to ask the model
- `driver` _WebDriver_ - The webdriver to use to scroll the page
  

**Returns**:

- `str` - The summary of the text

#### scroll\_to\_percentage

```python
def scroll_to_percentage(driver: WebDriver, ratio: float) -> None
```

Scroll to a percentage of the page

**Arguments**:

- `driver` _WebDriver_ - The webdriver to use
- `ratio` _float_ - The percentage to scroll to
  

**Raises**:

- `ValueError` - If the ratio is not between 0 and 1

#### create\_message

```python
def create_message(chunk: str, question: str) -> Dict[str, str]
```

Create a message for the chat completion

**Arguments**:

- `chunk` _str_ - The chunk of text to summarize
- `question` _str_ - The question to answer
  

**Returns**:

  Dict[str, str]: The message to send to the chat completion

#### write\_to\_file

```python
def write_to_file(filename: str, text: str) -> None
```

Write text to a file

**Arguments**:

- `text` _str_ - The text to write
- `filename` _str_ - The filename to write to



# File: html.md

---
sidebar_label: html
title: processing.html
---

HTML processing functions

#### extract\_hyperlinks

```python
def extract_hyperlinks(soup: BeautifulSoup,
                       base_url: str) -> list[tuple[str, str]]
```

Extract hyperlinks from a BeautifulSoup object

**Arguments**:

- `soup` _BeautifulSoup_ - The BeautifulSoup object
- `base_url` _str_ - The base URL
  

**Returns**:

  List[Tuple[str, str]]: The extracted hyperlinks

#### format\_hyperlinks

```python
def format_hyperlinks(hyperlinks: list[tuple[str, str]]) -> list[str]
```

Format hyperlinks to be displayed to the user

**Arguments**:

- `hyperlinks` _List[Tuple[str, str]]_ - The hyperlinks to format
  

**Returns**:

- `List[str]` - The formatted hyperlinks



# File: authors.yml

assafe:
  name: Assaf Elovic
  title: Creator @ GPT Researcher
  url: https://github.com/assafelovic
  image_url: https://lh3.googleusercontent.com/a/ACg8ocJtrLku69VG_2Y0sJa5mt66gIGNaEBX5r_mgE6CRPEb7A=s96-c


# File: index.md

---
slug: building-gpt-researcher
title: How we built GPT Researcher
authors: [assafe]
tags: [gpt-researcher, autonomous-agent, opensource, github]
---

After [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT) was published, we immediately took it for a spin. The first use case that came to mind was autonomous online research. Forming objective conclusions for manual research tasks can take time, sometimes weeks, to find the right resources and information. Seeing how well AutoGPT created tasks and executed them got me thinking about the great potential of using AI to conduct comprehensive research and what it meant for the future of online research.

But the problem with AutoGPT was that it usually ran into never-ending loops, required human interference for almost every step, constantly lost track of its progress, and almost never actually completed the task.

Nonetheless, the information and context gathered during the research task were lost (such as keeping track of sources), and sometimes hallucinated.

The passion for leveraging AI for online research and the limitations I found put me on a mission to try and solve it while sharing my work with the world. This is when I created [GPT Researcher](https://github.com/assafelovic/gpt-researcher) — an open source autonomous agent for online comprehensive research.

In this article, we will share the steps that guided me toward the proposed solution.

### Moving from infinite loops to deterministic results
The first step in solving these issues was to seek a more deterministic solution that could ultimately guarantee completing any research task within a fixed time frame, without human interference.

This is when we stumbled upon the recent paper [Plan and Solve](https://arxiv.org/abs/2305.04091). The paper aims to provide a better solution for the challenges stated above. The idea is quite simple and consists of two components: first, devising a plan to divide the entire task into smaller subtasks and then carrying out the subtasks according to the plan.

![Planner-Excutor-Model](./planner.jpeg)

As it relates to research, first create an outline of questions to research related to the task, and then deterministically execute an agent for every outline item. This approach eliminates the uncertainty in task completion by breaking the agent steps into a deterministic finite set of tasks. Once all tasks are completed, the agent concludes the research.

Following this strategy has improved the reliability of completing research tasks to 100%. Now the challenge is, how to improve quality and speed?

### Aiming for objective and unbiased results
The biggest challenge with LLMs is the lack of factuality and unbiased responses caused by hallucinations and out-of-date training sets (GPT is currently trained on datasets from 2021). But the irony is that for research tasks, it is crucial to optimize for these exact two criteria: factuality and bias.

To tackle this challenges, we assumed the following:

- Law of large numbers — More content will lead to less biased results. Especially if gathered properly.
- Leveraging LLMs for the summarization of factual information can significantly improve the overall better factuality of results.

After experimenting with LLMs for quite some time, we can say that the areas where foundation models excel are in the summarization and rewriting of given content. So, in theory, if LLMs only review given content and summarize and rewrite it, potentially it would reduce hallucinations significantly.

In addition, assuming the given content is unbiased, or at least holds opinions and information from all sides of a topic, the rewritten result would also be unbiased. So how can content be unbiased? The [law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers). In other words, if enough sites that hold relevant information are scraped, the possibility of biased information reduces greatly. So the idea would be to scrape just enough sites together to form an objective opinion on any topic.

Great! Sounds like, for now, we have an idea for how to create both deterministic, factual, and unbiased results. But what about the speed problem?

### Speeding up the research process
Another issue with AutoGPT is that it works synchronously. The main idea of it is to create a list of tasks and then execute them one by one. So if, let’s say, a research task requires visiting 20 sites, and each site takes around one minute to scrape and summarize, the overall research task would take a minimum of +20 minutes. That’s assuming it ever stops. But what if we could parallelize agent work?

By levering Python libraries such as asyncio, the agent tasks have been optimized to work in parallel, thus significantly reducing the time to research.

```python
# Create a list to hold the coroutine agent tasks
tasks = [async_browse(url, query, self.websocket) for url in await new_search_urls]

# Gather the results as they become available
responses = await asyncio.gather(*tasks, return_exceptions=True)
```

In the example above, we trigger scraping for all URLs in parallel, and only once all is done, continue with the task. Based on many tests, an average research task takes around three minutes (!!). That’s 85% faster than AutoGPT.

### Finalizing the research report
Finally, after aggregating as much information as possible about a given research task, the challenge is to write a comprehensive report about it.

After experimenting with several OpenAI models and even open source, I’ve concluded that the best results are currently achieved with GPT-4. The task is straightforward — provide GPT-4 as context with all the aggregated information, and ask it to write a detailed report about it given the original research task.

The prompt is as follows:
```commandline
"{research_summary}" Using the above information, answer the following question or topic: "{question}" in a detailed report — The report should focus on the answer to the question, should be well structured, informative, in depth, with facts and numbers if available, a minimum of 1,200 words and with markdown syntax and apa format. Write all source urls at the end of the report in apa format. You should write your report only based on the given information and nothing else.
```

The results are quite impressive, with some minor hallucinations in very few samples, but it’s fair to assume that as GPT improves over time, results will only get better.

### The final architecture
Now that we’ve reviewed the necessary steps of GPT Researcher, let’s break down the final architecture, as shown below:

<div align="center">
<img align="center" height="500" src="https://cowriter-images.s3.amazonaws.com/architecture.png"/>
</div>

More specifically:
- Generate an outline of research questions that form an objective opinion on any given task.
- For each research question, trigger a crawler agent that scrapes online resources for information relevant to the given task.
- For each scraped resource, keep track, filter, and summarize only if it includes relevant information.
- Finally, aggregate all summarized sources and generate a final research report.

### Going forward
The future of online research automation is heading toward a major disruption. As AI continues to improve, it is only a matter of time before AI agents can perform comprehensive research tasks for any of our day-to-day needs. AI research can disrupt areas of finance, legal, academia, health, and retail, reducing our time for each research by 95% while optimizing for factual and unbiased reports within an influx and overload of ever-growing online information.

Imagine if an AI can eventually understand and analyze any form of online content — videos, images, graphs, tables, reviews, text, audio. And imagine if it could support and analyze hundreds of thousands of words of aggregated information within a single prompt. Even imagine that AI can eventually improve in reasoning and analysis, making it much more suitable for reaching new and innovative research conclusions. And that it can do all that in minutes, if not seconds.

It’s all a matter of time and what [GPT Researcher](https://github.com/assafelovic/gpt-researcher) is all about.


# File: index.md

---
slug: building-openai-assistant
title: How to build an OpenAI Assistant with Internet access
authors: [assafe]
tags: [tavily, search-api, openai, assistant-api]
---

OpenAI has done it again with a [groundbreaking DevDay](https://openai.com/blog/new-models-and-developer-products-announced-at-devday) showcasing some of the latest improvements to the OpenAI suite of tools, products and services. One major release was the new [Assistants API](https://platform.openai.com/docs/assistants/overview) that makes it easier for developers to build their own assistive AI apps that have goals and can call models and tools.

The new Assistants API currently supports three types of tools: Code Interpreter, Retrieval, and Function calling. Although you might expect the Retrieval tool to support online information retrieval (such as search APIs or as ChatGPT plugins), it only supports raw data for now such as text or CSV files.

This blog will demonstrate how to leverage the latest Assistants API with online information using the function calling tool.

To skip the tutorial below, feel free to check out the full [Github Gist here](https://gist.github.com/assafelovic/579822cd42d52d80db1e1c1ff82ffffd).

At a high level, a typical integration of the Assistants API has the following steps:

- Create an [Assistant](https://platform.openai.com/docs/api-reference/assistants/createAssistant) in the API by defining its custom instructions and picking a model. If helpful, enable tools like Code Interpreter, Retrieval, and Function calling.
- Create a [Thread](https://platform.openai.com/docs/api-reference/threads) when a user starts a conversation.
- Add [Messages](https://platform.openai.com/docs/api-reference/messages) to the Thread as the user ask questions.
- [Run](https://platform.openai.com/docs/api-reference/runs) the Assistant on the Thread to trigger responses. This automatically calls the relevant tools.

As you can see below, an Assistant object includes Threads for storing and handling conversation sessions between the assistant and users, and Run for invocation of an Assistant on a Thread.

![OpenAI Assistant Object](./diagram-assistant.jpeg)

Let’s go ahead and implement these steps one by one! For the example, we will build a finance GPT that can provide insights about financial questions. We will use the [OpenAI Python SDK v1.2](https://github.com/openai/openai-python/tree/main#installation) and [Tavily Search API](https://tavily.com).

First things first, let’s define the assistant’s instructions:

```python
assistant_prompt_instruction = """You are a finance expert. 
Your goal is to provide answers based on information from the internet. 
You must use the provided Tavily search API function to find relevant online information. 
You should never use your own knowledge to answer questions.
Please include relevant url sources in the end of your answers.
"""
```
Next, let’s finalize step 1 and create an assistant using the latest [GPT-4 Turbo model](https://github.com/openai/openai-python/tree/main#installation) (128K context), and the call function using the [Tavily web search API](https://tavily.com/):

```python
# Create an assistant
assistant = client.beta.assistants.create(
    instructions=assistant_prompt_instruction,
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Get information on recent events from the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to use. For example: 'Latest news on Nvidia stock performance'"},
                },
                "required": ["query"]
            }
        }
    }]
)
```

Step 2+3 are quite straight forward, we’ll initiate a new thread and update it with a user message:

```python
thread = client.beta.threads.create()
user_input = input("You: ")
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=user_input,
)
```

Finally, we’ll run the assistant on the thread to trigger the function call and get the response:

```python
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant_id,
)
```

So far so good! But this is where it gets a bit messy. Unlike with the regular GPT APIs, the Assistants API doesn’t return a synchronous response, but returns a status. This allows for asynchronous operations across assistants, but requires more overhead for fetching statuses and dealing with each manually.

![Status Diagram](./diagram-1.png)

To manage this status lifecycle, let’s build a function that can be reused and handles waiting for various statuses (such as ‘requires_action’):

```python
# Function to wait for a run to complete
def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ['completed', 'failed', 'requires_action']:
            return run
```

This function will sleep as long as the run has not been finalized such as in cases where it’s completed or requires an action from a function call.

We’re almost there! Lastly, let’s take care of when the assistant wants to call the web search API:

```python
# Function to handle tool output submission
def submit_tool_outputs(thread_id, run_id, tools_to_call):
    tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name == "tavily_search":
            output = tavily_search(query=json.loads(function_args)["query"])

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )
```

As seen above, if the assistant has reasoned that a function call should trigger, we extract the given required function params and pass back to the runnable thread. We catch this status and call our functions as seen below:

```python
if run.status == 'requires_action':
    run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
    run = wait_for_run_completion(thread.id, run.id)
```

That’s it! We now have a working OpenAI Assistant that can be used to answer financial questions using real time online information. Below is the full runnable code:

```python
import os
import json
import time
from openai import OpenAI
from tavily import TavilyClient

# Initialize clients with API keys
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

assistant_prompt_instruction = """You are a finance expert. 
Your goal is to provide answers based on information from the internet. 
You must use the provided Tavily search API function to find relevant online information. 
You should never use your own knowledge to answer questions.
Please include relevant url sources in the end of your answers.
"""

# Function to perform a Tavily search
def tavily_search(query):
    search_result = tavily_client.get_search_context(query, search_depth="advanced", max_tokens=8000)
    return search_result

# Function to wait for a run to complete
def wait_for_run_completion(thread_id, run_id):
    while True:
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
        print(f"Current run status: {run.status}")
        if run.status in ['completed', 'failed', 'requires_action']:
            return run

# Function to handle tool output submission
def submit_tool_outputs(thread_id, run_id, tools_to_call):
    tool_output_array = []
    for tool in tools_to_call:
        output = None
        tool_call_id = tool.id
        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name == "tavily_search":
            output = tavily_search(query=json.loads(function_args)["query"])

        if output:
            tool_output_array.append({"tool_call_id": tool_call_id, "output": output})

    return client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_output_array
    )

# Function to print messages from a thread
def print_messages_from_thread(thread_id):
    messages = client.beta.threads.messages.list(thread_id=thread_id)
    for msg in messages:
        print(f"{msg.role}: {msg.content[0].text.value}")

# Create an assistant
assistant = client.beta.assistants.create(
    instructions=assistant_prompt_instruction,
    model="gpt-4-1106-preview",
    tools=[{
        "type": "function",
        "function": {
            "name": "tavily_search",
            "description": "Get information on recent events from the web.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to use. For example: 'Latest news on Nvidia stock performance'"},
                },
                "required": ["query"]
            }
        }
    }]
)
assistant_id = assistant.id
print(f"Assistant ID: {assistant_id}")

# Create a thread
thread = client.beta.threads.create()
print(f"Thread: {thread}")

# Ongoing conversation loop
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Create a message
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_input,
    )

    # Create a run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
    )
    print(f"Run ID: {run.id}")

    # Wait for run to complete
    run = wait_for_run_completion(thread.id, run.id)

    if run.status == 'failed':
        print(run.error)
        continue
    elif run.status == 'requires_action':
        run = submit_tool_outputs(thread.id, run.id, run.required_action.submit_tool_outputs.tool_calls)
        run = wait_for_run_completion(thread.id, run.id)

    # Print messages from the thread
    print_messages_from_thread(thread.id)
```

The assistant can be further customized and improved using additional retrieval information, OpenAI’s coding interpreter and more. Also, you can go ahead and add more function tools to make the assistant even smarter.

Feel free to drop a comment below if you have any further questions!


# File: README.md

# Website

This website is built using [Docusaurus 2](https://docusaurus.io/), a modern static website generator.

## Prerequisites

To build and test documentation locally, begin by downloading and installing [Node.js](https://nodejs.org/en/download/), and then installing [Yarn](https://classic.yarnpkg.com/en/).
On Windows, you can install via the npm package manager (npm) which comes bundled with Node.js:

```console
npm install --global yarn
```

## Installation

```console
pip install pydoc-markdown
cd website
yarn install
```

## Local Development

Navigate to the website folder and run:

```console
pydoc-markdown
yarn start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.


# File: babel.config.js

module.exports = {
  presets: [require.resolve('@docusaurus/core/lib/babel/preset')],
};


# File: package.json

{
  "name": "website",
  "version": "0.0.0",
  "private": true,
  "resolutions" :{
    "nth-check":"2.0.1",
    "trim":"0.0.3",
    "got": "11.8.5",
    "node-forge": "1.3.0",
    "minimatch": "3.0.5",
    "loader-utils": "2.0.4",
    "eta": "2.0.0",
    "@sideway/formula": "3.0.1",
    "http-cache-semantics": "4.1.1"
   },
  "scripts": {
    "docusaurus": "docusaurus",
    "start": "docusaurus start",
    "build": "docusaurus build",
    "swizzle": "docusaurus swizzle",
    "deploy": "docusaurus deploy",
    "clear": "docusaurus clear",
    "serve": "docusaurus serve",
    "write-translations": "docusaurus write-translations",
    "write-heading-ids": "docusaurus write-heading-ids"
  },
  "dependencies": {
    "@docusaurus/core": "0.0.0-4193",
    "@docusaurus/preset-classic": "0.0.0-4193",
    "@easyops-cn/docusaurus-search-local": "^0.21.1",
    "@mdx-js/react": "^1.6.21",
    "@svgr/webpack": "^5.5.0",
    "clsx": "^1.1.1",
    "file-loader": "^6.2.0",
    "hast-util-is-element": "1.1.0",
    "react": "^17.0.1",
    "react-dom": "^17.0.1",
    "rehype-katex": "4",
    "remark-math": "3",
    "trim": "^0.0.3",
    "url-loader": "^4.1.1",
    "minimatch": "3.0.5"
  },
  "browserslist": {
    "production": [
      ">0.5%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}


# File: docusaurus.config.js

/** @type {import('@docusaurus/types').DocusaurusConfig} */
const math = require('remark-math');
const katex = require('rehype-katex');

module.exports = {
  title: 'Tavily',
  tagline: 'Tavily is the leading search engine optimized for LLMs',
  url: 'https://docs.tavily.com',
  baseUrl: '/',
  onBrokenLinks: 'ignore',
  //deploymentBranch: 'master',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'assafelovic',
  trailingSlash: false,
  projectName: 'gpt-researcher',
  themeConfig: {
    navbar: {
      //title: 'Tavily',
      logo: {
        alt: 'Tavily',
        src: 'img/tavily.png',
      },
      items: [
        {
          type: 'doc',
          docId: 'welcome',
          position: 'left',
          label: 'Docs',
        },

        {to: 'blog', label: 'Blog', position: 'left'},
        {
          type: 'doc',
          docId: 'faq',
          position: 'left',
          label: 'FAQ',
        },
        {
            href: 'https://app.tavily.com',
            position: 'right',
            label: 'Get API Key',
        },
        {
            href: 'mailto:support@tavily.com',
            position: 'left',
            label: 'Contact',
        },
        {
          href: 'https://github.com/assafelovic/gpt-researcher',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Community',
          items: [
            {
              label: 'Discord',
              href: 'https://discord.gg/8YkBcCED5y',
            },
            {
              label: 'Twitter',
              href: 'https://twitter.com/tavilyai',
            },
            {
              label: 'LinkedIn',
              href: 'https://www.linkedin.com/company/tavily/',
            },
          ],
        },
        {
          title: 'Company',
          items: [
            {
              label: 'Homepage',
              href: 'https://tavily.com',
            },
            {
              label: 'Tavily Platform',
              href: 'https://tavily.com',
            },
            {
              label: 'Contact',
              href: 'mailto:support@tavily.com',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Tavily.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/assafelovic/gpt-researcher/tree/master/docs',
          remarkPlugins: [math],
          rehypePlugins: [katex],
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
  stylesheets: [
    {
        href: "https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css",
        integrity: "sha384-Um5gpz1odJg5Z4HAmzPtgZKdTBHZdw8S29IecapCSB31ligYPhHQZMIlWLYQGVoc",
        crossorigin: "anonymous",
    },
  ],

  plugins: [
    // ... Your other plugins.
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        // ... Your options.
        // `hashed` is recommended as long-term-cache of index file is possible.
        hashed: true,
        blogDir:"./blog/"
        // For Docs using Chinese, The `language` is recommended to set to:
        // ```
        // language: ["en", "zh"],
        // ```
        // When applying `zh` in language, please install `nodejieba` in your project.
      },
    ],
  ],
};


# File: sidebars.js

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

 module.exports = {
  docsSidebar: [
    'welcome',
    {
      type: 'category',
      label: 'GPT Researcher',
      collapsible: true,
      collapsed: false,
      items: [
        'gpt-researcher/introduction',
        'gpt-researcher/getting-started',
        'gpt-researcher/config',
         'gpt-researcher/example',
         'gpt-researcher/agent_frameworks',
        'gpt-researcher/pip-package',
        'gpt-researcher/troubleshooting',
      ],
    },
    {
      type: 'category',
      label: 'Tavily API',
      collapsible: true,
      collapsed: false,
      items: [
        'tavily-api/introduction',
        'tavily-api/python-sdk',
        'tavily-api/rest_api',
        'tavily-api/langchain',
        'tavily-api/llamaindex',
        //{'Topics': [{type: 'autogenerated', dirName: 'tavily-api/Topics'}]},
      ],
    },
    //{'GPT Researcher': [{type: 'autogenerated', dirName: 'gpt-researcher'}]},
    //{'Tavily API': [{type: 'autogenerated', dirName: 'tavily-api'}]},
    {'Examples': [{type: 'autogenerated', dirName: 'examples'}]},
    'contribute',
  ],
  // pydoc-markdown auto-generated markdowns from docstrings
  referenceSideBar: [require("./docs/reference/sidebar.json")]
};


# File: custom.css

:root {
  --ifm-font-size-base: 17px;
  --ifm-code-font-size: 90%;

  --ifm-color-primary: #0c4da2;
  --ifm-color-primary-dark: rgb(11, 69, 146);
  --ifm-color-primary-darker: #0a418a;
  --ifm-color-primary-darkest: #083671;
  --ifm-color-primary-light: #0d55b2;
  --ifm-color-primary-lighter: #0e59ba;
  --ifm-color-primary-lightest: #1064d3;

  --ifm-color-emphasis-300: #1064d3;
  --ifm-link-color: #1064d3;
  --ifm-menu-color-active: #1064d3;
}

.docusaurus-highlight-code-line {
background-color: rgba(0, 0, 0, 0.1);
display: block;
margin: 0 calc(-1 * var(--ifm-pre-padding));
padding: 0 var(--ifm-pre-padding);
}
html[data-theme='dark'] .docusaurus-highlight-code-line {
background-color: rgb(0, 0, 0, 0.3);
}

.admonition-content a {
text-decoration: underline;
font-weight: 600;
color: inherit;
}

a {
font-weight: 600;
}

blockquote {
  /* samsung blue with lots of transparency */
  background-color: #0c4da224;
}
@media (prefers-color-scheme: dark) {
:root {
  --ifm-hero-text-color: white;
}
}
@media (prefers-color-scheme: dark) {
.hero.hero--primary { --ifm-hero-text-color: white;}
}

@media (prefers-color-scheme: dark) {
blockquote {
  --ifm-color-emphasis-300: var(--ifm-color-primary);
  /* border-left: 6px solid var(--ifm-color-emphasis-300); */
}
}
@media (prefers-color-scheme: dark) {
code {
  /* background-color: rgb(41, 45, 62); */
}
}


/* Docusaurus still defaults to their green! */
@media (prefers-color-scheme: dark) {
.react-toggle-thumb {
  border-color: var(--ifm-color-primary) !important;
}
}


.header-github-link:hover {
opacity: 0.6;
}

.header-github-link:before {
content: '';
width: 24px;
height: 24px;
display: flex;
background: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12'/%3E%3C/svg%3E")
  no-repeat;
}

html[data-theme='dark'] .header-github-link:before {
background: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath fill='white' d='M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12'/%3E%3C/svg%3E")
  no-repeat;
}


# File: HomepageFeatures.module.css

/* stylelint-disable docusaurus/copyright-header */

.features {
  display: flex;
  align-items: center;
  padding: 2rem 0;
  width: 100%;
}

.featureSvg {
  height: 120px;
  width: 200px;
}


# File: HomepageFeatures.js

import React from 'react';
import clsx from 'clsx';
import { Link } from 'react-router-dom';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'GPT Researcher',
    Svg: require('../../static/img/gptresearcher.png').default,
    docLink: './docs/gpt-researcher/getting-started',
    description: (
      <>
        GPT Researcher is an open source autonomous agent designed for comprehensive online research on a variety of tasks.
      </>
    ),
  },
  {
    title: 'Tavily Search API',
    Svg: require('../../static/img/tavily.png').default,
    docLink: './docs/tavily-api/introduction',
    description: (
      <>
        Tavily Search API is a search engine optimized for LLMs, optimized for a factual, efficient, and persistent search experience
      </>
    ),
  },
  {
    title: 'Examples and Demos',
    Svg: require('../../static/img/examples.png').default,
    docLink: './docs/examples/examples',
    description: (
      <>
          Check out Tavily API in action across multiple frameworks and use cases
      </>
    ),
  },
];

function Feature({Svg, title, description, docLink}) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {/*<Svg className={styles.featureSvg} alt={title} />*/}
        <img src={Svg} alt={title} height="60"/>
      </div>
      <div className="text--center padding-horiz--md">
        <Link to={docLink}>
            <h3>{title}</h3>
        </Link>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}


# File: index.module.css

/* stylelint-disable docusaurus/copyright-header */

/**
 * CSS files with the .module.css suffix will be treated as CSS modules
 * and scoped locally.
 */

.heroBanner {
  padding: 4rem 0;
  text-align: center;
  position: relative;
  overflow: hidden;
}

@media screen and (max-width: 966px) {
  .heroBanner {
    padding: 2rem;
  }
}

.buttons {
  display: flex;
  align-items: center;
  justify-content: center;
}


# File: index.js

import React from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import styles from './index.module.css';
import HomepageFeatures from '../components/HomepageFeatures';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/welcome">
            Getting Started - 5 min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Tavily Documentation`}
      description="Tavily is the leading search engine optimized for AI agents - powered by LLMs.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
