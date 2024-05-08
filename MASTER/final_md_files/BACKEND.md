

# File: server.py

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from backend.websocket_manager import WebSocketManager
from backend.utils import write_md_to_pdf, write_md_to_word, write_text_to_md
import time
import json
import os


class ResearchRequest(BaseModel):
    task: str
    report_type: str
    agent: str


app = FastAPI()

app.mount("/site", StaticFiles(directory="./frontend"), name="site")
app.mount("/static", StaticFiles(directory="./frontend/static"), name="static")

templates = Jinja2Templates(directory="./frontend")

manager = WebSocketManager()


# Dynamic directory for outputs once first research is run
@app.on_event("startup")
def startup_event():
    if not os.path.isdir("outputs"):
        os.makedirs("outputs")
    app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {"request": request, "report": None})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data.startswith("start"):
                json_data = json.loads(data[6:])
                task = json_data.get("task")
                report_type = json_data.get("report_type")
                filename = f"task_{int(time.time())}_{task}"
                if task and report_type:
                    report = await manager.start_streaming(task, report_type, websocket)
                    # Saving report as pdf
                    pdf_path = await write_md_to_pdf(report, filename)
                    # Saving report as docx
                    docx_path = await write_md_to_word(report, filename)
                    # Returning the path of saved report files
                    md_path = await write_text_to_md(report, filename)
                    await websocket.send_json({"type": "path", "output": {"pdf": pdf_path, "docx": docx_path, "md": md_path}})
                else:
                    print("Error: not enough parameters provided.")

    except WebSocketDisconnect:
        await manager.disconnect(websocket)



# File: __init__.py



# File: README.md

## Detailed Reports

Introducing long and detailed reports, with a completely new architecture inspired by the latest [STORM](https://arxiv.org/abs/2402.14207) paper.

In this method we do the following:

1. Trigger Initial GPT Researcher report based on task
2. Generate subtopics from research summary
3. For each subtopic the headers of the subtopic report are extracted and accumulated
4. For each subtopic a report is generated making sure that any information about the headers accumulated until now are not re-generated.
5. An additional introduction section is written along with a table of contents constructed from the entire report.
6. The final report is constructed by appending these : Intro + Table of contents + Subsection reports

# File: detailed_report.py

import asyncio

from fastapi import WebSocket

from gpt_researcher.master.agent import GPTResearcher
from gpt_researcher.master.functions import (add_source_urls, extract_headers,
                                             table_of_contents)


class DetailedReport():
    def __init__(self, query: str, source_urls, config_path: str, websocket: WebSocket, subtopics=[]):
        self.query = query
        self.source_urls = source_urls
        self.config_path = config_path
        self.websocket = websocket
        self.subtopics = subtopics
        
        # A parent task assistant. Adding research_report as default
        self.main_task_assistant = GPTResearcher(self.query, "research_report", self.source_urls, self.config_path, self.websocket)

        self.existing_headers = []
        # This is a global variable to store the entire context accumulated at any point through searching and scraping
        self.global_context = []
    
        # This is a global variable to store the entire url list accumulated at any point through searching and scraping
        if self.source_urls:
            self.global_urls = set(self.source_urls)

    async def run(self):

        # Conduct initial research using the main assistant
        await self._initial_research()

        # Get list of all subtopics
        subtopics = await self._get_all_subtopics()
        
        # Generate report introduction
        report_introduction = await self.main_task_assistant.write_introduction()

        # Generate the subtopic reports based on the subtopics gathered
        _, report_body = await self._generate_subtopic_reports(subtopics)

        # Construct the final list of visited urls
        self.main_task_assistant.visited_urls.update(self.global_urls)

        # Construct the final detailed report (Optionally add more details to the report)
        report = await self._construct_detailed_report(report_introduction, report_body)

        return report

    async def _initial_research(self):
        # Conduct research using the main task assistant to gather content for generating subtopics
        await self.main_task_assistant.conduct_research()
        # Update context of the global context variable
        self.global_context = self.main_task_assistant.context
        # Update url list of the global list variable
        self.global_urls = self.main_task_assistant.visited_urls

    async def _get_all_subtopics(self) -> list:
        subtopics = await self.main_task_assistant.get_subtopics()
        return subtopics.dict()["subtopics"]

    async def _generate_subtopic_reports(self, subtopics: list) -> tuple:
        subtopic_reports = []
        subtopics_report_body = ""

        async def fetch_report(subtopic):

            subtopic_report = await self._get_subtopic_report(subtopic)

            return {
                "topic": subtopic,
                "report": subtopic_report
            }

        # This is the asyncio version of the same code below
        # Although this will definitely run faster, the problem
        # lies in avoiding duplicate information.
        # To solve this the headers from previous subtopic reports are extracted
        # and passed to the next subtopic report generation.
        # This is only possible to do sequentially

        # tasks = [fetch_report(subtopic) for subtopic in subtopics]
        # results = await asyncio.gather(*tasks)

        # for result in filter(lambda r: r["report"], results):
        #     subtopic_reports.append(result)
        #     subtopics_report_body += "\n\n\n" + result["report"]

        for subtopic in subtopics:
            result = await fetch_report(subtopic)
            if result["report"]:
                subtopic_reports.append(result)
                subtopics_report_body += "\n\n\n" + result["report"]

        return subtopic_reports, subtopics_report_body

    async def _get_subtopic_report(self, subtopic: dict) -> tuple:
        current_subtopic_task = subtopic.get("task")
        subtopic_assistant = GPTResearcher(
            query=current_subtopic_task,
            report_type="subtopic_report",
            websocket=self.websocket,
            parent_query=self.query,
            subtopics=self.subtopics,
            visited_urls=self.global_urls,
            agent=self.main_task_assistant.agent,
            role=self.main_task_assistant.role
        )

        # The subtopics should start research from the context gathered till now
        subtopic_assistant.context = list(set(self.global_context))

        # Conduct research on the subtopic
        await subtopic_assistant.conduct_research()

        # Here the headers gathered from previous subtopic reports are passed to the write report function
        # The LLM is later instructed to avoid generating any information relating to these headers as they have already been generated
        subtopic_report = await subtopic_assistant.write_report(self.existing_headers)

        # Update context of the global context variable
        self.global_context = list(set(subtopic_assistant.context))
        # Update url list of the global list variable
        self.global_urls.update(subtopic_assistant.visited_urls)

        # After a subtopic report has been generated then append the headers of the report to existing headers
        self.existing_headers.append(
            {
                "subtopic task": current_subtopic_task,
                "headers": extract_headers(subtopic_report),
            }
        )

        return subtopic_report

    async def _construct_detailed_report(self, introduction: str, report_body: str):
        # Generating a table of contents from report headers
        toc = table_of_contents(report_body)
        
        # Concatenating all source urls at the end of the report
        report_with_references = add_source_urls(report_body, self.main_task_assistant.visited_urls)
        
        return f"{introduction}\n\n{toc}\n\n{report_with_references}"

# File: __init__.py

from .basic_report.basic_report import BasicReport
from .detailed_report.detailed_report import DetailedReport

__all__ = [
    "BasicReport",
    "DetailedReport"
]

# File: basic_report.py

from gpt_researcher.master.agent import GPTResearcher
from fastapi import WebSocket

class BasicReport():
    def __init__(self, query: str, report_type: str, source_urls, config_path: str, websocket: WebSocket):
        self.query = query
        self.report_type = report_type
        self.source_urls = source_urls
        self.config_path = config_path
        self.websocket = websocket
        
    async def run(self):
        # Initialize researcher
        researcher = GPTResearcher(self.query, self.report_type, self.source_urls, self.config_path, self.websocket)
        
        # Run research
        await researcher.conduct_research()
        
        # and generate report        
        report = await researcher.write_report()
        
        return report

# File: __init__.py



# File: __init__.py



# File: utils.py

import aiofiles
import urllib
import uuid
from md2pdf.core import md2pdf
import mistune
from docx import Document
from htmldocx import HtmlToDocx

async def write_to_file(filename: str, text: str) -> None:
    """Asynchronously write text to a file in UTF-8 encoding.

    Args:
        filename (str): The filename to write to.
        text (str): The text to write.
    """
    # Convert text to UTF-8, replacing any problematic characters
    text_utf8 = text.encode('utf-8', errors='replace').decode('utf-8')

    async with aiofiles.open(filename, "w", encoding='utf-8') as file:
        await file.write(text_utf8)

async def write_text_to_md(text: str, filename: str = "") -> str:
    """Writes text to a Markdown file and returns the file path.

    Args:
        text (str): Text to write to the Markdown file.

    Returns:
        str: The file path of the generated Markdown file.
    """
    file_path = f"outputs/{filename}.md"
    await write_to_file(file_path, text)
    return file_path

async def write_md_to_pdf(text: str, filename: str = "") -> str:
    """Converts Markdown text to a PDF file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated PDF.
    """
    file_path = f"outputs/{filename}.pdf"

    try:
        md2pdf(file_path,
               md_content=text,
               #md_file_path=f"{file_path}.md",
               css_file_path="./frontend/pdf_styles.css",
               base_url=None)
        print(f"Report written to {file_path}.pdf")
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e}")
        return ""

    encoded_file_path = urllib.parse.quote(file_path)
    return encoded_file_path

async def write_md_to_word(text: str, filename: str = "") -> str:
    """Converts Markdown text to a DOCX file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated DOCX.
    """
    file_path = f"outputs/{filename}.docx"

    try:
        # Convert report markdown to HTML
        html = mistune.html(text)
        # Create a document object
        doc = Document()
        # Convert the html generated from the report to document format
        HtmlToDocx().add_html_to_document(html, doc)

        # Saving the docx document to file_path
        doc.save(file_path)
        
        print(f"Report written to {file_path}")

        encoded_file_path = urllib.parse.quote(file_path)
        return encoded_file_path
    
    except Exception as e:
        print(f"Error in converting Markdown to DOCX: {e}")
        return ""


# File: websocket_manager.py

# connect any client to gpt-researcher using websocket
import asyncio
import datetime
from typing import Dict, List

from fastapi import WebSocket

from backend.report_type import BasicReport, DetailedReport

from gpt_researcher.utils.enum import ReportType


class WebSocketManager:
    """Manage websockets"""

    def __init__(self):
        """Initialize the WebSocketManager class."""
        self.active_connections: List[WebSocket] = []
        self.sender_tasks: Dict[WebSocket, asyncio.Task] = {}
        self.message_queues: Dict[WebSocket, asyncio.Queue] = {}

    async def start_sender(self, websocket: WebSocket):
        """Start the sender task."""
        queue = self.message_queues.get(websocket)
        if not queue:
            return

        while True:
            message = await queue.get()
            if websocket in self.active_connections:
                try:
                    await websocket.send_text(message)
                except:
                    break
            else:
                break

    async def connect(self, websocket: WebSocket):
        """Connect a websocket."""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.message_queues[websocket] = asyncio.Queue()
        self.sender_tasks[websocket] = asyncio.create_task(
            self.start_sender(websocket))

    async def disconnect(self, websocket: WebSocket):
        """Disconnect a websocket."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            self.sender_tasks[websocket].cancel()
            await self.message_queues[websocket].put(None)
            del self.sender_tasks[websocket]
            del self.message_queues[websocket]

    async def start_streaming(self, task, report_type, websocket):
        """Start streaming the output."""
        report = await run_agent(task, report_type, websocket)
        return report


async def run_agent(task, report_type, websocket):
    """Run the agent."""
    # measure time
    start_time = datetime.datetime.now()
    # add customized JSON config file path here
    config_path = ""
    # Instead of running the agent directly run it through the different report type classes
    if report_type == ReportType.DetailedReport.value:
        researcher = DetailedReport(query=task, source_urls=None, config_path=config_path, websocket=websocket)
    else:
        researcher = BasicReport(query=task, report_type=report_type,
                                 source_urls=None, config_path=config_path, websocket=websocket)

    report = await researcher.run()
    # measure time
    end_time = datetime.datetime.now()
    await websocket.send_json({"type": "logs", "output": f"\nTotal run time: {end_time - start_time}\n"})

    return report
