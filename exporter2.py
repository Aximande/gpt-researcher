import os
from pathlib import Path
import subprocess

# Function to convert a file to markdown using Pandoc
def convert_to_markdown(file_path, output_path):
    subprocess.run(["pandoc", "-s", "-o", output_path, file_path])

# Function to process a directory and its files recursively
def process_directory(directory, output_file):
    for item in directory.iterdir():
        if item.is_file():
            if item.suffix in [".py", ".txt", ".toml", ".css", ".js", ".html", ".yml", ".json", ".md"]:
                with open(output_file, "a") as f:
                    f.write(f"\n\n# File: {item.name}\n\n")
                    with open(item, "r") as file_content:
                        f.write(file_content.read())
        elif item.is_dir():
            process_directory(item, output_file)

# Main script
if __name__ == "__main__":
    repo_path = Path("/Users/ladislas/Desktop/code/gpt-researcher/gpt-researcher")
    master_folder = repo_path / "MASTER"
    final_md_folder = master_folder / "final_md_files"
    final_md_folder.mkdir(parents=True, exist_ok=True)

    # Process and convert individual files
    convert_to_markdown(repo_path / "README.md", final_md_folder / "PROJECT_OVERVIEW.md")
    convert_to_markdown(repo_path / "LICENSE", final_md_folder / "PROJECT_OVERVIEW.md")
    convert_to_markdown(repo_path / "cli.py", final_md_folder / "SCRIPTS.md")
    convert_to_markdown(repo_path / "main.py", final_md_folder / "SCRIPTS.md")
    convert_to_markdown(repo_path / "requirements.txt", final_md_folder / "SETUP.md")
    convert_to_markdown(repo_path / "Dockerfile", final_md_folder / "SETUP.md")

    # Process and convert files in the docs folder and its subfolders
    docs_output_file = final_md_folder / "DOCS.md"
    process_directory(repo_path / "docs", docs_output_file)

    # Process and convert files in the specified folders
    gpt_researcher_output_file = final_md_folder / "GPT_RESEARCHER.md"
    folders_to_process = [
        "gpt_researcher/config", "gpt_researcher/context", "gpt_researcher/llm_provider",
        "gpt_researcher/master", "gpt_researcher/memory", "gpt_researcher/retrievers",
        "gpt_researcher/scraper", "gpt_researcher/utils"
    ]
    for folder in folders_to_process:
        process_directory(repo_path / folder, gpt_researcher_output_file)

    backend_output_file = final_md_folder / "BACKEND.md"
    process_directory(repo_path / "backend", backend_output_file)

    frontend_output_file = final_md_folder / "FRONTEND.md"
    process_directory(repo_path / "frontend", frontend_output_file)

    scraping_output_file = final_md_folder / "SCRAPING.md"
    process_directory(repo_path / "scraping", scraping_output_file)

    examples_output_file = final_md_folder / "EXAMPLES.md"
    process_directory(repo_path / "examples", examples_output_file)

    # Merge setup files
    setup_files = ["pyproject.toml", "poetry.lock", "poetry.toml"]
    with open(final_md_folder / "SETUP.md", "a") as setup_file:
        for file_name in setup_files:
            file_path = repo_path / file_name
            if file_path.exists():
                setup_file.write(f"## {file_name}\n\n")
                with open(file_path, "r") as f:
                    setup_file.write(f.read() + "\n\n")
