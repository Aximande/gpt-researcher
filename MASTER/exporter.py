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
    convert_to_markdown(repo_path / "README.md", final_md_folder / "README.md")
    convert_to_markdown(repo_path / "LICENSE", final_md_folder / "LICENSE.md")
    convert_to_markdown(repo_path / "cli.py", final_md_folder / "cli.md")
    convert_to_markdown(repo_path / "main.py", final_md_folder / "main.md")
    convert_to_markdown(repo_path / "requirements.txt", final_md_folder / "requirements.md")
    convert_to_markdown(repo_path / "Dockerfile", final_md_folder / "Dockerfile.md")

    # Process and convert files in the docs folder and its subfolders
    docs_subfolders = ["blog", "docs", "src", "static"]
    for subfolder in docs_subfolders:
        output_file = final_md_folder / f"docs_{subfolder.replace('/', '_')}.md"
        process_directory(repo_path / "docs" / subfolder, output_file)

    # Process and convert files in the specified folders
    folders_to_process = [
        "backend", "examples", "frontend", "scraping",
        "gpt_researcher/config", "gpt_researcher/context", "gpt_researcher/llm_provider",
        "gpt_researcher/master", "gpt_researcher/memory", "gpt_researcher/retrievers",
        "gpt_researcher/scraper", "gpt_researcher/utils"
    ]
    for folder in folders_to_process:
        output_file = final_md_folder / f"{folder.replace('/', '_')}.md"
        process_directory(repo_path / folder, output_file)

    # Merge setup fi
