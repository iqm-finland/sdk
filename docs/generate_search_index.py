import os
import json
import sys
from bs4 import BeautifulSoup

def extract_text_from_html(file_path):
    """Extracts text from <p> elements in an HTML file."""
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

        # Extract text from headers and paragraphs
        content = " ".join(
            tag.get_text(" ", strip=True) for tag in soup.find_all(["p"])
        )

        title_tag = soup.find("h1")
        title = title_tag.get_text(" ", strip=True) if title_tag else "Untitled"

        return content, title

def build_search_index(docs_dir="./public/", output_file="./search.json", version_name=None):
    """Builds a JSON search index from documentation HTML files."""
    search_index = []
    excluded_file_names = ["genindex.html", "license.html", "search.html", "changelog.html", "py-modindex.html"]

    if version_name:
        # For versioned docs, scan all packages in the version directory
        if os.path.exists(docs_dir):
            for item in os.listdir(docs_dir):
                package_path = os.path.join(docs_dir, item)

                if not os.path.isdir(package_path):
                    continue

                if item.startswith(('.', '_')):
                    continue

                print(f'Processing package: {item}')
                for root, dirs, files in os.walk(package_path):
                    dirs[:] = [d for d in dirs if not d.startswith(('_', '.'))]

                    for file in files:
                        if file.endswith('.html') and file not in excluded_file_names:
                            file_path = os.path.join(root, file)
                            try:
                                content, title = extract_text_from_html(file_path)
                                relative_url = os.path.relpath(file_path, docs_dir)

                                search_index.append({
                                    'package': item,
                                    'title': title,
                                    'url': f'/{version_name}/{relative_url}',
                                    'description': content[:200] + '...',
                                    'content': content
                                })
                            except Exception as e:
                                print(f'Error processing {file_path}: {e}')
    else:
        # For main version, use the package directories from environment or defaults
        try:
            package_dirs = os.getenv("PACKAGES").split()
        except:
            # Default package directories
            package_dirs = ["iqm-exa-common", "iqm-pulla", "iqm-pulse", "iqm-station-control-client", "iqm-qaoa", "iqm-data-definitions", "iqm-benchmarks", "iqm-client", "iqm-qubit-selector"]

        for package in package_dirs:
            package_path = os.path.join(docs_dir, package)

            if not os.path.exists(package_path):
                print(f"⚠️ Warning: Directory {package_path} not found. Skipping...")
                continue

            for root, dirs, files in os.walk(package_path):
                dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]

                for file in files:
                    if file.endswith(".html") and file not in excluded_file_names:
                        file_path = os.path.join(root, file)
                        content, title = extract_text_from_html(file_path)
                        relative_url = os.path.relpath(file_path, docs_dir)

                        search_index.append({
                            "package": package,
                            "title": title,
                            "url": f"/{relative_url}",
                            "description": content[:200] + "...",
                            "content": content
                        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(search_index, f, indent=2)

    print(f"✅ Search index generated at {output_file} with {len(search_index)} entries")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        version_name = sys.argv[1]
        docs_dir = sys.argv[2] if len(sys.argv) > 2 else f"./public/{version_name}/"
        output_file = sys.argv[3] if len(sys.argv) > 3 else f"./search_{version_name}.json"

        build_search_index(docs_dir, output_file, version_name)
    else:
        # Default behavior for main version
        build_search_index()
