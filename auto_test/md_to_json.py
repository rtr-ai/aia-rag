import re
import json

# File names
MARKDOWN_FILE = 'data/fragen-qa.md'
JSON_FILE = 'true_results.json'

def expand_source(source):
    """
    If the source text contains a 'bis' indicating a range of articles, e.g.
    "Art 8 bis 15 AIA", expand it into individual articles.
    Otherwise, return a list containing the source itself.
    """
    # Look for a pattern like "Art <num> bis <num>"
    match = re.search(r"Art\s+(\d+)\s+bis\s+(\d+)", source)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
        # Return individual articles like "Art 8", "Art 9", ... "Art 15"
        return [f"Art {i}" for i in range(start, end + 1)]
    else:
        return [source]

def process_sources(sources):
    """
    Process the list of raw source strings:
    
    1. Expand any "bis" range entries.
    2. For article references (lines starting with "Art") that are not exempt
       (i.e. not starting with "Art 3:" or "Art 5:"), combine multiple references 
       with the same article number into one simplified reference "Art <number>".
    3. For other sources (or those that start with "Art 3:" or "Art 5:"), keep them as is.
    """
    result = []
    # For combining article references, keep track of which article numbers we have added.
    seen_articles = set()
    
    for source in sources:
        # Expand if there is a "bis" range in the source.
        expanded_items = expand_source(source)
        for item in expanded_items:
            item = item.strip()
            # If the item looks like an article reference:
            if item.startswith("Art"):
                # Check if it is exempt from combining. We consider an exemption
                # if the text immediately after "Art" is "3:" or "5:".
                if re.match(r"Art\s+3:", item) or re.match(r"Art\s+5:", item):
                    # Do not combine: simply append the full text.
                    result.append(item)
                else:
                    # Otherwise, extract the article number.
                    num_match = re.match(r"Art\s+(\d+)", item)
                    if num_match:
                        art_num = num_match.group(1)
                        simplified = f"Art {art_num}"
                        if simplified not in seen_articles:
                            seen_articles.add(simplified)
                            result.append(simplified)
                    else:
                        # If for some reason it doesn't match the pattern, append as is.
                        result.append(item)
            else:
                # For non-article references (e.g. "KI-Servicestelle: ..."), just add.
                result.append(item)
    return result

def parse_markdown(md_text):
    """
    Parses the markdown text and returns a dictionary mapping each question to its list
    of processed source strings.
    """
    qa_dict = {}
    # Split into lines
    lines = md_text.splitlines()

    current_question = None
    collecting_sources = False
    current_sources = []
    
    for line in lines:
        line = line.strip()
        # New question header line: starts with '#' followed by a number and dot.
        if line.startswith('#'):
            # If a previous question block is done, process its sources.
            if current_question is not None:
                qa_dict[current_question] = process_sources(current_sources)
                current_sources = []
                collecting_sources = False
            # Remove the markdown header and any leading numbering.
            m = re.match(r'#\s*\d+\.\s*(.*)', line)
            if m:
                current_question = m.group(1).strip()
            else:
                current_question = line.lstrip('#').strip()
        elif line.startswith("Quellen:"):
            collecting_sources = True
        elif collecting_sources:
            # Expect source lines starting with [number]
            if re.match(r'^\[\d+\]', line):
                # Remove the "[number]" prefix.
                source_text = re.sub(r'^\[\d+\]\s*', '', line)
                current_sources.append(source_text)
            elif line == "":
                continue
        # Other lines (like "Antwort:") are skipped.
    
    # Add the last question block if any.
    if current_question is not None:
        qa_dict[current_question] = process_sources(current_sources)
    
    return qa_dict

def main():
    # Read the markdown file.
    with open(MARKDOWN_FILE, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Parse the markdown to create the Q&A dictionary.
    qa_dict = parse_markdown(md_text)

    # Write the dictionary as nicely formatted JSON.
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(qa_dict, f, ensure_ascii=False, indent=4)

    print(f"Processed markdown and wrote JSON output to '{JSON_FILE}'.")

if __name__ == '__main__':
    main()