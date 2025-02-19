import json
from fuzzywuzzy import fuzz

# Load JSON files
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def find_missing_titles(combined_data, true_results):
    combined_titles = [chunk.get('title', '') for chunk in combined_data.get('chunks', []) if 'title' in chunk]
    missing_entries = {}
    
    for key, values in true_results.items():
        for value in values:
            if combined_titles:
                best_match = max(combined_titles, key=lambda title: fuzz.ratio(title, value))
                similarity = fuzz.ratio(best_match, value)
            else:
                best_match, similarity = "", 0
            
            if similarity < 90:  # Threshold for minor variations
                if key not in missing_entries:
                    missing_entries[key] = []
                missing_entries[key].append(value)
    
    return missing_entries

def main():
    combined_data = load_json('data/combined.json')
    true_results = load_json('auto_test/true_results.json')
    
    missing_entries = find_missing_titles(combined_data, true_results)
    
    print("Missing or different entries:")
    print(json.dumps(missing_entries, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
