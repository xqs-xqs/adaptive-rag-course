import os
import re
import logging
from typing import Optional, List

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Regex to match key-value pairs wrapped in double quotes
KV_PATTERN = re.compile(r'^"(.+?)"\s*:\s*"(.*)"$')

# Mapping from keywords (lowercase) to standard fields.
# Order matters if a text key might contain multiple keywords. We'll iterate through mapping.
# The user specified "优先精确匹配，多个关键词命中同一标准字段时取第一个。"
# We will use a list of tuples to define the priority.
FIELD_MAPPINGS = [
    (["subject code"], "course_code"),
    (["subject title"], "course_title"),
    (["credit value"], "credits"),
    (["level"], "level"),
    (["pre-requisite", "exclusion", "prerequisite"], "prerequisites"),
    (["objectives"], "objectives"),
    (["intended learning outcomes", "learning outcomes"], "learning_outcomes"),
    (["subject synopsis", "indicative syllabus", "syllabus"], "syllabus"),
    (["teaching/learning methodology", "teaching methodology"], "teaching_methodology"),
    (["assessment methods", "assessment"], "assessment"),
    (["student study effort"], "study_effort"),
    (["reading list", "references"], "references"),
    (["class time", "schedule"], "class_time"),
]

def clean_value(val: str) -> str:
    """Clean the value by stripping and replacing multiple newlines with a single space."""
    val = val.strip()
    # Replace literal \n or \r sequences with a space
    val = re.sub(r'\\n|\\r', ' ', val)
    # Replace multiple spaces with a single space
    val = re.sub(r'\s+', ' ', val)
    return val.strip()

def parse_course_txt(filepath: str) -> Optional[dict]:
    """
    Parse a single course txt file and return a structured dictionary.
    Returns None if the file cannot be read or parsed completely.
    """
    try:
        parsed_data = {}
        # Initialize standard fields with default values
        standard_fields = {
            "course_code": "",
            "course_title": "",
            "credits": 0,
            "level": 0,
            "prerequisites": "",
            "objectives": "",
            "learning_outcomes": "",
            "syllabus": "",
            "teaching_methodology": "",
            "assessment": "",
            "study_effort": "",
            "references": "",
            "class_time": ""
        }
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                
                match = KV_PATTERN.match(line)
                if not match:
                    continue
                
                raw_key = match.group(1).strip()
                raw_val = match.group(2)
                
                cleaned_val = clean_value(raw_val)
                lower_key = raw_key.lower()
                
                # Find matching standard field
                matched_field = None
                
                # First pass: Exact match
                for keywords, std_field in FIELD_MAPPINGS:
                    if any(kw == lower_key for kw in keywords):
                        matched_field = std_field
                        break
                        
                # Second pass: Fuzzy match (find earliest occurring keyword)
                if not matched_field:
                    best_index = len(lower_key)
                    for keywords, std_field in FIELD_MAPPINGS:
                        for kw in keywords:
                            idx = lower_key.find(kw)
                            if idx != -1 and idx < best_index:
                                best_index = idx
                                matched_field = std_field
                
                if matched_field:
                    if matched_field not in parsed_data:
                        if matched_field == "course_code":
                            cleaned_val = cleaned_val.replace(" ", "")
                        elif matched_field in ["credits", "level"]:
                            try:
                                cleaned_val = int(cleaned_val)
                            except ValueError:
                                cleaned_val = 0
                        parsed_data[matched_field] = cleaned_val
                else:
                    # Unmatched key, keep as is
                    if raw_key not in parsed_data:
                        parsed_data[raw_key] = cleaned_val
        
        # Merge standard fields with parsed data (filling missing ones with defaults)
        for k, v in standard_fields.items():
            if k not in parsed_data:
                parsed_data[k] = v
                
        return parsed_data
        
    except Exception as e:
        logging.warning(f"Failed to parse {filepath}: {e}")
        return None

def parse_all_txts(directory: str) -> List[dict]:
    """
    Parse all .txt files in the given directory.
    """
    results = []
    if not os.path.isdir(directory):
        logging.warning(f"Directory not found: {directory}")
        return results
        
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if not os.path.isfile(filepath):
            continue
            
        if not filename.lower().endswith(".txt"):
            logging.info(f"Skipping non-txt file: {filename}")
            continue
            
        parsed = parse_course_txt(filepath)
        if parsed is not None:
            results.append(parsed)
            
    return results
