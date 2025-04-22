import os
import re

# Root folder of the project
PROJECT_ROOT = 'vllm'

# Directories to ignore
IGNORE_DIRS = {'venv', '.git', '__pycache__', '.idea', '.vscode'}

# Special comment for easy search and cleanup
AUTO_COMMENT = '# added by auto-decorator-script'

# Regex pattern to find class definitions
class_pattern = re.compile(r'^(\s*)class\s+\w+\s*[\(:]')

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    has_class = any('class ' in line for line in lines)
    if not has_class:
        return
    
    last_import_pos = 0
    has_future = any('__future__' in line for line in lines)
    
    found_comment = False
    found_future = False
    for i, line in enumerate(lines):
        if "__future__" in line:
            found_future = True

        if line.count('"""') == 1:
            found_comment = False if found_comment else True 

        if not found_comment:
            if line == "\n": 
                if not has_future:
                    last_import_pos = i
                    break
                else:
                    if found_future:
                        last_import_pos = i
                        break


    # Check if the import is already present
    has_import = any('from vllm.my_utils import decorate_all_methods' in line for line in lines)
    
    new_lines = []
    for i, line in enumerate(lines):
        # Check for class definitions
        match = class_pattern.match(line)
        
        if i == last_import_pos:
            if not has_import and has_class:
                # Add the import at the top of the file with a comment
                new_lines.append(f'from vllm.my_utils import decorate_all_methods, profile_function {AUTO_COMMENT}\n')
                
        # Check for class definitions
        match = class_pattern.match(line)
        if match:
            indent = match.group(1)  # Capture leading spaces
            # Add decorator before the class with a comment
            if "dataclass(frozen=True)" not in new_lines[-1]: 
                new_lines.append(f'{indent}@decorate_all_methods(profile_function) {AUTO_COMMENT}\n')
            else:
                print("frozen ", line)
        new_lines.append(line)

    # Overwrite the file with modified content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

def walk_project(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # Filter out ignored directories
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]

        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                process_file(filepath)

if __name__ == '__main__':
    walk_project(PROJECT_ROOT)
