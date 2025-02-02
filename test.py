import os
import re

def extract_class_names(folder_path):
    """
    Extracts folder names from a given directory and formats them into a list.

    Args:
        folder_path: The path to the directory containing the folders.

    Returns:
        A string representing the CLASS_NAMES list, or None if an error occurs.
    """
    try:
        class_names = [
            f
            for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f))
        ]
        # Sort for consistency (optional but recommended)
        class_names.sort()

        # Sanitize folder names to ensure they are valid Python identifiers.
        sanitized_names = []
        for name in class_names:
            sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", name)  # Replace invalid chars with _
            if not re.match(r"^[a-zA-Z_]", sanitized_name): #must start with letter or _
                sanitized_name = "_" + sanitized_name
            sanitized_names.append(sanitized_name)

        class_names_str = "CLASS_NAMES = ["
        for i, name in enumerate(sanitized_names):
            class_names_str += f"'{name}'"
            if i < len(sanitized_names) - 1:
                class_names_str += ", "
        class_names_str += "]"

        return class_names_str

    except FileNotFoundError:
        print(f"Error: Folder not found: {folder_path}")
        return None
    except OSError as e:
        print(f"OS Error: {e}")
        return None

# Example usage:
folder_path = r"C:\Users\Administrator\Documents\GitHub\Fruit-Image-Classifier\datasets\moltean\fruits\versions\11\fruits-360_dataset_100x100\fruits-360\Training\train"  # Replace with the actual path
class_names_output = extract_class_names(folder_path)

if class_names_output:
    print(class_names_output)
    # To use the list:
    exec(class_names_output) # Executes the string and creates the CLASS_NAMES variable
    print(CLASS_NAMES) # Now you can access the list