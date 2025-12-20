#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean AI-generated comments from code files
Remove emojis, overly friendly comments, and excessive documentation
Keep only essential technical comments
"""

import os
import re
import sys


def clean_file(filepath):
    """Clean AI-style comments from a Python file"""

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # Remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    content = emoji_pattern.sub('', content)

    # Remove overly friendly comment patterns
    patterns_to_remove = [
        r'# 【.*?】',  # Chinese brackets
        r'# ============.*?============\n',  # Decorative lines
        r'# ━━━.*?━━━\n',
        r'# ───.*?───\n',
        r'# ===.*?===\n(?!# )',  # Only if not followed by another comment
        r'# 💡.*?\n',  # Light bulb comments
        r'# ⚠️.*?\n',  # Warning emoji
        r'# 注意：.*?\n',
        r'# 重要：.*?\n',
        r'# TIPS:.*?\n',
        r'# NOTE:.*?\n(?=\n)',  # Only standalone NOTE
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content)

    # Simplify verbose Chinese comments
    verbose_replacements = [
        (r'# 核心功能：\n', ''),
        (r'# 主要步骤：\n', ''),
        (r'# 详细说明：\n', ''),
        (r'# (\d+)\. ', r'# '),  # Remove numbering
    ]

    for pattern, replacement in verbose_replacements:
        content = re.sub(pattern, replacement, content)

    # Remove excessive blank lines (more than 2)
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # Remove standalone comment blocks with only decorations
    content = re.sub(r'^# [=\-─━]{3,}$', '', content, flags=re.MULTILINE)

    return content if content != original_content else None


def main():
    """Clean all Python files in the project"""

    # Directories to clean
    dirs_to_clean = [
        'models',
        'algorithms',
        'retrieval',
        'envs',
        'data',
        'utils',
    ]

    # Also clean training scripts
    files_to_clean = [
        'train.py',
        'train_local_simplified.py',
        'train_5090_optimized.py',
        'test_training.py',
        'test_setup.py',
    ]

    cleaned_count = 0

    # Clean directory files
    for dir_name in dirs_to_clean:
        if not os.path.exists(dir_name):
            continue

        for filename in os.listdir(dir_name):
            if not filename.endswith('.py'):
                continue

            filepath = os.path.join(dir_name, filename)
            print(f"Cleaning {filepath}...")

            cleaned_content = clean_file(filepath)
            if cleaned_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                cleaned_count += 1
                print(f"  [OK] Cleaned")
            else:
                print(f"  [SKIP] No changes needed")

    # Clean individual files
    for filename in files_to_clean:
        if not os.path.exists(filename):
            continue

        print(f"Cleaning {filename}...")
        cleaned_content = clean_file(filename)
        if cleaned_content:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            cleaned_count += 1
            print(f"  [OK] Cleaned")
        else:
            print(f"  [SKIP] No changes needed")

    print(f"\n{'='*50}")
    print(f"Cleaning completed!")
    print(f"Files cleaned: {cleaned_count}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
