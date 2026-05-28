"""
import_knowledge.py — NeHe Productions / LLM Training Studio
Author: Jeff Molofee (aka NeHe) — 2026
================================================
Parses all Q&A knowledge files from the training_data/ folder.

Knowledge files use Q:/A: text format with a .txt extension:

    Q: Your question here?
    A: The answer here.

Both single-line and multi-line answers are supported:

    Q: What is the capital of France?
    A: Paris.

    Q: Apply BIO tags to this title.
    A: BTS -> B-ART
    - -> O
    Dynamite -> B-SONG

Pairs may be separated by blank lines (standard) or run back-to-back with
no blank line between them (as in the BIO sequence files):

    Q: Tag this title.
    A: BTS -> B-ART
    - -> O
    Q: Tag this title.           <- no blank line before next Q:
    A: TWICE -> B-ART
    ...

The manual/manual_entries.txt file is edited via the Training Studio UI
and is also picked up by collect_all_files().

Usage:
    python train/import_knowledge.py           # scan and report
    python train/import_knowledge.py --dry-run # preview only
"""

import os
import re
import argparse

ROOT              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_KNOWLEDGE = os.path.join(ROOT, 'knowledge')


def parse_qa_file(filepath: str) -> list:
    """Parse a Q:/A: knowledge file into list of {'question', 'answer'} dicts.

    Supports single-line and multi-line answers.
    Pairs may be separated by blank lines OR run back-to-back with no gap —
    the splitter detects a new Q: at the start of any line regardless.
    """
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f'  [WARN] Could not read {filepath}: {e}')
        return []

    pairs = []

    # Split on any newline immediately followed by Q:
    # Handles both blank-line-separated pairs (\n\nQ:) and
    # back-to-back pairs with no blank line (\nQ:) as in the BIO sequences file.
    blocks = re.split(r'\n(?=Q:)', content.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Capture Q: question and A: answer (re.DOTALL for multi-line answers)
        m = re.match(r'^Q:\s*(.+?)\s*\nA:\s*(.+)$', block, re.DOTALL)
        if m:
            question = m.group(1).strip()
            answer   = m.group(2).strip()
            if question and answer:
                pairs.append({'question': question, 'answer': answer})

    return pairs


def collect_all_files(knowledge_dir: str) -> list:
    """Return list of (relative_path, absolute_path) for all knowledge files."""
    result = []
    for root, dirs, files in os.walk(knowledge_dir):
        dirs.sort()
        for fname in sorted(files):
            abs_path = os.path.join(root, fname)
            if fname.endswith('.txt'):
                rel_path = os.path.relpath(abs_path, knowledge_dir).replace('\\', '/')
                result.append((rel_path, abs_path))
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Scan knowledge files and report Q&A pair counts'
    )
    parser.add_argument('--knowledge-dir', default=DEFAULT_KNOWLEDGE,
                        help=f'Path to knowledge/ folder (default: {DEFAULT_KNOWLEDGE})')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview pairs without building dataset')
    args = parser.parse_args()

    print('=' * 60)
    print('  LLM Training Studio - Knowledge Base Scanner')
    print('=' * 60 + '\n')
    print(f'  Knowledge dir : {args.knowledge_dir}\n')

    if not os.path.isdir(args.knowledge_dir):
        print(f'[ERROR] Directory not found: {args.knowledge_dir}')
        return

    files = collect_all_files(args.knowledge_dir)
    print(f'  Found {len(files)} knowledge files\n')

    total = 0
    for rel_path, abs_path in files:
        pairs = parse_qa_file(abs_path)
        total += len(pairs)
        print(f'  {rel_path:55s} -> {len(pairs)} pairs')

    print(f'\n  Total: {total} Q&A pairs across {len(files)} files')

    if args.dry_run and files:
        print('\n  [DRY RUN] First 5 pairs from first file:')
        pairs = parse_qa_file(files[0][1])
        for p in pairs[:5]:
            print(f'\n  Q: {p["question"]}')
            a = p["answer"]
            print(f'  A: {a[:100]}...' if len(a) > 100 else f'  A: {a}')


if __name__ == '__main__':
    main()
