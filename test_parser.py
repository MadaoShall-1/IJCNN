# test_parser.py

import json
from parser import parse_question


examples = [
    "A 2 kg object accelerates at 3 m/s². What is the net force?",
    "Calculate the equivalent resistance of the following circuit, given that each resistor has a resistance of r.",
    "A circuit has a voltage of 12 V and a current of 2 A. What is the resistance?",
    "Which option correctly gives the equivalent resistance? A. 3r B. r/3 C. 2r D. r",
    "A block slides down a frictionless incline and compresses a spring. What is the maximum compression?"
]


for q in examples:
    print("=" * 100)
    print("QUESTION:", q)
    parsed = parse_question(
        q,
        use_qwen=True,
        load_4bit=True,
    )
    print(json.dumps(parsed, indent=2, ensure_ascii=False))