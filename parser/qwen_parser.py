# parser/qwen_parser.py

import json
import re
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .schema import ParsedQuestion


MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


class QwenStructuredParser:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        load_4bit: bool = True,
        max_new_tokens: int = 700,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        if load_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )

        self.model.eval()

    def build_prompt(self, question: str, rule_hints: Dict[str, Any]) -> str:
        schema_description = {
            "domain": "physics",
            "topic": "one of: electric_circuit, dynamics, kinematics, energy, momentum, waves, thermodynamics, optics, unknown",
            "subtopic": "specific subtopic or unknown",
            "question_type": "calculation, multiple_choice, yes_no_uncertain, open_ended",
            "target_quantity": "quantity being asked for, e.g. equivalent_resistance, net_force, voltage",
            "known_variables": {"variable_name": "value with unit if available"},
            "unknown_variables": ["symbols to solve for"],
            "answer_type": "numeric_value, symbolic_expression, multiple_choice, yes_no_uncertain, open_ended",
            "unit_expected": "expected unit, e.g. N, V, A, ohm_or_symbolic_resistance, J",
            "requires_diagram_reasoning": "boolean",
            "requires_formula_retrieval": "boolean",
            "answer_options": "object like {'A':'...', 'B':'...'} or null",
            "implicit_conditions": ["only conditions clearly implied by the question"],
            "physical_relations": [
                "relations explicitly stated in the question text only; do not include general formulas or inferred laws"
            ],
            "parser_confidence": "number from 0 to 1"
        }

        return f"""
You are a physics question parser.

Your task is to convert the problem into a structured JSON object.
Do NOT solve the problem.
Do NOT compute the final answer.
Only extract information explicitly stated or strongly implied by the problem.
Do not invent numerical values.
For physical_relations, use a stricter rule: include only relationships explicitly written in the question text.
Do not put general formulas, inferred laws, solution formulas, or background physics facts in physical_relations.
If the question does not explicitly state a relationship, physical_relations must be [].
Return ONLY valid JSON. Do not wrap it in markdown.

Required JSON schema:
{json.dumps(schema_description, indent=2)}

Question:
{question}

Rule-based hints:
{json.dumps(rule_hints, indent=2)}

Return only the JSON object.
""".strip()

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        """
        Extract the first JSON object from model output.
        This is needed because small open-source models may add extra text.
        """
        text = text.strip()

        # Direct parse first.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Remove markdown code fence if present.
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract first {...} block.
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"No JSON object found in model output:\n{text}")

        json_str = text[start:end + 1]
        return json.loads(json_str)

    def parse(self, question: str, rule_hints: Dict[str, Any]) -> ParsedQuestion:
        prompt = self.build_prompt(question, rule_hints)

        messages = [
            {
                "role": "system",
                "content": "You are a strict JSON parser for physics problems."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            [text],
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
        output_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        data = self._extract_json_object(output_text)

        return ParsedQuestion(**data)
