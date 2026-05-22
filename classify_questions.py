from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASET = "Physics_Problems_Text_Only.xlsx"
DEFAULT_OUTPUT_DIR = "outputs/question_classification"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify questions in the physics dataset.")
    parser.add_argument("--data", default=DEFAULT_DATASET, help="Path to the Excel dataset.")
    parser.add_argument("--sheet", default=None, help="Optional sheet name. Defaults to the first sheet.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for classification outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def id_prefix(value: Any) -> str:
    text = clean_text(value).upper()
    match = re.match(r"([A-Z]+)", text)
    return match.group(1) if match else "UNKNOWN"


def has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def has_regex(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def answer_mode(answer: Any, unit: Any) -> str:
    answer_text = clean_text(answer).lower()
    unit_text = clean_text(unit)
    if not answer_text or not unit_text:
        return "unlabeled"
    if answer_text in {"yes", "no", "true", "false"}:
        return "boolean"
    if ";" in answer_text or ";" in unit_text:
        return "multi_value"
    if re.search(r"sqrt|\\sqrt", answer_text):
        return "symbolic"
    if re.search(r"(×|x|\*)\s*10|\be[+-]?\d+", answer_text, flags=re.IGNORECASE):
        return "scientific_notation"
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", answer_text):
        return "numeric"
    return "text"


def task_type(question: Any, answer: Any, unit: Any) -> str:
    text = clean_text(question).lower()
    mode = answer_mode(answer, unit)
    if not text:
        return "missing_question"
    if mode == "unlabeled":
        return "unlabeled_question"
    if mode == "boolean" or has_regex(text, r"^(does|do|will|can|should)\b") or "whether" in text or "determine if" in text:
        return "yes_no_judgment"
    if mode == "text":
        return "qualitative_explanation"
    if mode == "multi_value":
        return "multi_part_calculation"
    if has_any(text, ["ratio", "percentage", "relative error", "efficiency"]):
        return "ratio_or_percent_calculation"
    return "single_numeric_calculation"


def classify_topic(question: Any, cot: Any = "") -> dict[str, str]:
    question_text = clean_text(question)
    text = f"{question_text} {clean_text(cot)}".lower()
    if not question_text:
        return {
            "domain": "Data quality",
            "topic": "Missing question",
            "subtopic": "Missing question text",
            "confidence": "high",
            "matched_rules": "empty question",
        }

    rules: list[str] = []

    if has_any(text, ["rlc", "resonance", "resonant", "ac circuit", "angular frequency"]):
        rules.append("rlc/resonance")
        if has_any(text, ["does", "will", "whether", "occur"]):
            subtopic = "Resonance yes/no judgment"
        elif has_any(text, ["resistance", "impedance"]):
            subtopic = "RLC impedance/resistance"
        elif has_any(text, ["capacitance", "capacitor", "choose for the capacitor"]):
            subtopic = "RLC capacitance"
        elif has_any(text, ["inductance", "inductor"]):
            subtopic = "RLC inductance"
        elif has_any(text, ["frequency", "angular"]):
            subtopic = "RLC frequency"
        else:
            subtopic = "RLC resonance calculation"
        return {
            "domain": "Electricity and magnetism",
            "topic": "AC circuits and resonance",
            "subtopic": subtopic,
            "confidence": "high",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["capacitor", "capacitance", "parallel-plate", "dielectric", "permittivity", "electric field energy"]):
        rules.append("capacitor/capacitance")
        if has_any(text, ["dielectric", "permittivity", "parallel-plate", "plate separation", "area of each plate"]):
            subtopic = "Parallel-plate capacitor and dielectric"
        elif has_any(text, ["energy", "stored"]):
            subtopic = "Capacitor energy"
        elif has_any(text, ["charge", "fully charged", "stores q"]):
            subtopic = "Capacitor charge"
        elif has_any(text, ["potential difference", "voltage"]):
            subtopic = "Capacitor voltage"
        elif has_any(text, ["capacitance"]):
            subtopic = "Capacitance calculation"
        else:
            subtopic = "Capacitor calculation"
        return {
            "domain": "Electricity and magnetism",
            "topic": "Capacitors",
            "subtopic": subtopic,
            "confidence": "high",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["induced electromotive force", "emf", "electromotive force"]):
        rules.append("electromagnetic induction")
        return {
            "domain": "Electricity and magnetism",
            "topic": "Electromagnetic induction",
            "subtopic": "Induced EMF",
            "confidence": "medium",
            "matched_rules": "; ".join(rules),
        }

    if "efficiency" in text and has_any(text, ["circuit", "electrical energy", "magnetic energy"]):
        rules.append("electrical energy efficiency")
        return {
            "domain": "Electricity and magnetism",
            "topic": "Basic circuits",
            "subtopic": "Electrical energy efficiency",
            "confidence": "medium",
            "matched_rules": "; ".join(rules),
        }

    if (
        has_any(text, ["electric force", "electric forces", "resultant force"])
        and has_any(text, ["charge", "electric"])
    ):
        rules.append("electric force vector composition")
        return {
            "domain": "Electricity and magnetism",
            "topic": "Electrostatics",
            "subtopic": "Electric force vector composition",
            "confidence": "high",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["lc circuit", "simple harmonic", "oscillation", "oscillations", "spring pendulum", "amplitude", "initial phase", "higher-pitched sound"]):
        rules.append("oscillation/waves")
        if "lc circuit" in text:
            subtopic = "LC oscillation"
        elif has_any(text, ["sound", "higher-pitched", "pitch"]):
            subtopic = "Sound frequency"
        elif has_any(text, ["spring pendulum", "simple harmonic", "amplitude", "initial phase"]):
            subtopic = "Simple harmonic motion"
        else:
            subtopic = "Oscillation calculation"
        return {
            "domain": "Waves and oscillations",
            "topic": "Oscillations and waves",
            "subtopic": subtopic,
            "confidence": "medium",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["solenoid", "magnetic field", "magnetic flux", "magnetic field energy", "inductor", "inductance"]):
        rules.append("magnetism/inductor")
        if "solenoid" in text and has_any(text, ["magnetic field", "inside"]):
            subtopic = "Solenoid magnetic field"
        elif has_any(text, ["turns per meter", "turn density"]):
            subtopic = "Solenoid turn density"
        elif has_any(text, ["magnetic flux", "flux linkage"]):
            subtopic = "Magnetic flux"
        elif has_any(text, ["magnetic field energy", "energy stored", "inductor energy"]):
            subtopic = "Inductor energy"
        elif has_any(text, ["inductance", "inductor"]):
            subtopic = "Inductance"
        else:
            subtopic = "Magnetism calculation"
        return {
            "domain": "Electricity and magnetism",
            "topic": "Magnetism and inductors",
            "subtopic": subtopic,
            "confidence": "high",
            "matched_rules": "; ".join(rules),
        }

    if (
        has_any(text, ["point charge", "electric charge", "charges", "q1", "q2", "coulomb", "test charge"])
        or has_regex(text, r"\bq\s*=")
    ):
        rules.append("point charges/electrostatics")
        if has_any(text, ["electric field", "field strength", "resultant field"]):
            subtopic = "Electric field from point charges"
        elif has_any(text, ["force", "acting", "net electric force"]):
            subtopic = "Coulomb force"
        elif has_any(text, ["potential", "voltage"]):
            subtopic = "Electric potential"
        else:
            subtopic = "Point-charge electrostatics"
        return {
            "domain": "Electricity and magnetism",
            "topic": "Electrostatics",
            "subtopic": subtopic,
            "confidence": "high",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["ammeter", "voltmeter", "least count", "absolute error", "relative error", "uncertainty", "measurement"]):
        rules.append("measurement/error")
        if has_any(text, ["absolute error", "maximum possible", "minimum possible"]):
            subtopic = "Absolute error and bounds"
        elif has_any(text, ["relative error", "percentage error"]):
            subtopic = "Relative error"
        else:
            subtopic = "Measurement uncertainty"
        return {
            "domain": "Measurement",
            "topic": "Measurement and uncertainty",
            "subtopic": subtopic,
            "confidence": "high",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["resistor", "resistance", "current", "voltage", "power", "lamp", "bulb", "parallel circuit", "series circuit"]):
        rules.append("basic circuit")
        if has_any(text, ["power", "watt"]):
            subtopic = "Electrical power"
        elif has_any(text, ["lamp", "bulb", "brighter", "brightness"]):
            subtopic = "Circuit qualitative behavior"
        elif has_any(text, ["ohm", "resistance", "current", "voltage"]):
            subtopic = "Ohm's law"
        else:
            subtopic = "Basic circuit calculation"
        return {
            "domain": "Electricity and magnetism",
            "topic": "Basic circuits",
            "subtopic": subtopic,
            "confidence": "medium",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["lens", "focal length", "image", "object distance", "mirror", "principal axis"]):
        rules.append("optics")
        return {
            "domain": "Optics",
            "topic": "Geometric optics",
            "subtopic": "Lens/image calculation",
            "confidence": "high",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["young", "double-slit", "interference", "monochromatic", "wavelength", "refractive index", "light ray"]):
        rules.append("wave optics")
        if has_any(text, ["young", "double-slit", "interference"]):
            subtopic = "Interference"
        elif has_any(text, ["refractive index", "light ray"]):
            subtopic = "Refraction"
        else:
            subtopic = "Wave optics"
        return {
            "domain": "Optics",
            "topic": "Wave optics",
            "subtopic": subtopic,
            "confidence": "medium",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["car", "motorbike", "motorboat", "airplane", "travels", "speed", "velocity", "distance", "time", "downstream", "upstream"]):
        rules.append("motion/speed")
        if has_any(text, ["downstream", "upstream", "boat"]):
            subtopic = "Relative motion in current"
        elif has_any(text, ["meet", "toward", "from a to b", "from b to a"]):
            subtopic = "Meeting-point motion"
        else:
            subtopic = "Kinematics"
        return {
            "domain": "Mechanics",
            "topic": "Motion",
            "subtopic": subtopic,
            "confidence": "medium",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["mass", "height", "potential energy", "kinetic energy", "mechanical energy", "dropped", "gravity", "work"]):
        rules.append("mechanical energy")
        return {
            "domain": "Mechanics",
            "topic": "Energy",
            "subtopic": "Mechanical energy",
            "confidence": "medium",
            "matched_rules": "; ".join(rules),
        }

    if has_any(text, ["temperature", "heat", "thermal", "specific heat", "celsius"]):
        rules.append("thermal")
        return {
            "domain": "Thermodynamics",
            "topic": "Heat and temperature",
            "subtopic": "Thermal calculation",
            "confidence": "medium",
            "matched_rules": "; ".join(rules),
        }

    return {
        "domain": "Other",
        "topic": "Unclassified",
        "subtopic": "Needs manual review",
        "confidence": "low",
        "matched_rules": "",
    }


def load_dataset(path: Path, sheet: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    dataframe = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    required = {"id", "question", "cot", "answer", "unit"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return dataframe


def classify_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for record in dataframe.to_dict("records"):
        topic = classify_topic(record.get("question"), record.get("cot"))
        rows.append(topic)

    classified = dataframe.copy()
    classified.insert(1, "prefix", classified["id"].map(id_prefix))
    for column in ["domain", "topic", "subtopic", "confidence", "matched_rules"]:
        classified[column] = [row[column] for row in rows]
    classified["task_type"] = [
        task_type(record.get("question"), record.get("answer"), record.get("unit"))
        for record in dataframe.to_dict("records")
    ]
    classified["answer_mode"] = [
        answer_mode(record.get("answer"), record.get("unit"))
        for record in dataframe.to_dict("records")
    ]
    classified["question_text"] = classified["question"].map(clean_text)
    return classified


def write_plots(classified: pd.DataFrame, output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Skipped plots because matplotlib could not be imported: {exc}"]

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    topic_counts = classified["topic"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    topic_counts.plot(kind="barh", ax=ax, color="#4477aa")
    ax.set_title("Question count by topic")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Topic")
    fig.tight_layout()
    fig.savefig(plot_dir / "topics.png", dpi=160)
    plt.close(fig)
    messages.append("plots/topics.png")

    domain_prefix = pd.crosstab(classified["prefix"], classified["domain"])
    fig, ax = plt.subplots(figsize=(11, 6))
    domain_prefix.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Domain mix by ID prefix")
    ax.set_xlabel("Prefix")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "domain_by_prefix.png", dpi=160)
    plt.close(fig)
    messages.append("plots/domain_by_prefix.png")

    task_counts = classified["task_type"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    task_counts.plot(kind="barh", ax=ax, color="#228833")
    ax.set_title("Question count by task type")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Task type")
    fig.tight_layout()
    fig.savefig(plot_dir / "task_types.png", dpi=160)
    plt.close(fig)
    messages.append("plots/task_types.png")

    return messages


def markdown_table(dataframe: pd.DataFrame, max_rows: int | None = None) -> str:
    table = dataframe.head(max_rows) if max_rows else dataframe
    return table.to_markdown(index=False)


def write_report(
    classified: pd.DataFrame,
    output_dir: Path,
    data_path: Path,
    topic_summary: pd.DataFrame,
    subtopic_summary: pd.DataFrame,
    prefix_topic_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    low_confidence: pd.DataFrame,
    plot_messages: list[str],
) -> None:
    total = len(classified)
    low_count = int(classified["confidence"].eq("low").sum())
    report = [
        "# Question Classification",
        "",
        f"- Dataset: `{data_path}`",
        f"- Rows classified: {total}",
        f"- Domains: {classified['domain'].nunique()}",
        f"- Topics: {classified['topic'].nunique()}",
        f"- Subtopics: {classified['subtopic'].nunique()}",
        f"- Low-confidence rows needing manual review: {low_count}",
        "",
        "## Topic Summary",
        "",
        markdown_table(topic_summary),
        "",
        "## Task Type Summary",
        "",
        markdown_table(task_summary),
        "",
        "## Top Subtopics",
        "",
        markdown_table(subtopic_summary.head(30)),
        "",
        "## Prefix x Topic",
        "",
        markdown_table(prefix_topic_summary),
        "",
        "## Low-Confidence Examples",
        "",
        markdown_table(low_confidence[["id", "prefix", "question"]].head(30))
        if not low_confidence.empty
        else "No low-confidence examples.",
        "",
        "## Generated Files",
        "",
        "- `classified_questions.csv`",
        "- `topic_summary.csv`",
        "- `subtopic_summary.csv`",
        "- `prefix_topic_summary.csv`",
        "- `task_type_summary.csv`",
        "- `low_confidence_review.csv`",
    ]
    if plot_messages:
        report.extend(["", "## Plots", ""])
        report.extend(f"- `{message}`" for message in plot_messages)

    (output_dir / "classification_report.md").write_text("\n".join(report) + "\n", encoding="utf-8-sig")


def main() -> None:
    configure_stdout()
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_dataset(data_path, args.sheet)
    classified = classify_dataframe(dataframe)

    topic_summary = (
        classified.groupby(["domain", "topic"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    topic_summary["share"] = (topic_summary["rows"] / len(classified)).round(4)

    subtopic_summary = (
        classified.groupby(["domain", "topic", "subtopic"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    subtopic_summary["share"] = (subtopic_summary["rows"] / len(classified)).round(4)

    prefix_topic_summary = pd.crosstab(classified["prefix"], classified["topic"]).reset_index()
    task_summary = (
        classified.groupby(["task_type", "answer_mode"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    task_summary["share"] = (task_summary["rows"] / len(classified)).round(4)

    low_confidence = classified.loc[classified["confidence"].eq("low")].copy()

    classified.to_csv(output_dir / "classified_questions.csv", index=False, encoding="utf-8-sig")
    topic_summary.to_csv(output_dir / "topic_summary.csv", index=False, encoding="utf-8-sig")
    subtopic_summary.to_csv(output_dir / "subtopic_summary.csv", index=False, encoding="utf-8-sig")
    prefix_topic_summary.to_csv(output_dir / "prefix_topic_summary.csv", index=False, encoding="utf-8-sig")
    task_summary.to_csv(output_dir / "task_type_summary.csv", index=False, encoding="utf-8-sig")
    low_confidence.to_csv(output_dir / "low_confidence_review.csv", index=False, encoding="utf-8-sig")

    plot_messages = [] if args.no_plots else write_plots(classified, output_dir)
    write_report(
        classified=classified,
        output_dir=output_dir,
        data_path=data_path,
        topic_summary=topic_summary,
        subtopic_summary=subtopic_summary,
        prefix_topic_summary=prefix_topic_summary,
        task_summary=task_summary,
        low_confidence=low_confidence,
        plot_messages=plot_messages,
    )

    print(f"Wrote question classification to {output_dir}")
    print(f"Rows classified: {len(classified)}")
    print(f"Topics: {classified['topic'].nunique()}")
    print(f"Low-confidence rows: {len(low_confidence)}")
    print(f"Report: {output_dir / 'classification_report.md'}")


if __name__ == "__main__":
    main()
