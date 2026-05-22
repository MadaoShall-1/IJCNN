from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_DATASET = "Physics_Problems_Text_Only.xlsx"
DEFAULT_OUTPUT_DIR = "outputs/data_analysis"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "back",
    "be",
    "by",
    "calculate",
    "determine",
    "does",
    "find",
    "for",
    "from",
    "given",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "respectively",
    "the",
    "this",
    "through",
    "to",
    "two",
    "what",
    "when",
    "where",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the physics Excel dataset.")
    parser.add_argument("--data", default=DEFAULT_DATASET, help="Path to the Excel dataset.")
    parser.add_argument("--sheet", default=None, help="Optional sheet name. Defaults to the first sheet.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for report, CSVs, and plots.")
    parser.add_argument("--top-units", type=int, default=30, help="Number of units to show in the report.")
    parser.add_argument("--sample-per-prefix", type=int, default=3, help="Rows sampled per ID prefix.")
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


def word_count(value: Any) -> int:
    text = clean_text(value)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def char_count(value: Any) -> int:
    return len(clean_text(value))


def normalize_dash_unit(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return "<missing>"
    if text in {"-", "—", "–"}:
        return "-"
    return text


def classify_answer(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return "missing"

    lowered = text.lower()
    if lowered in {"yes", "no", "true", "false"}:
        return "boolean"

    if any(separator in text for separator in [";", ","]):
        return "multi_value"

    if re.search(r"sqrt|\\sqrt", text, flags=re.IGNORECASE):
        return "symbolic_sqrt"

    if re.search(r"(×|x|\*)\s*10|\be[+-]?\d+", text, flags=re.IGNORECASE):
        return "scientific_notation"

    if "/" in text:
        return "fraction_or_ratio"

    if re.fullmatch(r"[+-]?\d+(\.\d+)?", text):
        return "plain_number"

    if re.search(r"[A-Za-z]", text):
        return "text_or_formula"

    return "other_numeric"


def extract_keywords(series: pd.Series, top_n: int = 25) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    for value in series.dropna():
        words = re.findall(r"[A-Za-z][A-Za-z-]{2,}", str(value).lower())
        counter.update(word for word in words if word not in STOPWORDS)
    return [{"keyword": word, "count": count} for word, count in counter.most_common(top_n)]


def load_dataset(path: Path, sheet: str | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    required = {"id", "question", "cot", "answer", "unit"}
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return dataframe


def add_derived_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()
    df["prefix"] = df["id"].map(id_prefix)
    df["question_text"] = df["question"].map(clean_text)
    df["cot_text"] = df["cot"].map(clean_text)
    df["answer_text"] = df["answer"].map(clean_text)
    df["unit_text"] = df["unit"].map(clean_text)
    df["normalized_unit"] = df["unit"].map(normalize_dash_unit)
    df["answer_format"] = df["answer"].map(classify_answer)
    df["has_question"] = df["question_text"].ne("")
    df["has_cot"] = df["cot_text"].ne("")
    df["has_answer"] = df["answer_text"].ne("")
    df["has_unit"] = df["unit_text"].ne("")
    df["is_labeled"] = df["has_question"] & df["has_answer"] & df["has_unit"]
    df["question_chars"] = df["question"].map(char_count)
    df["question_words"] = df["question"].map(word_count)
    df["cot_chars"] = df["cot"].map(char_count)
    df["cot_words"] = df["cot"].map(word_count)
    return df


def prefix_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("prefix", dropna=False)
    summary = grouped.agg(
        rows=("id", "size"),
        labeled_rows=("is_labeled", "sum"),
        missing_question=("has_question", lambda values: int((~values).sum())),
        missing_cot=("has_cot", lambda values: int((~values).sum())),
        missing_answer=("has_answer", lambda values: int((~values).sum())),
        missing_unit=("has_unit", lambda values: int((~values).sum())),
        unique_units=("normalized_unit", lambda values: values[values.ne("<missing>")].nunique()),
        avg_question_words=("question_words", "mean"),
        median_question_words=("question_words", "median"),
        avg_cot_words=("cot_words", "mean"),
        median_cot_words=("cot_words", "median"),
    ).reset_index()
    summary["labeled_rate"] = summary["labeled_rows"] / summary["rows"]
    numeric_columns = [
        "avg_question_words",
        "median_question_words",
        "avg_cot_words",
        "median_cot_words",
        "labeled_rate",
    ]
    summary[numeric_columns] = summary[numeric_columns].round(3)
    return summary.sort_values(["rows", "prefix"], ascending=[False, True])


def unit_summary(df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df["normalized_unit"]
        .value_counts(dropna=False)
        .rename_axis("unit")
        .reset_index(name="rows")
    )
    counts["share"] = (counts["rows"] / len(df)).round(4)
    return counts


def answer_format_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["prefix", "answer_format"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["prefix", "rows"], ascending=[True, False])
    )
    return summary


def length_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for prefix, group in df.groupby("prefix", dropna=False):
        for column in ["question_words", "cot_words", "question_chars", "cot_chars"]:
            values = group[column]
            rows.append(
                {
                    "prefix": prefix,
                    "metric": column,
                    "min": int(values.min()),
                    "p25": round(float(values.quantile(0.25)), 2),
                    "median": round(float(values.median()), 2),
                    "mean": round(float(values.mean()), 2),
                    "p75": round(float(values.quantile(0.75)), 2),
                    "max": int(values.max()),
                }
            )
    return pd.DataFrame(rows)


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    flags: list[pd.DataFrame] = []

    checks = {
        "missing_question": ~df["has_question"],
        "missing_cot": ~df["has_cot"],
        "missing_answer": ~df["has_answer"],
        "missing_unit": ~df["has_unit"],
        "duplicate_id": df["id"].duplicated(keep=False),
        "duplicate_question": df["question_text"].ne("") & df["question_text"].duplicated(keep=False),
        "unit_dash_variant": df["unit_text"].isin(["—", "–"]),
        "unit_micro_variant": df["unit_text"].str.contains("µ", regex=False, na=False),
    }

    for flag, mask in checks.items():
        subset = df.loc[mask, ["id", "prefix", "question_text", "answer_text", "unit_text"]].copy()
        if not subset.empty:
            subset.insert(0, "flag", flag)
            flags.append(subset)

    if not flags:
        return pd.DataFrame(columns=["flag", "id", "prefix", "question_text", "answer_text", "unit_text"])

    return pd.concat(flags, ignore_index=True)


def sample_rows(df: pd.DataFrame, sample_per_prefix: int) -> pd.DataFrame:
    return (
        df.sort_values("id")
        .groupby("prefix", group_keys=False)
        .head(sample_per_prefix)[["id", "prefix", "question_text", "answer_text", "unit_text", "answer_format"]]
    )


def write_plots(df: pd.DataFrame, output_dir: Path, top_units: int) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Skipped plots because matplotlib could not be imported: {exc}"]

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    messages: list[str] = []

    prefix_counts = df["prefix"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    prefix_counts.plot(kind="barh", ax=ax, color="#4477aa")
    ax.set_title("Rows by ID prefix")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Prefix")
    fig.tight_layout()
    fig.savefig(plot_dir / "rows_by_prefix.png", dpi=160)
    plt.close(fig)
    messages.append("plots/rows_by_prefix.png")

    missing_by_prefix = df.groupby("prefix")[["has_question", "has_cot", "has_answer", "has_unit"]].apply(
        lambda group: pd.Series(
            {
                "question": int((~group["has_question"]).sum()),
                "cot": int((~group["has_cot"]).sum()),
                "answer": int((~group["has_answer"]).sum()),
                "unit": int((~group["has_unit"]).sum()),
            }
        )
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    missing_by_prefix.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Missing fields by prefix")
    ax.set_xlabel("Prefix")
    ax.set_ylabel("Missing rows")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "missing_by_prefix.png", dpi=160)
    plt.close(fig)
    messages.append("plots/missing_by_prefix.png")

    unit_counts = df["normalized_unit"].value_counts().head(top_units).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    unit_counts.plot(kind="barh", ax=ax, color="#228833")
    ax.set_title(f"Top {min(top_units, len(unit_counts))} units")
    ax.set_xlabel("Rows")
    ax.set_ylabel("Unit")
    fig.tight_layout()
    fig.savefig(plot_dir / "top_units.png", dpi=160)
    plt.close(fig)
    messages.append("plots/top_units.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    df.boxplot(column="question_words", by="prefix", ax=ax, grid=False)
    ax.set_title("Question word counts by prefix")
    ax.set_xlabel("Prefix")
    ax.set_ylabel("Question words")
    fig.suptitle("")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(plot_dir / "question_words_by_prefix.png", dpi=160)
    plt.close(fig)
    messages.append("plots/question_words_by_prefix.png")

    return messages


def markdown_table(dataframe: pd.DataFrame, max_rows: int | None = None) -> str:
    table = dataframe.head(max_rows) if max_rows else dataframe
    return table.to_markdown(index=False)


def write_report(
    df: pd.DataFrame,
    output_dir: Path,
    data_path: Path,
    prefix_df: pd.DataFrame,
    unit_df: pd.DataFrame,
    format_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    keyword_rows: list[dict[str, Any]],
    plot_messages: list[str],
    top_units: int,
) -> None:
    total = len(df)
    labeled = int(df["is_labeled"].sum())
    duplicate_ids = int(df["id"].duplicated().sum())
    duplicate_question_mask = df["question_text"].ne("") & df["question_text"].duplicated(keep=False)
    duplicate_question_rows = int(duplicate_question_mask.sum())
    duplicate_question_groups = int(df.loc[duplicate_question_mask, "question_text"].nunique())
    qa_missing = int(df.loc[df["prefix"].eq("QA"), "has_answer"].eq(False).sum())

    top_keywords = pd.DataFrame(keyword_rows)
    top_format = (
        df["answer_format"]
        .value_counts()
        .rename_axis("answer_format")
        .reset_index(name="rows")
    )

    lines = [
        "# Physics Dataset Analysis",
        "",
        f"- Dataset: `{data_path}`",
        f"- Rows: {total}",
        f"- Columns: {', '.join(df.columns[:5])}",
        f"- Fully labeled rows: {labeled} ({labeled / total:.1%})",
        f"- Duplicate IDs: {duplicate_ids}",
        f"- Duplicate non-empty question rows: {duplicate_question_rows}",
        f"- Duplicate non-empty question groups: {duplicate_question_groups}",
        f"- QA rows without gold answers: {qa_missing}",
        "",
        "## Prefix Summary",
        "",
        markdown_table(prefix_df),
        "",
        "## Top Units",
        "",
        markdown_table(unit_df.head(top_units)),
        "",
        "## Answer Format Summary",
        "",
        markdown_table(top_format),
        "",
        "## Frequent Question Keywords",
        "",
        markdown_table(top_keywords),
        "",
        "## Quality Flags",
        "",
        markdown_table(
            quality_df["flag"].value_counts().rename_axis("flag").reset_index(name="rows")
            if not quality_df.empty
            else pd.DataFrame([{"flag": "none", "rows": 0}])
        ),
        "",
        "## Generated Files",
        "",
        "- `prefix_summary.csv`",
        "- `unit_summary.csv`",
        "- `answer_format_summary.csv`",
        "- `length_summary.csv`",
        "- `quality_flags.csv`",
        "- `sample_by_prefix.csv`",
        "- `keyword_summary.json`",
    ]

    if plot_messages:
        lines.extend(["", "## Plots", ""])
        lines.extend(f"- `{message}`" for message in plot_messages)

    report_path = output_dir / "dataset_profile.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def main() -> None:
    configure_stdout()
    args = parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(data_path, args.sheet)
    df = add_derived_columns(raw_df)

    prefix_df = prefix_summary(df)
    unit_df = unit_summary(df)
    format_df = answer_format_summary(df)
    length_df = length_summary(df)
    quality_df = quality_flags(df)
    sample_df = sample_rows(df, args.sample_per_prefix)
    keyword_rows = extract_keywords(df["question_text"])

    prefix_df.to_csv(output_dir / "prefix_summary.csv", index=False, encoding="utf-8-sig")
    unit_df.to_csv(output_dir / "unit_summary.csv", index=False, encoding="utf-8-sig")
    format_df.to_csv(output_dir / "answer_format_summary.csv", index=False, encoding="utf-8-sig")
    length_df.to_csv(output_dir / "length_summary.csv", index=False, encoding="utf-8-sig")
    quality_df.to_csv(output_dir / "quality_flags.csv", index=False, encoding="utf-8-sig")
    sample_df.to_csv(output_dir / "sample_by_prefix.csv", index=False, encoding="utf-8-sig")
    (output_dir / "keyword_summary.json").write_text(
        json.dumps(keyword_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_messages = [] if args.no_plots else write_plots(df, output_dir, args.top_units)
    write_report(
        df=df,
        output_dir=output_dir,
        data_path=data_path,
        prefix_df=prefix_df,
        unit_df=unit_df,
        format_df=format_df,
        quality_df=quality_df,
        keyword_rows=keyword_rows,
        plot_messages=plot_messages,
        top_units=args.top_units,
    )

    print(f"Wrote dataset analysis to {output_dir}")
    print(f"Rows: {len(df)}")
    print(f"Fully labeled rows: {int(df['is_labeled'].sum())}")
    print(f"Prefixes: {', '.join(prefix_df['prefix'].astype(str).tolist())}")
    print(f"Report: {output_dir / 'dataset_profile.md'}")


if __name__ == "__main__":
    main()
