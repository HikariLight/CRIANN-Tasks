import csv
import json
import logging
import re
from pathlib import Path

from tqdm import tqdm
from transformers import AutoTokenizer


def read_labels(labels_path, test=False):
    labels = {}
    labels_start = 2 if test else 4
    with open(Path(labels_path), encoding="utf-8") as f:
        labels_reader = csv.reader(f)
        next(labels_reader, None)
        for row in labels_reader:
            if row[0] not in ["451", "458"]:  # broken interviews
                try:
                    labels[row[0]] = [int(x) for x in row[labels_start:]]
                except ValueError:
                    logging.warning("[Could not read the labels for the transcript %s!]", row[0])
    return labels


def read_data(labels, data_path, method):
    ellie_line = re.compile(r"\((.+)\)")
    data = []
    for idx, label in tqdm(labels.items(), desc=f"Reading labels for {method} chunking."):
        transcript_path = data_path / f"{idx}_TRANSCRIPT.csv"
        transcript_lines = []
        if transcript_path.exists():
            with open(transcript_path, encoding="utf-8") as f:
                csv_reader = csv.reader(f, delimiter="\t")
                header = next(csv_reader)
                for row in csv_reader:
                    if row:
                        try:
                            if row[2] == "Ellie" and ellie_line.findall(row[3]):
                                transcript_lines.append((row[2], re.findall(ellie_line, row[3])[0]))
                            else:
                                transcript_lines.append((row[2], row[3]))
                        except IndexError as e:
                            print(idx, row)
                            raise e
            chunks = chunk_interview(transcript_lines, method)
            data_sample = {"id": idx, "turns": chunks, "labels": label}
            data.append(data_sample)
        else:
            print(f"Skipping transcript {transcript_path} because it does not exist.")
    return data


def chunk_interview(lines, method):
    chunks = []
    chunk = []
    if method == "lines":
        for sent in lines:
            chunks.append(sent[1])
    else:
        raise NotImplementedError("Invalid chunking method! Only 'lines' method is supported.")
    return chunks


def main():
    data_path = Path("data/transcripts")
    train_labels_path = Path("data/train_split_Depression_AVEC2017.csv")
    dev_labels_path = Path("data/dev_split_Depression_AVEC2017.csv")
    test_labels_path = Path("data/test_split_Depression_AVEC2017.csv")

    train_labels = read_labels(train_labels_path)
    dev_labels = read_labels(dev_labels_path)
    test_labels = read_labels(test_labels_path, test=True)

    for method in ["lines"]:
        save_dir = Path(f"data/json/{method}")
        save_dir.mkdir(parents=True, exist_ok=True)
        train_data = read_data(train_labels, data_path, method)
        dev_data = read_data(dev_labels, data_path, method)
        test_data = read_data(test_labels, data_path, method)

        with open(save_dir / "train.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join([json.dumps(line) for line in train_data]))
        with open(save_dir / "validation.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join([json.dumps(line) for line in dev_data]))
        with open(save_dir / "test.jsonl", "w", encoding="utf-8") as f:
            f.write("\n".join([json.dumps(line) for line in test_data]))


if __name__ == "__main__":
    main()
