# %%
import os
from pathlib import Path
import pickle
import typer
import pandas as pd
import re

import numpy as np
import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer

NO_LLM = not torch.cuda.is_available()

try:
    from jaxtyping import Float
except ImportError:
    Float = list

import tqdm

app = typer.Typer()

DSDIR = Path(os.environ.get("DSDIR", ""))

DOCUMENTATION_PATH = Path('./platform-docs-versions')
if not DOCUMENTATION_PATH.exists():
    DOCUMENTATION_PATH = DSDIR / 'hackathon_peren/platform-docs-versions'

EMBEDDING_MODEL_PATH = DSDIR / "HuggingFace_Models/sentence-transformers/all-mpnet-base-v2"
if not EMBEDDING_MODEL_PATH.exists():
    EMBEDDING_MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"

BLOB_FILE = Path('blob.pkl')

@app.command()
def preprocess(max_chunks: int = -1):
    # Make chunks
    frame = chunk_markdown_files_to_dataframe(DOCUMENTATION_PATH)

    if max_chunks > 0:
        frame = frame.head(max_chunks)
    print("Number of chunks", len(frame))

    # Create keys from chunks
    add_key_to_chunks(frame)
    # Embed keys
    add_embeddings(frame, "key")

    # Save the BLOB = panda frame
    # Columns: embedding, key, chuck, url
    with open(BLOB_FILE, "wb") as f:
        pickle.dump(frame, f)


@app.command()
def run(input_file: Path, output_file: Path, max_inputs: int = -1):

    # Load the csv input file
    questions = pd.read_csv(input_file, sep=";", dtype=str)
    print(questions)
    if max_inputs > 0:
        questions = questions.head(max_inputs)

    # Generate keys for search
    add_key_to_questions(questions)

    # Embed the questions
    add_embeddings(questions, "key")

    # Load the blob memory store (questions -> paragraphs)
    # %%
    # input_file = "questions.csv"
    blob = pickle.load(BLOB_FILE.open("rb"))

    # Compute similiarities
    similiarities = compute_similiarities(questions, blob)

    # Filter similiarities
    selected_chunk_ids: list[list[int]] = filter_similiarities(similiarities)

    # url, chuncks -> reponses, urls
    make_response(questions, blob, selected_chunk_ids)

    # Create output format
    make_output(questions, output_file)

#%%


# Function to extract URL from the first line of Markdown content
def extract_url_from_markdown(content: str) -> str:
    lines = content.strip().split('\n')
    if lines:
        first_line = lines[0].strip()
        if first_line.startswith("# Resource URL:"):
            match = re.search(r'https?://[^\s]+', first_line)
            if match:
                return match.group()
    print("No Url found. Start", content[:100])
    return "https://Unspecified Url"


def chunk_markdown_files_to_dataframe(directory: Path, chunk_size=1500):
    dfs = []

    for markdown_file_path in directory.glob('**/*.md'):
        # TODO: Check if readme
        chunks = []

        with open(markdown_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            url = extract_url_from_markdown(content)
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                file = markdown_file_path.parent.name + "/" + markdown_file_path.name
                chunk = f"Url: {url}\nFile: {file}\n{chunk}"
                chunks.append(chunk)

        chunk_numbers = list(range(1, len(chunks) + 1))

        df = pd.DataFrame({
            'Chunk Number': chunk_numbers,
            'Markdown File Path': [str(markdown_file_path).replace("platform-docs-versions/", "")] * len(chunks),
            'chunk': chunks,
            'url': url,
        })

        dfs.append(df)

    if dfs:
        result_df = pd.concat(dfs, ignore_index=True)
        return result_df
    else:
        return None





def add_key_to_chunks(frame: pd.DataFrame):
    """Add a 'key' column to the frame, from the 'chunk' column."""
    if NO_LLM:
        frame["key"] = frame["chunk"]
        return

    from alex import get_key_from_text
    frame['key'] = get_key_from_text(frame['chunk'].tolist())


def add_key_to_questions(frame: pd.DataFrame):
    """
    Add a "key" column to the frame, from the "question" column.
    """
    if NO_LLM:
        frame["key"] = frame["question"]
        return

    from alex import get_key_from_text
    frame['key'] = get_key_from_text(frame['question'].tolist())


def add_embeddings(frame: pd.DataFrame, from_column: str):
    sentences = frame[from_column].tolist()
    model = SentenceTransformer(EMBEDDING_MODEL_PATH)
    embeddings = model.encode(sentences, show_progress_bar=True, normalize_embeddings=True)
    frame["embedding"] = list(embeddings)


def compute_similiarities(questions: pd.DataFrame, blob: pd.DataFrame) -> Float[Tensor, "question chunk"]:
    """
    Compute similiarities between questions and chunks.

    Args:
        questions (pd.DataFrame): The questions with columns "embedding"
        blob (pd.DataFrame): The blob with columns "embedding"

    Returns:
        Float[Tensor, "question chunk"]: The similiarities. Indices correspond to the indices in the pandas
    """

    def to_np(x):
        return torch.tensor(np.stack(x))


    questions = to_np(questions["embedding"].values)
    chunks = to_np(blob["embedding"].values)

    similiarities = torch.nn.functional.cosine_similarity(questions[:, None, :], chunks[None, :, :], dim=-1)

    return similiarities


def filter_similiarities(similiarities: Float[Tensor, "question chunk"], k=1) -> list[list[int]]:
    """
    Filter the similiarities to select the best chunks for each question.

    Returns:
        list[int]: The indices of the selected chunks
    """

    # Select the best chunk for each question
    selected_chunks = similiarities.topk(k, dim=1).indices
    return selected_chunks.tolist()

def make_response(questions: pd.DataFrame, blob: pd.DataFrame, selected_chunk_ids: list[list[int]]):
    """
    Add a 'answer', 'prompt' and 'sources' column to each question.

    Args:
        questions (pd.DataFrame): The questions with columns "question"
        blob (pd.DataFrame): The blob with columns "chunk", "url"
        selected_chunk_ids (list[list[int]]): The indices of the selected chunks
    """

    questions["answer"] = [blob["chunk"].iloc[i[0]] for i in selected_chunk_ids]
    questions["prompt"] = [""] * len(questions)
    questions["source"] = [blob["url"].iloc[i[0]] for i in selected_chunk_ids]


def make_output(frame: pd.DataFrame, output_dir: Path):
    """
    Save the chunks in the output_dir.

    Args:
        frame (pd.DataFrame): The frame with columns "id", "answer", "prompt", "sources"
    """

    answers = output_dir / "answers"
    prompts = output_dir / "prompts"
    sources = output_dir / "sources"

    for dir in [answers, prompts, sources]:
        dir.mkdir(exist_ok=True, parents=True)

    for row in frame.itertuples():
        id = row.id
        answer = row.answer
        prompt = row.prompt
        source = row.source


        (sources / f"{id}.txt").write_text(source + "\n")
        (prompts / f"{id}.txt").write_text(prompt + "\n")
        (answers / f"{id}.txt").write_text(answer + "\n")


@app.command()
def diff(our_dir: Path, ground_truth: Path):
    """Check that the sources match the ground truth."""

    our_sources = list(our_dir.glob("*.txt"))
    ground_truth_sources = list(ground_truth.glob("*.txt"))

    assert len(our_sources) == len(ground_truth_sources)

    for our_source in our_sources:
        with open(our_source, "r") as f:
            our_content = f.read()

        ground_truth_source = ground_truth / our_source.name
        with open(ground_truth_source, "r") as f:
            ground_truth_content = f.read()

        if our_content.strip() == ground_truth_content.strip():
            print(f"{our_source.name} ✅")
        else:
            print(f"{our_source.name} ❌")


@app.command()
def top5(input_file):
    """Compute the top 5 accuracy."""

    # Load the csv input file
    questions = pd.read_csv(input_file, sep=";", dtype=str)

    # Generate keys for search
    add_key_to_questions(questions)

    # Embed the questions
    add_embeddings(questions, "key")

    # Load the blob memory store (questions -> paragraphs)
    blob = pickle.load(BLOB_FILE.open("rb"))

    # Compute similiarities
    similiarities = compute_similiarities(questions, blob)

    # Filter similiarities
    selected_chunk_ids: list[list[int]] = filter_similiarities(similiarities)

    # Get the urls
    urls = [
        [blob["url"].iloc[i] for i in indices]
        for indices in selected_chunk_ids
    ]

    # Compare to the ground truth
    ground = """https://developers.facebook.com/docs/graph-api/reference/ads_archive/
https://www.facebook.com/ads/library/api/
https://about.linkedin.com/transparency
https://about.linkedin.com/transparency/government-requests-report
https://policy.pinterest.com/en/digital-services-act-transparency-report
https://help.crowdtangle.com/en/articles/3443476-api-cheat-sheet
https://developer.twitter.com/en/docs/twitter-api/compliance/batch-compliance/introduction
https://developers.tiktok.com/doc/research-api-specs-query-video-comments
https://values.snap.com/fr-FR/privacy/transparency/european-union
https://transparency.twitter.com/dsa-transparency-report.html"""
    ground = ground.strip().split("\n")

    for i, url in enumerate(urls):
        try:
            index = url.index(ground[i])
        except ValueError:
            print("Not found ❌")
        else:
            print(f"Good index {index} ✅")





# %%
if __name__ == "__main__":
    app()
