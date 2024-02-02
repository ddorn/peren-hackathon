# %%
from json import load
from pathlib import Path
import pickle
import typer
import pandas as pd
import re

import torch
from torch import Tensor
from sentence_transformers import SentenceTransformer
from jaxtyping import Float

app = typer.Typer()


DOCUMENTATION_PATH = Path('./platform-docs-versions')
if not DOCUMENTATION_PATH.exists():
    DOCUMENTATION_PATH = Path('/gpfsdswork/dataset/hackathon_peren/platform-docs-versions')

@app.command()
def preprocess():
    # Make chunks
    frame = chunk_markdown_files_to_dataframe(DOCUMENTATION_PATH)
    frame = frame.head(10)

    # Create keys from chunks
    create_keys_from_chunks(frame)
    # Embed keys
    add_embeddings(frame, "key")

    # Save the BLOB = panda frame
    # Columns: embedding, key, chuck, url
    pickle.dump(frame, "blob.pkl")


@app.command()
def run(input_file: Path, output_file: Path):
    ...
    # Load the blob memory store (questions -> paragraphs)
    # %%
    input_file = "questions.csv"
    blob = pickle.load("blob.pkl")

    # Load the csv input file
    questions = pd.read_csv(input_file, sep=";")  # columns: id;question
    questions = questions.head(2)

    # Generate keys for search
    add_keys_to_questions(questions)

    # Embed the questions
    add_embeddings(questions, "question")

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


def chunk_markdown_files_to_dataframe(directory: Path, chunk_size=500):
    dfs = []

    for markdown_file_path in directory.glob('**/*.md'):
        # TODO: Check if readme
        chunks = []

        with open(markdown_file_path, 'r', encoding='utf-8') as file:
            content = file.read()

            url = extract_url_from_markdown(content)
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
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


def create_keys_from_chunks(frame: pd.DataFrame):
    frame["key"] = frame["chunk"].apply(lambda x: x[:100])


def add_embeddings(frame: pd.DataFrame, from_column: str):
    sentences = frame[from_column].tolist()
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = model.encode(sentences)
    frame["embedding"] = list(embeddings)


def add_keys_to_questions(questions: pd.DataFrame):
    """
    Add keys to the questions.

    Args:
        questions (pd.DataFrame): The questions with columns "question"
    """

    questions["key"] = questions["question"].apply(lambda x: x[:100])


def compute_similiarities(questions: pd.DataFrame, blob: pd.DataFrame) -> Float[Tensor, "question chunk"]:
    """
    Compute similiarities between questions and chunks.

    Args:
        questions (pd.DataFrame): The questions with columns "embedding"
        blob (pd.DataFrame): The blob with columns "embedding"

    Returns:
        Float[Tensor, "question chunk"]: The similiarities. Indices correspond to the indices in the pandas
    """

    questions = torch.tensor(questions["embedding"].values)
    chunks = torch.tensor(blob["embedding"].values)

    similiarities = torch.nn.functional.cosine_similarity(questions, chunks)

    return similiarities


def filter_similiarities(similiarities: Float[Tensor, "question chunk"]) -> list[list[int]]:
    """
    Filter the similiarities to select the best chunks for each question.

    Returns:
        list[int]: The indices of the selected chunks
    """

    # Select the best chunk for each question
    selected_chunks = similiarities.argmax(dim=1, keepdim=True)
    return selected_chunks.tolist()

def make_response(questions: pd.DataFrame, blob: pd.DataFrame, selected_chunk_ids: list[list[int]]):
    """
    Add a 'answer', 'prompt' and 'sources' column to each question.

    Args:
        questions (pd.DataFrame): The questions with columns "question"
        blob (pd.DataFrame): The blob with columns "chunk", "url"
        selected_chunk_ids (list[list[int]]): The indices of the selected chunks
    """

    questions["answer"] = [blob.iloc[i]["chunk"] for i in selected_chunk_ids]
    questions["prompt"] = [""] * len(questions)
    questions["source"] = [blob.iloc[i]["url"] for i in selected_chunk_ids]


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

    for i, row in frame.iterrows():
        answer = row["answer"]
        prompt = row["prompt"]
        source = row["source"]
        id = row["id"]

        with open(answers / f"{id}.txt", "w") as f:
            f.write(answer)

        with open(prompts / f"{id}.txt", "w") as f:
            f.write(prompt)

        with open(sources / f"{id}.txt", "w") as f:
            f.write(source)


if __name__ == "__main__":
    app()