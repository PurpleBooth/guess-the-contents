import fnmatch
from pathlib import Path

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Small, public (ungated), code-focused instruct model that downloads from
# the Hugging Face Hub on first run and runs comfortably on CPU.
# Bigger / smarter alternatives: "Qwen/Qwen2.5-Coder-3B-Instruct",
# "Qwen/Qwen2.5-Coder-7B-Instruct".
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Directories that add noise and tokens without helping the model guess a
# source file's contents.
IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
}


def _ignored(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pat) for pat in IGNORED_DIRS)


def list_tree(root: str | Path, max_depth: int = 4) -> str:
    """Return the directory tree under ``root`` as a string (up to ``max_depth``)."""
    root = Path(root)

    lines: list[str] = []

    def _walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        indent = "  " * depth
        prefix = "📁" if path.is_dir() else "📄"
        lines.append(f"{indent}{prefix} {path.name}")
        if path.is_dir():
            try:
                for entry in sorted(path.iterdir()):
                    if _ignored(entry.name):
                        continue
                    _walk(entry, depth + 1)
            except PermissionError:
                ...

    _walk(root, 0)
    return "\n".join(lines)


def main(
    filename: str = typer.Argument(..., help="Path of the file whose contents to guess."),
    working_directory: Path = typer.Option(
        None,
        "--cwd",
        "-C",
        help="Directory whose tree to use as context. Defaults to the current dir.",
    ),
    model_name: str = typer.Option(
        DEFAULT_MODEL,
        "--model",
        "-m",
        help="Hugging Face model id. Must be a chat/instruct model.",
    ),
    max_new_tokens: int = typer.Option(
        512, "--max-tokens", "-n", help="Maximum number of new tokens to generate."
    ),
    max_depth: int = typer.Option(
        4, "--max-depth", "-d", help="Maximum directory tree depth to include."
    ),
) -> None:
    """Guess the likely contents of ``filename`` from its surrounding directory tree."""
    working_directory = working_directory or Path.cwd()

    tree = list_tree(working_directory, max_depth=max_depth)
    if not tree:
        typer.secho(
            f"Warning: empty tree for {working_directory}", fg=typer.colors.YELLOW, err=True
        )

    typer.secho(f"Loading model {model_name} ...", fg=typer.colors.CYAN, err=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # bfloat16 works on CPU and halves memory vs float32; float16 on CUDA.
        dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a code-completion assistant. Given a directory tree and a "
                "filename, output ONLY the likely contents of that file. No "
                "explanation, no markdown fences, no surrounding prose."
            ),
        },
        {
            "role": "user",
            "content": f"Directory tree:\n\n{tree}\n\nGuess the contents of: {filename}",
        },
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Drop the prompt tokens, keep only newly generated ones.
    generated = out[0][inputs["input_ids"].shape[1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip("```"))


if __name__ == "__main__":
    typer.run(main)
