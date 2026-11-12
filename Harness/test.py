from pathlib import Path
import glob as g

WORKDIR = Path.cwd() / "RAG/data/"



matches = sorted({
    match for match in g.glob(
        "load_file/*", root_dir=WORKDIR, recursive=True)
    if (WORKDIR / match).resolve().is_relative_to(WORKDIR)
})

print(matches)