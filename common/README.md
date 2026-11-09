# common
`llm.py`: the one LLM door (provider from env, offline fallback). Projects import it with
`sys.path.insert(0, str(Path(__file__).resolve().parents[N]))` then `from common.llm import complete, complete_json`,
or by `uv add --editable ../../common` where a package is cleaner. Keep it dependency-free at import time: the SDKs are imported lazily.
