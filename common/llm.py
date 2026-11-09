"""One LLM door for every project. Provider from env, never from source.

    from common.llm import complete, complete_json
    text = complete("system prompt", "user text")
    data = complete_json("system prompt", "user text", schema_hint={"tags": ["..."], "priority": "Low|Medium|High|Urgent"})

Provider order: LLM_PROVIDER env (anthropic | openai | gemini | offline) else the first key found
(ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY) else offline. Offline returns a deterministic,
schema-shaped answer so demos and tests run with no network. Models: LLM_MODEL env or the provider default.
"""
from __future__ import annotations
import hashlib, json, os, re

DEFAULTS = {"anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini", "gemini": "gemini-2.0-flash"}

def provider() -> str:
    p = os.getenv("LLM_PROVIDER", "").lower()
    if p in ("anthropic", "openai", "gemini", "offline"):
        return p
    if os.getenv("ANTHROPIC_API_KEY"): return "anthropic"
    if os.getenv("OPENAI_API_KEY"): return "openai"
    if os.getenv("GOOGLE_API_KEY"): return "gemini"
    return "offline"

def model() -> str:
    return os.getenv("LLM_MODEL") or DEFAULTS.get(provider(), "offline")

def _offline(system: str, user: str, schema_hint=None) -> str:
    h = hashlib.sha256((system + "\n" + user).encode()).hexdigest()[:8]
    if schema_hint is not None:
        out = {}
        for k, v in schema_hint.items():
            if isinstance(v, list): out[k] = [str(v[0]).split("|")[0].strip("<>. ")] if v else []
            elif isinstance(v, (int, float)): out[k] = v
            elif isinstance(v, str) and "|" in v: out[k] = v.split("|")[0].strip()
            else: out[k] = f"offline answer {h}: " + user[:80].replace("\n", " ")
        return json.dumps(out)
    return f"[offline {h}] " + (user.strip().splitlines() or [""])[0][:200]

def complete(system: str, user: str, max_tokens: int = 800, temperature: float = 0.2) -> str:
    p = provider()
    if p == "offline":
        return _offline(system, user)
    if p == "anthropic":
        import anthropic
        c = anthropic.Anthropic()
        r = c.messages.create(model=model(), max_tokens=max_tokens, temperature=temperature, system=system,
                              messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    if p == "openai":
        from openai import OpenAI
        r = OpenAI().chat.completions.create(model=model(), temperature=temperature, max_tokens=max_tokens,
                                             messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or ""
    if p == "gemini":
        from google import genai
        c = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        r = c.models.generate_content(model=model(), contents=f"{system}\n\n{user}")
        return r.text or ""
    raise RuntimeError(f"unknown provider {p}")

def complete_json(system: str, user: str, schema_hint: dict | None = None, **kw) -> dict:
    if provider() == "offline":
        return json.loads(_offline(system, user, schema_hint or {}))
    hint = f"\nReturn only a JSON object with keys {json.dumps(schema_hint)}." if schema_hint else "\nReturn only a JSON object."
    text = complete(system + hint, user, **kw)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)
