# SPDX-License-Identifier: MIT
"""Model interface for the hard problem.

Two modes matter for the paper. Zero-shot configuration reads the scenario
description and proposes one configuration with no search. Search runs an
iterative loop that refines from feedback. Both build their prompt from the
parameter schema, so nothing here is parameter specific.

A mock mode lets the whole comparison run without Ollama. In mock mode the
zero-shot call returns the informed prior, the textbook ceiling, and the search
call refines around it. The mock therefore measures the harness and the ceiling,
not the real model. Real numbers require Ollama, set mock to False.

The prompt never asks the model to compute arithmetic. A checker reports the
offered load. This matches the fix applied to the V7 optimizer.
"""

from __future__ import annotations

import json
import random
import re
from typing import Dict, List, Optional

from . import hard_problem as hp


class LLMParseError(RuntimeError):
    """Raised when a model reply cannot be converted into a complete configuration."""



def param_spec_text() -> str:
    lines = []
    for name, spec in hp.PARAM_SCHEMA.items():
        if spec["type"] == "cat":
            lines.append(f"- {name}: one of {spec['values']}")
        elif spec["type"] == "int":
            lines.append(f"- {name}: integer from {spec['low']} to {spec['high']}")
        else:
            lines.append(f"- {name}: number from {spec['low']} to {spec['high']}")
    return "\n".join(lines)


def _scenario_block(sc: Dict) -> str:
    return (
        f"SCENARIO: {sc['name']} ({sc['use_case']})\n"
        f"DESCRIPTION: {sc['desc']}\n"
        f"Estimated capacity: {sc['capacity']:.0f} Mbps. "
        f"Traffic is non-stationary across load phases.\n"
    )


def _goal_block(sc: Dict) -> str:
    return (
        "Maximize a weighted score of throughput, packet delivery ratio, delay, "
        "and fairness, weighted for this use case, and stable across the load "
        "phases. Keep the offered load below the estimated capacity. A checker "
        "computes the offered load and reports it. Do not compute it yourself "
        "and put no arithmetic in the reasoning.\n"
    )


def build_zeroshot_prompt(sc: Dict) -> str:
    return (
        "You configure a 5G NR cell from its description, using your knowledge "
        "of 5G NR.\n\n"
        + _scenario_block(sc)
        + "\nChoose values for every parameter:\n"
        + param_spec_text()
        + "\n\n" + _goal_block(sc)
        + "\nRespond with ONLY one JSON object containing all parameters and a "
        "short \"reasoning\" field. No text outside the JSON."
    )


def build_search_prompt(sc: Dict, tried: List[Dict], best: Optional[Dict],
                        num_candidates: int) -> str:
    tried_text = "none yet"
    if tried:
        rows = []
        for e in tried[-12:]:
            p = e["params"]
            rows.append(f"  score {e['score']:.1f}: " + ", ".join(
                f"{k}={p[k]}" for k in hp.PARAM_ORDER))
        tried_text = "\n".join(rows)
    best_text = "none yet"
    if best:
        best_text = ", ".join(f"{k}={best['params'][k]}" for k in hp.PARAM_ORDER)
        best_text += f"  (score {best['score']:.1f})"

    return (
        "You tune a 5G NR cell to maximize a score, using feedback from past "
        "evaluations.\n\n"
        + _scenario_block(sc)
        + "\nParameters and ranges:\n" + param_spec_text()
        + "\n\nALREADY EVALUATED (do not repeat):\n" + tried_text
        + "\n\nBEST SO FAR: " + best_text
        + "\n\n" + _goal_block(sc)
        + f"\nPropose {num_candidates} new configurations, most promising first, "
        "each differing from the evaluated ones and from each other in at least "
        "two parameters. Respond with ONLY a JSON array of objects, each with all "
        "parameters and a short \"reasoning\" field. No text outside the JSON."
    )


# ----------------------------------------------------------------------------
# Ollama client and parsing
# ----------------------------------------------------------------------------
def query_ollama(prompt: str, url: str, model: str, temperature: float,
                 seed: int = None) -> str:
    import requests
    options = {"temperature": temperature}
    if seed is not None:
        options["seed"] = int(seed)
    r = requests.post(
        f"{url}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": options},
        timeout=120,
    )
    r.raise_for_status()
    return r.json().get("response", "")


def _extract_json(text: str):
    text = text.strip()
    # try a JSON array first, then an object
    for opener, closer in (("[", "]"), ("{", "}")):
        i = text.find(opener)
        j = text.rfind(closer)
        if i != -1 and j != -1 and j > i:
            chunk = text[i:j + 1]
            try:
                return json.loads(chunk)
            except Exception:
                # strip trailing commas and retry
                cleaned = re.sub(r",\s*([}\]])", r"\1", chunk)
                try:
                    return json.loads(cleaned)
                except Exception:
                    continue
    return None


def _extract_by_keys(text: str) -> Dict:
    """Pull the known parameters out by key, ignoring overall JSON validity.

    Small models often break JSON inside the free-text reasoning field with an
    unescaped quote or a line break. The twelve parameters appear before it as
    plain key-value pairs, so they survive. This recovers them directly.
    """
    found = {}
    for name in hp.PARAM_ORDER:
        m = re.search(r'"' + re.escape(name) + r'"\s*:\s*"?([A-Za-z0-9_.\-]+)"?', text)
        if not m:
            continue
        raw = m.group(1)
        spec = hp.PARAM_SCHEMA[name]
        try:
            if spec["type"] == "cat":
                if isinstance(spec["values"][0], str):
                    found[name] = raw
                else:
                    found[name] = int(float(raw))
            elif spec["type"] == "int":
                found[name] = int(float(raw))
            else:
                found[name] = float(raw)
        except (ValueError, TypeError):
            continue
    return found


def _is_complete_config(item: Dict) -> bool:
    """Return True only when the model supplied every required parameter.

    Range violations are repaired by :func:`hard_problem.validate`, but missing
    parameters are not silently invented. This keeps a model run distinct from
    a default-filled or random fallback configuration.
    """
    return all(name in item and item[name] is not None for name in hp.PARAM_ORDER)


def parse_configs(text: str) -> List[Dict]:
    obj = _extract_json(text)
    if obj is not None:
        items = obj if isinstance(obj, list) else [obj]
        out = [
            hp.validate(item)
            for item in items
            if isinstance(item, dict) and _is_complete_config(item)
        ]
        if out:
            return out
    # Fallback: recover all twelve parameters by key from malformed JSON. A
    # partial recovery is rejected rather than completed with hidden defaults.
    found = _extract_by_keys(text)
    if _is_complete_config(found):
        return [hp.validate(found)]
    return []


# ----------------------------------------------------------------------------
# Mock model
# ----------------------------------------------------------------------------
def _mock_perturb(base: Dict, rng: random.Random) -> Dict:
    p = dict(base)
    for name in rng.sample(hp.PARAM_ORDER, 3):
        spec = hp.PARAM_SCHEMA[name]
        if spec["type"] == "cat":
            p[name] = rng.choice(spec["values"])
        elif spec["type"] == "int":
            span = max(1, (spec["high"] - spec["low"]) // 6)
            p[name] = p[name] + rng.randint(-span, span)
        else:
            step = 0.1 * (spec["high"] - spec["low"])
            p[name] = p[name] + rng.uniform(-step, step)
    return hp.validate(p)


# ----------------------------------------------------------------------------
# Zero-shot and search
# ----------------------------------------------------------------------------
def llm_zeroshot(scenario, url: str, model: str, mock: bool,
                 seed: int = 0) -> Dict:
    sc = hp.sc_lookup(scenario)
    if mock:
        return hp.knowledgeable_config(sc)
    text = query_ollama(build_zeroshot_prompt(sc), url, model, temperature=0.4, seed=seed)
    cfgs = parse_configs(text)
    if not cfgs:
        raise LLMParseError(
            f"Zero-shot reply could not be parsed for scenario {sc['id']}. "
            "No random fallback is used because that would confound the model result."
        )
    return cfgs[0]


def llm_search(scenario, budget: int, url: str, model: str, mock: bool,
               seed: int = 0, num_candidates: int = 3) -> Dict:
    """Iterative refinement. Evaluations against the scorer are capped at budget,
    matching the baselines. Starts from the zero-shot configuration.
    """
    sc = hp.sc_lookup(scenario)
    rng = random.Random(seed)

    tried: List[Dict] = []
    best = None
    history: List[float] = []
    best_score = -1e9

    def record(params):
        nonlocal best, best_score
        s = hp.evaluate(params, sc, seed=0)["score"]
        tried.append({"params": params, "score": s})
        if s > best_score:
            best_score = s
            best = {"params": params, "score": s}
        history.append(best_score)
        return s

    def is_tabu(params):
        xv = hp.encode_params(params)
        for e in tried:
            xe = hp.encode_params(e["params"])
            if max(abs(a - b) for a, b in zip(xv, xe)) < 0.05:
                return True
        return False

    # first evaluation is the zero-shot configuration
    record(llm_zeroshot(sc, url, model, mock, seed))

    while len(history) < budget:
        if mock:
            cands = [_mock_perturb(best["params"], rng) for _ in range(num_candidates)]
        else:
            temp = 0.3 if (len(history) >= 2 and history[-1] > history[-2]) else 1.0
            text = query_ollama(
                build_search_prompt(sc, tried, best, num_candidates),
                url, model, temperature=temp, seed=seed + len(history))
            cands = parse_configs(text)
            if not cands:
                raise LLMParseError(
                    f"Search reply could not be parsed for scenario {sc['id']} "
                    f"at iteration {len(history) + 1}."
                )
        # evaluate the first non-tabu candidate
        chosen = next((c for c in cands if not is_tabu(c)), None)
        if chosen is None:
            if mock:
                chosen = _mock_perturb(best["params"], rng)
            else:
                raise LLMParseError(
                    f"All proposed configurations were duplicates for scenario "
                    f"{sc['id']} at iteration {len(history) + 1}; no random "
                    "fallback is used."
                )
        record(chosen)

    return {"name": "LLM-search", "best_score": best_score,
            "best_params": best["params"], "history": history}


# ----------------------------------------------------------------------------
# Few-shot transfer. The model sees solved configurations for other scenarios
# and configures a held-out one. Examples are high-scoring reference configs.
# ----------------------------------------------------------------------------
def build_fewshot_prompt(sc: Dict, example_scs: List[Dict]) -> str:
    blocks = []
    for ex in example_scs:
        cfg = hp.knowledgeable_config(ex)
        score = hp.evaluate(cfg, ex, seed=0)["score"]
        cfg_json = {k: cfg[k] for k in hp.PARAM_ORDER}
        blocks.append(
            f"EXAMPLE SCENARIO: {ex['name']} ({ex['use_case']})\n"
            f"DESCRIPTION: {ex['desc']}\n"
            f"GOOD CONFIGURATION (score {score:.0f} out of 100): "
            f"{json.dumps(cfg_json)}\n")
    examples_text = "\n".join(blocks)
    return (
        "You configure 5G NR cells from their descriptions, using your knowledge "
        "of 5G NR. Here are solved examples for other scenarios, each with a "
        "high-scoring configuration. Study how the parameters relate to the "
        "scenario.\n\n"
        + examples_text
        + "\nNow configure this NEW scenario, applying the same reasoning:\n\n"
        + _scenario_block(sc)
        + "\nChoose values for every parameter:\n" + param_spec_text()
        + "\n\n" + _goal_block(sc)
        + "\nRespond with ONLY one JSON object with all parameters and a short "
        "\"reasoning\" field. No text outside the JSON."
    )


def llm_fewshot(scenario, examples, url: str, model: str, mock: bool,
                seed: int = 0) -> Dict:
    sc = hp.sc_lookup(scenario)
    ex_scs = [hp.sc_lookup(e) for e in examples]
    if mock:
        return hp.knowledgeable_config(sc)
    text = query_ollama(build_fewshot_prompt(sc, ex_scs), url, model,
                        temperature=0.4, seed=seed)
    cfgs = parse_configs(text)
    if not cfgs:
        raise LLMParseError(
            f"Few-shot reply could not be parsed for scenario {sc['id']}. "
            "No random fallback is used because that would confound the model result."
        )
    return cfgs[0]
