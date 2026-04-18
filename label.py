import json
import os
import re
import torch
from collections import namedtuple, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

try:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    _VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None
    GuidedDecodingParams = None
    _VLLM_AVAILABLE = False
from tqdm import tqdm

def extract_final_answer(gt_text):
    """Extract the final numeric answer from the ground truth."""
    numbers = re.findall(r'\d+', gt_text)
    return numbers[-1] if numbers else gt_text


def split_solutions(response):
    """Split the response into 'Solution X:' sections."""
    if "===== Solution 1 =====" in response:
        parts = [s.strip() for s in re.split(r'===== Solution \d+ =====', response) if s.strip()]
        
        solutions = [{"id": i + 1, "text": text} for i, text in enumerate(parts)]
    
    else:
        parts = re.split(r'(Solution\s+\d+:)', response)

        if len(parts) == 1:
            return [{"id": 1, "text": response.strip()}]

        solutions = []
        current_id = None
        current_text = []

        for part in parts:
            header_match = re.match(r'Solution\s+(\d+):', part)
            if header_match:
                if current_id is not None:
                    solutions.append({"id": current_id, "text": "".join(current_text).strip()})
                current_id = int(header_match.group(1))
                current_text = []
            else:
                current_text.append(part)

        if current_id is not None:
            solutions.append({"id": current_id, "text": "".join(current_text).strip()})

    return solutions


def build_label_prompt(query, solution_text, gt_answer, dataset_name):
    """Make the labeling prompt."""
    
    SAMPLE_ANSWER_START_STRING = "## Sample Answers (Use them to guide the style of your answer)"
    SAMPLE_ANSWER_END_STRING = "--- End of Sample Answers ---"        
    
    
    prompt = re.sub(
        rf"{re.escape(SAMPLE_ANSWER_START_STRING)}.*?\n{SAMPLE_ANSWER_END_STRING}",
        "",
        query,
        flags=re.DOTALL,
    )
    
    part2 = ""
    if dataset_name == "QMSum":
        assert "## Meeting Transcript" in prompt
        parts = prompt.split("## Meeting Transcript")
        assert len(parts) == 2
        query, part2 = parts
        
    elif dataset_name == "QASPER":
        assert "# Paper Content" in prompt
        parts = prompt.split("# Paper Content")
        assert len(parts) == 2
        query, part2 = parts
        part2 = part2.split("## Conclusion")[0].strip()
    
    prompt = f"""
You are labeling whether a solution correctly solves a question.
--------------------------------
# QUESTION
{query}
--------------------------------
# GROUND TRUTH FINAL ANSWER:
{gt_answer}
--------------------------------
# WHAT YOU SHOULD DO
Does the candidate solution given below contain the correct final answer?
Respond ONLY with: true or false

Rules
- If the ground truth is a number, the solution should match in numeric value.
- If the ground truth is a person's name, the solution might slightly differ, e.g. `Richard Bertrand Spencer`, `Richard B. Spencer`, `Spencer` are all the same person.
--------------------------------
# CANDIDATE SOLUTION:
{solution_text}
""".strip()

    if dataset_name == "QMSum":
        
        prompt += f"""

--------------------------------
## Meeting Transcript
{part2}
"""
        # transcript = parts[0].split("\n")[:1200]
        # transcript = "\n".join(transcript)
        # new_prompt = "--------------------------------".join([""] + parts[1:] + [""]) + "## Meeting Transcript" + transcript
        # prompt = new_prompt
        
    elif dataset_name == "QASPER":
        prompt += f"""
--------------------------------
# Paper Content
{part2}
"""

    return prompt

def truncate_for_vllm(prompt, tokenizer, max_model_len=2048, keep_tail_tokens=256, dataset_name=None):
    """
    Truncate a prompt so that AFTER applying the chat template,
    total tokens <= max_model_len.
    """
    
    chat = [{"role": "user", "content": prompt}]
    full_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True
    )

    raw_ids = tokenizer.encode(prompt)
    keep_head = max_model_len - keep_tail_tokens
    if keep_head < 0:
        keep_head = max_model_len

    truncated_ids = raw_ids[:keep_head] + raw_ids[-keep_tail_tokens:]
    truncated_prompt = tokenizer.decode(truncated_ids)

    chat = [{"role": "user", "content": truncated_prompt}]
    full_ids = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True
    )
    if len(full_ids) > max_model_len:
        print(f"Truncated prompt is too long: {len(full_ids)} > {max_model_len}")
        # Hard fallback: keep only last tokens
        truncated_ids = raw_ids[-keep_tail_tokens:]
        truncated_prompt = tokenizer.decode(truncated_ids)

    return truncated_prompt + "\n\n...[TRUNCATED]..."

def label_file_with_vllm(
    input_path,
    output_path,
    model_name="Qwen/Qwen2.5-72B-Instruct",
    start=0,
    end=None,
    max_tokens=5,
    max_model_len=40960,
    dataset_name=None,
):
    print(f"Loading model: {model_name}")
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.95,
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=max_model_len,
        max_num_seqs=4
    )

    tokenizer = llm.get_tokenizer()

    guided_decoding = GuidedDecodingParams(choice=["true", "false"])
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        guided_decoding=guided_decoding,
    )

    # Load data
    with open(input_path, "r", encoding="utf-8") as fin:
        all_lines = [json.loads(l) for l in fin]

    lines = all_lines[start:end]
    print(f"Processing {len(lines)} items (indices {start} to {start + len(lines) - 1})")

    # ===== PHASE 1: COLLECT ALL PROMPTS =====
    PromptMetadata = namedtuple('PromptMetadata', ['prompt', 'line_idx', 'solution', 'item'])
    all_prompts_metadata = []

    for line_idx, item in enumerate(tqdm(lines, desc="Processing"), start=start):
        if dataset_name in {"Math", "GSM8K"}:
            gt_answer = extract_final_answer(item["gt"])
        else:
            gt_answer = item["gt"]
        solutions = split_solutions(item["response"])

        for solution in solutions:
            full_prompt = build_label_prompt(item["query"], solution["text"], gt_answer, dataset_name=dataset_name)
            truncated = truncate_for_vllm(
                full_prompt,
                tokenizer,
                max_model_len=max_model_len,
                keep_tail_tokens=256,
                dataset_name=dataset_name
            )

            all_prompts_metadata.append(PromptMetadata(
                prompt=truncated,
                line_idx=line_idx,
                solution=solution,
                item=item
            ))

    print(f"Collected {len(all_prompts_metadata)} prompts across {len(lines)} items")

    # ===== PHASE 2: BATCH INFERENCE =====
    prompts = [m.prompt for m in all_prompts_metadata]
    print(f"Running batch inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling_params)

    assert len(outputs) == len(prompts), f"Output/prompt mismatch: {len(outputs)} vs {len(prompts)}"

    # ===== PHASE 3: RECONSTRUCT RESULTS =====
    results_by_line = defaultdict(list)

    for output, metadata in zip(outputs, all_prompts_metadata):
        pred = output.outputs[0].text.strip().lower()
        if pred not in ["true", "false"]:
            pred = None
        is_correct = (pred == "true")

        solution_result = {
            "id": metadata.solution["id"],
            "text": metadata.solution["text"],
            "is_correct": is_correct,
        }

        results_by_line[metadata.line_idx].append({
            'solution': solution_result,
            'item': metadata.item
        })

    # Build final output structures
    final_results = []
    for line_idx in sorted(results_by_line.keys()):
        line_data = results_by_line[line_idx]
        item = line_data[0]['item']
        solutions = [entry['solution'] for entry in line_data]

        solutions_sorted = sorted(solutions, key=lambda x: x["id"])
        labels = [s["is_correct"] for s in solutions_sorted]

        result = {
            "query": item["query"],
            "gt": item["gt"],
            "solutions": solutions_sorted,
            "labels": labels,
            "tag": item.get("tag"),
            "source": item.get("source"),
        }
        final_results.append(result)

    # ===== PHASE 4: WRITE OUTPUT =====
    print(f"Writing {len(final_results)} labeled items to {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        for result in final_results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Done. Saved labeled file to: {output_path}")


def _parse_true_false(text):
    """Parse 'true'/'false' out of a free-form LLM response."""
    if text is None:
        return None
    lowered = text.strip().lower()
    # Match the first standalone true/false token.
    m = re.search(r"\b(true|false)\b", lowered)
    if m is None:
        return None
    return m.group(1) == "true"


def _call_api_once(prompt, model_name, model_url, api_key, timeout, max_tokens=8, temperature=0.0):
    """Single OpenAI-compatible chat-completions call. Returns response text or ''."""
    headers = {"Content-Type": "application/json"}
    if api_key and api_key != "ollama":
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(model_url, headers=headers,
                             data=json.dumps(payload), timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[api] call failed: {e}")
        return ""


def label_file_with_api(
    input_path,
    output_path,
    model_api_config_path,
    model_name,
    start=0,
    end=None,
    dataset_name=None,
    max_workers=None,
    request_timeout=120,
):
    """Label via an OpenAI-compatible endpoint (Ollama, LM Studio, OpenAI, etc.).

    Writes the same output schema as ``label_file_with_vllm``.
    """
    with open(model_api_config_path, "r") as f:
        cfg = json.load(f)
    if model_name not in cfg:
        raise KeyError(
            f"Model '{model_name}' not in {model_api_config_path}. "
            f"Available keys: {list(cfg.keys())}"
        )
    model_cfg = cfg[model_name]
    model_list = model_cfg["model_list"]
    workers = max_workers or (model_cfg.get("max_workers_per_model", 4) * len(model_list))
    print(f"[api] using model={model_name} endpoints={len(model_list)} workers={workers}")

    with open(input_path, "r", encoding="utf-8") as fin:
        all_lines = [json.loads(l) for l in fin]
    lines = all_lines[start:end]
    print(f"[api] processing {len(lines)} items (indices {start} to {start + len(lines) - 1})")

    PromptMeta = namedtuple("PromptMeta", ["prompt", "line_idx", "solution", "item"])
    all_prompts = []

    for line_idx, item in enumerate(tqdm(lines, desc="Building prompts"), start=start):
        if dataset_name in {"Math", "MATH", "GSM8K"}:
            gt_answer = extract_final_answer(item["gt"])
        else:
            gt_answer = item["gt"]
        for solution in split_solutions(item["response"]):
            prompt = build_label_prompt(
                item["query"], solution["text"], gt_answer, dataset_name=dataset_name
            )
            all_prompts.append(PromptMeta(prompt, line_idx, solution, item))

    print(f"[api] collected {len(all_prompts)} solution prompts")

    # Round-robin endpoint picker across workers.
    responses = [None] * len(all_prompts)

    def _do(i):
        ep = model_list[i % len(model_list)]
        return i, _call_api_once(
            all_prompts[i].prompt,
            ep["model_name"], ep["model_url"], ep.get("api_key", ""),
            timeout=request_timeout,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_do, i) for i in range(len(all_prompts))]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Labeling"):
            i, text = fut.result()
            responses[i] = text

    # Reconstruct per-line outputs.
    results_by_line = defaultdict(list)
    for meta, text in zip(all_prompts, responses):
        verdict = _parse_true_false(text)
        results_by_line[meta.line_idx].append({
            "solution": {
                "id": meta.solution["id"],
                "text": meta.solution["text"],
                "is_correct": (verdict is True),
                "_raw_verdict": text.strip()[:80],
            },
            "item": meta.item,
        })

    final_results = []
    for line_idx in sorted(results_by_line.keys()):
        entries = results_by_line[line_idx]
        item = entries[0]["item"]
        solutions_sorted = sorted([e["solution"] for e in entries], key=lambda x: x["id"])
        labels = [s["is_correct"] for s in solutions_sorted]
        final_results.append({
            "query": item["query"],
            "gt": item["gt"],
            "response": item.get("response", ""),
            "solutions": solutions_sorted,
            "labels": labels,
            "tag": item.get("tag"),
            "source": item.get("source"),
        })

    print(f"[api] writing {len(final_results)} labeled items to {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        for result in final_results:
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[api] done. Saved labeled file to: {output_path}")
    return output_path


def preview_labeled(path, n=2, max_chars=400):
    """Pretty-print the first N labeled items so you can eyeball
    'original response vs split+labeled solutions'.
    """
    print("\n" + "=" * 72)
    print(f"PREVIEW: {path} (showing up to {n} items)")
    print("=" * 72)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            item = json.loads(line)
            print(f"\n----- ITEM {i} -----")
            query = str(item.get("query", ""))
            gt = str(item.get("gt", ""))
            print(f"QUERY    : {query[:max_chars]}{'...' if len(query) > max_chars else ''}")
            print(f"GROUND T.: {gt[:max_chars]}{'...' if len(gt) > max_chars else ''}")
            resp = str(item.get("response", ""))
            if resp:
                print(f"\nORIGINAL RESPONSE (first {max_chars} chars):")
                print(resp[:max_chars] + ("..." if len(resp) > max_chars else ""))
            print(f"\nPARSED {len(item.get('solutions', []))} SOLUTION(S):")
            for sol in item.get("solutions", []):
                verdict = "TRUE " if sol.get("is_correct") else "false"
                raw = sol.get("_raw_verdict", "")
                text = str(sol.get("text", ""))[:max_chars]
                print(f"  [{sol.get('id')}] {verdict}  raw={raw!r}")
                print(f"       text: {text}{'...' if len(sol.get('text', '')) > max_chars else ''}")
            print(f"\nLABELS   : {item.get('labels')}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)

    # Max model length for Qwen2.5-72B-Instruct is 32768
    parser.add_argument("--max_model_len", type=int, default=32768)

    # macOS / API-backed labeling.
    parser.add_argument(
        "--backend",
        choices=["api", "vllm"],
        default="api",
        help="'api' routes labeling through an OpenAI-compatible endpoint (Ollama, "
             "LM Studio, OpenAI) and works on macOS. 'vllm' is the original CUDA path.",
    )
    parser.add_argument(
        "--model_api_config",
        default="model_api_configs/model_api_config.json",
        help="Path to the model_api_config.json used by the API backend.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Override worker count for the API backend. Defaults to the "
             "max_workers_per_model * len(model_list) from the config.",
    )
    parser.add_argument(
        "--request_timeout",
        type=int,
        default=120,
        help="Per-request HTTP timeout in seconds for the API backend.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="After labeling, print the first N items so you can eyeball the "
             "original response vs the split + labeled solutions. 0 = no preview.",
    )

    args = parser.parse_args()

    if not args.input.endswith("infer.jsonl"):
        print(f"[warn] --input does not end with 'infer.jsonl' ({args.input}); continuing anyway.")
    if args.dataset_name not in args.input:
        print(f"[warn] --dataset_name '{args.dataset_name}' not found in --input path; continuing anyway.")

    model_tag = args.model.split("/")[-1].replace(":", "_")
    if args.input.endswith("infer.jsonl"):
        args.output = args.input.replace("infer.jsonl", f"infer_labeled_by_{model_tag}.jsonl")
    else:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_labeled_by_{model_tag}{ext or '.jsonl'}"

    format_map = {
        "QMSum": "short_answer",
        "HotpotQA": "short_answer",
        "NarrativeQA": "short_answer",
        "QASPER": "short_answer",
        "MedMCQA": "multiple_choice",
        "MedQA": "multiple_choice",
        "MMLU": "multiple_choice",
        "GSM8K": "short_answer",
    }

    if args.backend == "vllm":
        if not _VLLM_AVAILABLE:
            raise RuntimeError(
                "--backend vllm was requested but vllm is not installed. "
                "On macOS, use --backend api (the default) with an OpenAI-compatible "
                "endpoint such as Ollama."
            )
        label_file_with_vllm(
            input_path=args.input,
            output_path=args.output,
            model_name=args.model,
            start=args.start,
            end=args.end,
            max_model_len=args.max_model_len,
            dataset_name=args.dataset_name,
        )
    else:
        label_file_with_api(
            input_path=args.input,
            output_path=args.output,
            model_api_config_path=args.model_api_config,
            model_name=args.model,
            start=args.start,
            end=args.end,
            dataset_name=args.dataset_name,
            max_workers=args.max_workers,
            request_timeout=args.request_timeout,
        )

    if args.preview > 0:
        preview_labeled(args.output, n=args.preview)
