"""Continuous request admission bounded by worst-case KV token reservations."""
import math
import re
from pathlib import Path
import time


def reservation(input_tokens, output_tokens, block_size):
    return math.ceil((input_tokens + output_tokens) / block_size) * block_size


def stream_requests(engine, requests, max_sequences, token_budget, block_size, stopped=lambda: False):
    """Yield each finished request immediately; refill freed slots on the next step.

    Each request is independent. ``params`` is its own decoding configuration.
    Capacity reserves the entire possible response, never truncating a question.
    """
    waiting = iter(requests)
    candidate = next(waiting, None)
    active, seen = {}, set()
    reserved = 0
    peak = 0
    try:
        while candidate is not None or active:
            while candidate is not None and not stopped() and len(active) < max_sequences:
                request = candidate
                needed = reservation(len(request['ids']), request['max_tokens'], block_size)
                if needed > token_budget:
                    raise ValueError('One complete request exceeds the reserved KV token budget')
                if reserved + needed > token_budget:
                    break
                qid = request['question_id']
                if qid in seen:
                    raise ValueError(f'Duplicate request ID: {qid}')
                seen.add(qid)
                engine.add_request(qid, {'prompt_token_ids': request['ids']}, request['params'])
                active[qid] = {'request': request, 'reserved': needed, 'started': time.monotonic(),
                               'concurrency_at_submit': len(active) + 1}
                reserved += needed
                peak = max(peak, len(active))
                print(f'BATCH_ADMISSION active={len(active)} reserved_tokens={reserved}/{token_budget} peak={peak}', flush=True)
                candidate = next(waiting, None)
            if not active:
                break
            for result in engine.step():
                if not result.finished:
                    continue
                item = active.pop(result.request_id)
                reserved -= item['reserved']
                yield item['request'], result, {'generation_seconds': time.monotonic() - item['started'],
                                                'concurrency_at_submit': item['concurrency_at_submit'],
                                                'peak_concurrency': peak}
            # A stop request drains active answers but does not submit new ones.
    finally:
        if active:
            engine.abort_request(list(active))


def profiled_cache_tokens(log_path):
    samples = re.findall(r'GPU KV cache size: ([\d,]+) tokens',
                         Path(log_path).read_text(errors='replace'))
    if not samples:
        raise RuntimeError('Current engine KV-cache profile is unavailable; refusing unbounded batching')
    return int(samples[-1].replace(',', ''))
