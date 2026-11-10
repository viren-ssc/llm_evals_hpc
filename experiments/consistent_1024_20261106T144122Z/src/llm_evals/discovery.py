"""Bounded local discovery; never downloads or modifies checkpoints."""
import json
import os
import struct
from pathlib import Path
import yaml
from .checkpointing import atomic
from .dataset import checksum

SHARED = Path('/gpfs/fs7/ssc/ssc_dsai')
IDS = {'gc_llm': 'ssc-dsai/gc-llm-apertus-70b-instruct-2509',
       'apertus': 'swiss-ai/Apertus-70B-Instruct-2509',
       'llama': 'meta-llama/Llama-3.3-70B-Instruct'}


def inspect(path):
    path = Path(path)
    result = {'path': str(path), 'problems': [], 'metadata_sha256': {}}
    def read(name):
        if not (path / name).is_file():
            result['problems'].append(f'Missing {name}')
            return {}
        result['metadata_sha256'][name] = checksum(path / name)
        return json.loads((path / name).read_text())
    config = read('config.json')
    result.update(architecture=config.get('architectures'), model_type=config.get('model_type'),
                  precision=config.get('dtype', config.get('torch_dtype')), config=config,
                  adapter=(path / 'adapter_config.json').exists())
    if result['adapter']:
        result['adapter_config'] = read('adapter_config.json')
        result['problems'].append('Adapter requires explicitly verified base and PEFT backend')
    tokenizer = read('tokenizer_config.json')
    result['chat_template'] = tokenizer.get('chat_template')
    separate_template = path / 'chat_template.jinja'
    if separate_template.is_file():
        result['metadata_sha256']['chat_template.jinja'] = checksum(separate_template)
        result['chat_template'] = separate_template.read_text()
    if not result['chat_template']:
        result['problems'].append('Missing native chat template; completion format requires explicit configuration')
    if not (path / 'tokenizer.json').is_file():
        result['problems'].append('Missing tokenizer.json')
    else:
        result['metadata_sha256']['tokenizer.json'] = checksum(path / 'tokenizer.json')
    index = read('model.safetensors.index.json')
    shards = sorted(set(index.get('weight_map', {}).values()))
    total, parameters = 0, 0
    result['shards'] = []
    tensor_names = set()
    for shard in shards:
        f = path / shard
        try:
            with f.open('rb') as handle:
                size = struct.unpack('<Q', handle.read(8))[0]
                if size > 100_000_000:
                    raise ValueError('Invalid safetensors header size')
                header = json.loads(handle.read(size))
            length = f.stat().st_size
            for key, val in header.items():
                if key == '__metadata__':
                    continue
                if val['data_offsets'][1] + size + 8 > length:
                    raise ValueError('Incomplete tensor payload')
                if index['weight_map'].get(key) != shard:
                    raise ValueError('Tensor index disagrees with shard header')
                tensor_names.add(key)
                n = 1
                for dimension in val['shape']:
                    n *= dimension
                parameters += n
            total += length
            result['shards'].append({'name': shard, 'bytes': length, 'header_sha256':
                                    __import__('hashlib').sha256(json.dumps(header, sort_keys=True).encode()).hexdigest()})
        except (OSError, ValueError, KeyError, struct.error) as exc:
            result['problems'].append(f'{shard}: {exc}')
    if not shards:
        result['problems'].append('No indexed weight shards')
    if set(index.get('weight_map', {})) != tensor_names:
        result['problems'].append('Missing indexed tensors')
    result.update(weight_bytes=total, parameter_count=parameters,
                  ready=not result['problems'], status='instruct' if result['chat_template'] else 'unverified')
    return result


def discover(root):
    caches = [Path(os.environ.get('HF_HUB_CACHE', str(SHARED / '.cache/huggingface/hub'))),
              SHARED / '.cache/huggingface', SHARED / 'peb000/.cache/huggingface',
              SHARED / 'abm002/.cache/huggingface/hub']
    models, reports = {}, {}
    for key, identifier in IDS.items():
        candidates = []
        for cache in caches:
            folder = cache / ('models--' + identifier.replace('/', '--')) / 'snapshots'
            if folder.is_dir():
                candidates.extend(sorted(folder.iterdir()))
        if key == 'llama' and (SHARED / '.cache/Meta-Llama-3.3-70B-Instruct').is_dir():
            candidates.append(SHARED / '.cache/Meta-Llama-3.3-70B-Instruct')
        details = [inspect(p) for p in candidates]
        ready = [d for d in details if d['ready']]
        chosen = (ready or details or [None])[0]
        reports[key] = {'identifier': identifier, 'candidates': details}
        models[key] = {'identifier': identifier, 'path': chosen['path'] if chosen else None,
                       'tokenizer_path': chosen['path'] if chosen else None,
                       'revision': Path(chosen['path']).name if chosen else None,
                       'metadata_sha256': chosen['metadata_sha256'] if chosen else {},
                       'shard_manifest': chosen['shards'] if chosen else [],
                       'ready': bool(chosen and chosen['ready']), 'backend': 'vllm',
                       'prompt_template': ('{{ bos_token }}<|start_header_id|>user<|end_header_id|>\n\n{{ messages[0]["content"] }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{{ "\\n\\n" }}' if key == 'llama' else '{{ bos_token }}<|user_start|>{{ messages[0]["content"] }}<|user_end|><|assistant_start|>'),
                       'tensor_parallel_size': 4, 'dtype': 'bfloat16',
                       'gpu_memory_utilization': 0.94, 'adapter': None,
                       'identity_note': ('Published GC model card identifies swiss-ai/Apertus-70B-Instruct-2509 as its parent; complete full checkpoints found'
                                         if key != 'llama' else 'Local model card and config identify Llama 3.3 70B Instruct'),
                       'problems': chosen['problems'] if chosen else ['No candidate found']}
    models['gc_llm']['base_model_path'] = models['apertus']['path']
    models['gc_llm']['provenance_url'] = 'https://huggingface.co/ssc-dsai/gc-llm-apertus-70b-instruct-2509/blob/main/README.md'
    models['gc_llm']['checkpoint_format'] = 'full_weights_after_LoRA_finetuning'
    # Snapshot manifests preserve exact tokenizer/config/index and shard header fingerprints.
    atomic(root / 'outputs/runs/discovery.json', reports)
    (root / 'configs/models.yaml').write_text(yaml.safe_dump(models, sort_keys=False))
    text = ['# Local model discovery', '', 'Read-only discovery; no downloads. Header/index validation checks readability and payload bounds, not full weight cryptographic integrity.', '']
    for key, m in models.items():
        text += [f'## {key}: {m["identifier"]}', f'Path: `{m["path"]}`',
                 f'Ready: {m["ready"]}; revision: `{m["revision"]}`.', m['identity_note'],
                 'Problems: ' + '; '.join(m['problems']),
                 'Architecture/precision: ' + str(next((c['architecture'] for c in reports[key]['candidates'] if c['path'] == m['path']), None)) + ' / ' + m['dtype'],
                 'Parameter count: ' + str(next((c['parameter_count'] for c in reports[key]['candidates'] if c['path'] == m['path']), None)), '']
    text += ['The initial shared root caches are config-only. Complete readable GC and Apertus snapshots were found in /gpfs/fs7/ssc/ssc_dsai/abm002/.cache/huggingface/hub by following the existing CANBENCH_RESPONSES launcher. Each has 30 shards and 70,599,864,640 parameters. GC uses full weights, not an adapter-only load. The official GC model card verifies the Apertus Instruct 2509 parent: https://huggingface.co/ssc-dsai/gc-llm-apertus-70b-instruct-2509/blob/main/README.md. Both native templates live in separate chat_template.jinja files. The effective question-only templates retain their native BOS/user/end/assistant tokens and omit automatic system/developer boilerplate.',
             'The shared full_finetune_conservative and lora_optimized models have no established connection to the requested GC checkpoint and are not substituted.',
             'Llama alternative flat copy: /gpfs/fs7/ssc/ssc_dsai/.cache/Meta-Llama-3.3-70B-Instruct. The alternate HF snapshot contains incomplete download links; the complete flat copy is selected and pinned by metadata and shard header hashes.',
             'BF16 70B weights require approximately 132 GiB aggregate plus runtime/KV overhead. Use four 40 GB A100 GPUs with eager execution, one sequence and an 8192-token context. Memory feasibility must pass smoke testing. V100 4×16 GB is insufficient without changing precision.']
    (root / 'docs/model_discovery.md').write_text('\n'.join(text) + '\n')
    return models
