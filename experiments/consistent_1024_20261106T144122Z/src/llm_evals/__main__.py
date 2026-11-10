import argparse
import json
from pathlib import Path
from .checkpointing import atomic


def main():
    p = argparse.ArgumentParser(description='Canadian-ness Stage 1; local generation only')
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('discover')
    sub.add_parser('validate')
    sub.add_parser('status')
    for name in ('generate', 'smoke-test'):
        s = sub.add_parser(name)
        group = s.add_mutually_exclusive_group(required=True)
        group.add_argument('--model', choices=['gc_llm', 'apertus', 'llama'])
        group.add_argument('--all', action='store_true')
        s.add_argument('--resume', action='store_true')
    s = sub.add_parser('export')
    s.add_argument('--partial', action='store_true')
    args = p.parse_args()
    root = args.root.resolve()
    if args.command == 'discover':
        from .discovery import discover
        result = discover(root)
    elif args.command == 'validate':
        from .dataset import load, benchmark_path
        rows, headers, sha = load(benchmark_path(root))
        result = {'rows': len(rows), 'headers': headers, 'sha256': sha,
                  'languages': {lang: sum(r['language'] == lang for r in rows) for lang in ('en', 'fr')}}
        atomic(root / 'outputs/runs/validation.json', result)
    elif args.command == 'status':
        from .export import status
        result = status(root)
    elif args.command == 'export':
        from .export import export
        result = export(root, args.partial)
    else:
        from .generation import run
        result = {}
        errors = {}
        for model in (['gc_llm', 'apertus', 'llama'] if args.all else [args.model]):
            try:
                result[model] = run(root, model, args.command == 'smoke-test', args.resume)
            except Exception as exc:
                errors[model] = str(exc)
        if errors:
            print(json.dumps({'results': result, 'errors': errors}, indent=2))
            raise SystemExit(1)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
