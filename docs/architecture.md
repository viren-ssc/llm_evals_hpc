# Architecture

`dataset.py` verifies the immutable CORE workbook and validates IDs, question text, required context/reference fields, languages and metric labels. `discovery.py` inspects local checkpoint metadata and safetensors shard structure without changing shared caches.

`inference.py` is the GPU boundary. It accepts only a question string, constructs one role-only user request and uses the verified checkpoint. `generation.py` manages run fingerprints, resume, batching and per-answer metadata. The active configuration disables smoke inference and retains nonempty capped output. Historical versions retain their own continuation policies.

`batching.py` bounds independent requests using measured KV-cache capacity and reserved output tokens. `checkpointing.py` uses worker locks and atomic writes. Manifests pin model/config identity, source hashes and runtime details. `export.py` joins by identity and checks original question text and workbook checksum; final export gates follow the experiment's success policy.

The root application writes new data to `outputs/`; archived runs live under `experiments/<name>/outputs/`. Historical code snapshots remain beside their configs so that their fingerprints and continuation behavior stay interpretable. Original data/result bytes were preserved in reorganization.

Part 2 is a separate repository running the judge in icelab. It receives exported answers, context, references and provenance. No judge runs in this repository.
