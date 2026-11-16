"""CuratorKIT prompt curation with bilingual and held-out-split safeguards."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from curatorkit.normalizers import ExactDeduplicator, TextCleaner
from curatorkit.schema import DataSample

try:
    from .pilot_validation import validate_rows
except ImportError:  # Supports sibling imports from scripts executed directly.
    from pilot_validation import validate_rows


CLEANING_TRANSFORMS = {
    "strip_html": False,
    "normalise_unicode": True,
    "fix_encoding_artifacts": False,
    "collapse_whitespace": True,
    "remove_control_chars": True,
}


def _duplicate_key(sample: DataSample) -> str:
    # ExactDeduplicator's prompt_only key is instruction + empty output,
    # lowercased and whitespace-collapsed using str.split(). Metadata is absent
    # from that key. Keep this boundary explicit rather than using private APIs.
    return " ".join(sample.instruction.lower().split())


def curate_rows(rows: list[dict], source_uri: str) -> tuple[list[dict], dict]:
    """Return independent curated row copies and a JSON-serializable manifest.

    Only request text changes. Raw rows, nested metadata and existing provenance
    are preserved. Cross-split duplicates within a language raise before any
    deduplication. Within-split deduplication keeps the first sample, separately
    for each language, and must leave complete pilot families. No safety or
    translation quality judgment is made by this structural curation.
    """
    validate_rows(rows)
    if not isinstance(source_uri, str) or not source_uri.strip():
        raise ValueError("source_uri must be a nonempty string")

    originals = {row["id"]: row for row in rows}
    samples = [
        DataSample(
            id=row["id"],
            source_uri=source_uri,
            task_type="prompt_only",
            instruction=row["request"],
            metadata={"pilot_row": deepcopy(row)},
        )
        for row in rows
    ]
    cleaned = TextCleaner(transforms=CLEANING_TRANSFORMS, fields=["instruction"]).run(samples)

    seen_across_splits: dict[tuple[str, str], tuple[str, str]] = {}
    groups: dict[tuple[str, str], list[DataSample]] = defaultdict(list)
    for sample in cleaned:
        row = originals[sample.id]
        language, split = row["language"], row["split"]
        key = (language, _duplicate_key(sample))
        previous = seen_across_splits.get(key)
        if previous is not None and previous[0] != split:
            raise ValueError(
                f"Cross-split duplicate in {language}: {previous[1]!r} ({previous[0]}) "
                f"and {sample.id!r} ({split})"
            )
        seen_across_splits.setdefault(key, (split, sample.id))
        groups[(language, split)].append(sample)

    surviving_ids: set[str] = set()
    duplicate_of: dict[str, str] = {}
    for group in groups.values():
        first_ids: dict[str, str] = {}
        for sample in group:
            key = _duplicate_key(sample)
            if key in first_ids:
                duplicate_of[sample.id] = first_ids[key]
            else:
                first_ids[key] = sample.id
        surviving_ids.update(sample.id for sample in ExactDeduplicator().run(group))

    curated = []
    for sample in cleaned:
        if sample.id in surviving_ids:
            row = deepcopy(originals[sample.id])
            row["request"] = sample.instruction
            curated.append(row)
    # Do not silently publish an unpaired family after within-split deduplication.
    validate_rows(curated)

    manifest = {
        "source_uri": source_uri,
        "input_count": len(rows),
        "output_count": len(curated),
        "changed_ids": [sample.id for sample in cleaned if sample.instruction != originals[sample.id]["request"]],
        "removed_ids": [sample.id for sample in cleaned if sample.id not in surviving_ids],
        "cleaning_transforms": dict(CLEANING_TRANSFORMS),
        "deduplication_group_by": ["language", "split"],
        "per_sample": {
            sample.id: {
                "source_uri": sample.source_uri,
                "status": "kept" if sample.id in surviving_ids else "removed",
                "duplicate_of": duplicate_of.get(sample.id),
                "provenance_chain": [record.model_dump(mode="json") for record in sample.provenance_chain],
            }
            for sample in cleaned
        },
    }
    return curated, manifest
