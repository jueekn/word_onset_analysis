#!/usr/bin/env python
# Copied verbatim from Riley's cml_anatomy_standalone (cml_tools branch
# atlas_prioritization_robustness_check, 4d562e9), 2026-10-21. DB is set by fc.session_regions.
"""Recover whole-brain PAIR labels the DB (pairs.json) silently dropped, from the
label-extension database (raw imaging-pipeline files cmlreaders cannot load).

Sources, in priority order, for a pair's whole_brain label:
  1. 'imaging_pair_file'        — adjacent pair labeled directly in
                      electrodelabels_and_coordinates_mni_mid.csv (order-insensitive).
  2. 'imaging_midpoint_contact' — even-gap skip pair (e.g. LA1-LA3): label of the
                      midpoint contact (LA2) from
                      electrodenames_coordinates_native_and_T1.csv (this contact is
                      often non-recorded, so absent from contacts.json).
Returns (label, source); (None, None) if unrecoverable. Callers apply this only where
the DB label is missing.

Name matching (`norm_pair`/`norm_contact`) splits a pair into its two contacts
(`split_pair`, hyphen/underscore-safe — see below), UPPERCASES, and sorts them by
NUMERIC order (lower contact number first). Guarantees / guards:
  * order-insensitive + case-insensitive + space-insensitive (audited: 0 ambiguous
    collisions across the dataset; capitalization differs in 78% of matches yet always
    resolves to a unique label);
  * `_load` raises RecoveryNormalizationError if two DISTINCT raw names in an imaging
    file collapse to the same key with conflicting labels (never silently mis-map);
  * `assert_no_case_collisions` applies the same guard to the CONTACTS/PAIRS (DB) side
    — raising if two labels differ only by capitalization;
  * `dedupe_reversed_pairs` keeps the ascending (lower-first) order-variant.
Hyphenated/underscored contact names are handled: contacts that CONTAIN a hyphen
('RP-Thal1', 'RSO-RIP1', 'A-1') are split via the ' - ' imaging delimiter or, in the
no-space form, by remerging a 4-way '-' split (1+2, 3+4); underscores are never
delimiters. Only 2- and 4-part '-' splits occur among analysis pairs (imaging has a
single stray 3-split, which falls back to the whole uppercased string and matches
nothing — harmless).
"""
from __future__ import annotations

import collections
import os
import re

DB = os.path.expanduser("~/data/localization_pipeline_missing_label_extension")
PAIR_FILE = "electrodelabels_and_coordinates_mni_mid.csv"
CONTACT_FILE = "electrodenames_coordinates_native_and_T1.csv"
INVALID = {"", "nan", "no atlas", "unknown", "none", "clear label", "/"}


class RecoveryNormalizationError(ValueError):
    """Raised when name normalization is ambiguous — two DISTINCT raw contact/pair names
    collapse to the same normalized key but carry different labels. Matching by that key
    would be a coin-flip, so we refuse rather than silently mis-map (audited as never
    occurring in the current dataset; this is a permanent guard against regressions)."""


def _valid(v):
    return v is not None and str(v).strip().lower() not in INVALID


def split_contact(clab):
    m = re.match(r"^(.*?)(\d+)$", str(clab).strip())
    return (m.group(1), int(m.group(2))) if m else (None, None)


def _contact_key(c):
    """Sort key: (prefix, contact-number) so LA9 < LA10 (NUMERIC, not string)."""
    p, n = split_contact(c)
    return (p if p is not None else str(c), n if n is not None else -1)


def split_pair(name):
    """Return the pair's two contact names, or None if it can't be split confidently.
    Handles contact names that CONTAIN a hyphen (e.g. 'RP-Thal1', 'RSO-RIP1'):
      * imaging form 'A - B' has a space-padded ' - ' delimiter -> split on that;
      * no-space form 'A-B' splits on '-' into 2 parts; the both-contacts-hyphenated
        form 'RP-Thal1-RP-Thal2' splits into 4 -> remerge (1+2, 3+4).
    Underscores are never delimiters (they live inside contact names)."""
    if " - " in name:                                   # imaging 'A - B' (unambiguous)
        parts = name.split(" - ")
        if len(parts) == 2:
            return [p.replace(" ", "") for p in parts]
    parts = name.replace(" ", "").split("-")
    if len(parts) == 2:
        return parts
    if len(parts) == 4:                                 # each contact hyphenated once
        return [parts[0] + "-" + parts[1], parts[2] + "-" + parts[3]]
    return None                                         # 3/5/... splits: unparseable


def norm_pair(name):
    """Order-insensitive canonical key: split into the two contacts, UPPERCASE, and put
    them in ASCENDING numeric order (lower contact number first). 'LA10-LA9', 'LA9-LA10'
    and 'la9 - la10' all canonicalize to 'LA9-LA10'; 'RP-Thal1 - RP-Thal2' to
    'RP-THAL1-RP-THAL2'. Unsplittable names fall back to the whole uppercased string."""
    contacts = split_pair(name)
    if contacts is None:
        return name.replace(" ", "").upper()
    return "-".join(sorted((c.upper() for c in contacts), key=_contact_key))


def norm_contact(name):
    return name.replace(" ", "").upper()


def dedupe_reversed_pairs(pair_labels):
    """Given pair labels, drop reversed-order duplicates — when the same two contacts
    appear in both orders (e.g. 'LI9-LI10' and 'LI10-LI9'), keep the ASCENDING one
    (lower contact number first) and drop the higher-first one. Order preserved."""
    kept = {}
    for lab in pair_labels:
        k = norm_pair(lab)
        contacts = split_pair(lab)
        ascending = contacts is not None and [c.upper() for c in contacts] == k.split("-")
        if k not in kept or (ascending and not kept[k][1]):
            kept[k] = (lab, ascending)
    return [lab for lab, _ in kept.values()]


def _canon_casepreserved(name, is_pair):
    """Order-normalized but CASE-PRESERVED form (to tell case-only differences apart
    from mere order differences)."""
    if not is_pair:
        return name.replace(" ", "")
    contacts = split_pair(name)
    if contacts is None:
        return name.replace(" ", "")
    return "-".join(sorted(contacts, key=lambda c: _contact_key(c.upper())))


def assert_no_case_collisions(labels, is_pair=True, source="pairs.json"):
    """Guard the CONTACTS/PAIRS side (not just the imaging file): raise if two labels
    differ ONLY by capitalization and so collapse to the same normalized key — we could
    not then tell whether they are the same electrode or two distinct ones. Pure
    order-variants (same case) do NOT trigger. Returns the labels unchanged if safe."""
    groups = collections.defaultdict(set)
    for lab in labels:
        key = norm_pair(lab) if is_pair else norm_contact(lab)
        groups[key].add(_canon_casepreserved(lab, is_pair))
    for key, forms in groups.items():
        if len(forms) > 1:
            raise RecoveryNormalizationError(
                "%s: labels differing only by capitalization map to the same key %r: %s"
                % (source, key, sorted(forms)))
    return labels


def _load(subj, fname):
    """{normalized name: whole_brain label} from an imaging label file (col1 before '/').
    Raises RecoveryNormalizationError on an ambiguous normalization collision."""
    p = os.path.join(DB, subj, "imaging", fname)
    if not os.path.isfile(p):
        return {}
    out = {}
    for line in open(p):
        parts = line.split(",")
        if len(parts) < 2 or not parts[0].strip():
            continue
        raw = parts[0]
        key = norm_pair(raw) if "-" in raw else norm_contact(raw)
        # strip whitespace and stray CSV quotes from the label (e.g. '" Right Hippocampus')
        label = parts[1].split("/")[0].strip().strip('"').strip()
        if key in out and _valid(out[key]) and _valid(label) and out[key] != label:
            raise RecoveryNormalizationError(
                "%s/%s: distinct names normalize to %r with conflicting labels "
                "(%r vs %r)" % (subj, fname, key, out[key], label))
        if key not in out or not _valid(out[key]):   # prefer a valid label over invalid
            out[key] = label
    return out


def load_pair_wb(subj):
    return _load(subj, PAIR_FILE)


def load_contact_wb(subj):
    return _load(subj, CONTACT_FILE)


def recover_pair_label(pair_label, pair_wb, contact_wb):
    """(label, source) for one pair, or (None, None). pair_wb/contact_wb from load_*."""
    key = norm_pair(pair_label)
    if key in pair_wb and _valid(pair_wb[key]):
        return pair_wb[key], "imaging_pair_file"
    contacts = split_pair(pair_label)
    if contacts is not None:
        (p1, n1), (p2, n2) = split_contact(contacts[0]), split_contact(contacts[1])
        if None not in (p1, n1, p2, n2) and p1 == p2 and (n1 + n2) % 2 == 0:
            mid = norm_contact("{0}{1}".format(p1, (n1 + n2) // 2))
            if mid in contact_wb and _valid(contact_wb[mid]):
                return contact_wb[mid], "imaging_midpoint_contact"
    return None, None


def fill_pair_wb(pair_label, current_label, pair_wb, contact_wb):
    """FILL-ONLY recovery (the integration rule): keep a valid existing label untouched
    and recover ONLY where the current label is missing/invalid. This guarantees no
    already-valid electrode label is altered.
    Returns (label, source); source is 'existing_db_label' when the current label is
    kept, 'imaging_pair_file'/'imaging_midpoint_contact' when recovered, or None if
    still unrecoverable."""
    if _valid(current_label):
        return current_label, "existing_db_label"
    return recover_pair_label(pair_label, pair_wb, contact_wb)


if __name__ == "__main__":
    # smokescreen on known cases: R1135E (skip-heavy), R1286J (Bug A), R1062J
    for subj, pairs in [("R1135E", ["LAMYD1-LAMYD3", "LHCD1-LHCD3"]),
                        ("R1286J", ["LA1-LA2", "LA2-LA3"]),
                        ("R1062J", ["RA1-RA2", "RF1-RF2"])]:
        pw, cw = load_pair_wb(subj), load_contact_wb(subj)
        print(f"{subj}: pair_file entries={len(pw)}, contact_file entries={len(cw)}")
        for pl in pairs:
            print(f"    {pl:16s} -> {recover_pair_label(pl, pw, cw)}")
