"""Validate local collection metadata; never uploads photos or reads consent documents."""
import csv
from pathlib import Path
import argparse

def validate(catalog):
    rows=list(csv.DictReader(catalog.open(encoding="utf-8")))
    errors=[]; subjects={}; seen=set()
    for number,row in enumerate(rows,2):
        prefix=f"row {number}: "
        for key in ("session_id","subject_id","consent_ref"):
            if not row.get(key): errors.append(prefix+key+" required")
        if row.get("cut_code") not in {"MID_FADE","LOW_FADE","TAPER"}: errors.append(prefix+"unknown cut")
        if row.get("view") not in {"front","profile","back"}: errors.append(prefix+"unknown view")
        if row.get("consent_training") != "true" or row.get("professional_verified") != "true": errors.append(prefix+"permission and professional review required")
        split=row.get("split")
        if split not in {"train","validation","test"}: errors.append(prefix+"invalid split")
        subject=row.get("subject_id")
        if subject in subjects and subjects[subject] != split: errors.append(prefix+"subject leaks across splits")
        subjects[subject]=split
        key=(row.get("session_id"),row.get("view"))
        if key in seen: errors.append(prefix+"duplicate view")
        seen.add(key)
        for field in ("before_path","after_path"):
            path=(catalog.parent/row.get(field,"")).resolve()
            if not path.is_relative_to(catalog.parent.resolve()) or not path.is_file(): errors.append(prefix+field+" must be an existing local collection image")
    for session in {r.get("session_id") for r in rows}:
        if {view for sid,view in seen if sid==session} != {"front","profile","back"}: errors.append(f"session {session}: missing views")
    if not rows: errors.append("Collection is empty; add authorized sessions before training.")
    return errors

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("catalog",type=Path); args=parser.parse_args()
    errors=validate(args.catalog)
    print("\n".join(errors) if errors else "Collection metadata passed; visual review is still required.")
    raise SystemExit(bool(errors))
