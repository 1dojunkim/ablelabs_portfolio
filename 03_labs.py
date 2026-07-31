"""
논문 기반 액체 핸들링 자동화 타겟 리드 발굴 파이프라인
3단계: 연구실 단위 집계 (개선판)

원본 03_labs.py 대비 변경점:
- 문제: 같은 사람의 논문이라도 일부에만 이메일이 있으면, 이메일 유무로
  lab_key가 갈려 한 연구실이 두 개 이상의 lab_id로 쪼개졌다(fragmentation).
  예: Ae-Son Om (Hanyang University) — 이메일 있는 논문 1편은
  "aesonom@hanyang.ac.kr"로, 이메일 없는 논문 3편은
  "Ae-Son Om @ Hanyang University"로 갈라져 서로 다른 연구실로 집계됨.
- 해결: 행 단위로 "이메일 있으면 이메일, 없으면 이름+소속"을 쓰는 대신,
  먼저 (정규화된 교신저자명, 정규화된 소속)별로 그 사람이 다른 논문에서
  남긴 이메일이 있는지 조회하는 조회표(email_lookup)를 만들고,
  이메일 없는 논문도 그 조회표에 이메일이 있으면 그 이메일 키로 승격시킨다.
  즉 "행 단위 규칙"을 "사람 단위 규칙"으로 바꾼다.

사용법:
    python 03_labs_patched.py
입력:
    data/papers_normalized.csv
출력:
    data/labs.csv
"""

import os
import re
import numpy as np
import pandas as pd

IN = "data/papers_normalized.csv"
OUT = "data/labs.csv"

_INST_NOISE = re.compile(
    r"(hospital|medical center|college of medicine|graduate school of|"
    r"research institute of|department of|inc\.?|co\.? ltd\.?)"
)


def norm_name(name: str) -> str:
    return re.sub(r"[-\s]+", "", str(name).lower())


def norm_inst(inst: str) -> str:
    t = _INST_NOISE.sub("", str(inst).lower())
    return re.sub(r"[^a-z0-9가-힣]", "", t)


def main():
    if not os.path.exists(IN):
        raise SystemExit(f"{IN} 이 없습니다. 02_normalize.py 를 먼저 실행하세요.")

    df = pd.read_csv(IN)
    df["email"] = df["email"].fillna("")
    df["institution"] = df["institution"].fillna("")
    df["last_author"] = df["last_author"].fillna("")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df[(df["institution"] != "") & (df["last_author"] != "")].copy()

    aff_key = df["last_author_aff"].fillna("").str.split("|||").str[0].str.strip()

    # --- 개선: 사람 단위로 이메일을 먼저 조회 ---
    # 정규화된 (이름, 기관 표준명) -> 그 사람이 어느 논문에서든 남긴 이메일
    ident = list(zip(
        df["last_author"].map(norm_name),
        df["institution"].map(norm_inst),
    ))
    df["_ident"] = ident

    email_lookup = {}
    for i, row_email in zip(ident, df["email"]):
        if row_email and i not in email_lookup:
            email_lookup[i] = row_email

    def resolve_key(i, row_email, author, aff):
        if row_email:
            return row_email
        looked_up = email_lookup.get(i)
        if looked_up:
            return looked_up
        return f"{author} @ {aff}"

    df["lab_key"] = [
        resolve_key(i, e, a, af)
        for i, e, a, af in zip(df["_ident"], df["email"], df["last_author"], aff_key)
    ]
    df.drop(columns=["_ident"], inplace=True)
    # --- 개선 끝 ---

    df["protocol_list"] = df["protocol"].fillna("").str.split("|")

    rows = []
    for key, g in df.groupby("lab_key"):
        protos = sorted({p for lst in g["protocol_list"] for p in lst if p})
        years = g["year"].dropna()
        emails = [e for e in g["email"] if e]
        inst_type = g["inst_type"].mode().iat[0]

        rows.append({
            "lab_key": key,
            "pi_name": g["last_author"].mode().iat[0],
            "institution": g["institution"].mode().iat[0],
            "inst_type": inst_type,
            "email": emails[0] if emails else "",
            "n_papers": len(g),
            "n_protocols": len(protos),
            "protocols": "|".join(protos),
            "workflows": " / ".join(sorted({w for w in g["workflow"].fillna("") if w})),
            "latest_year": int(years.max()) if len(years) else 0,
            "earliest_year": int(years.min()) if len(years) else 0,
            "n_since_2025": int((years >= 2025).sum()),
            "avg_authors": round(g["n_authors"].mean(), 1),
            "sample_pmids": ",".join(g["pmid"].astype(str).head(3)),
            "sample_title": g.sort_values("year", ascending=False)["title"].iat[0],
        })

    labs = pd.DataFrame(rows)
    labs = labs.sort_values("lab_key").reset_index(drop=True)
    labs.insert(0, "lab_id", labs.index + 1)
    labs.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"연구실 총계: {len(labs)}개 (논문 {len(df)}건)")
    print(f"이메일 확보: {(labs['email'] != '').sum()}개 ({(labs['email'] != '').mean() * 100:.1f}%)")
    print(f"복수 프로토콜: {(labs['n_protocols'] >= 2).sum()}개")
    print()
    print("[기관 유형별]")
    print(labs["inst_type"].value_counts().to_string())
    print()
    print("[논문 수 분포]")
    print(labs["n_papers"].describe().round(1).to_string())
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
