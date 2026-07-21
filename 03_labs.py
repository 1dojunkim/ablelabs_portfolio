"""
논문 기반 액체 핸들링 자동화 타겟 리드 발굴 파이프라인
3단계: 연구실 단위 집계

리드의 단위는 논문이 아니라 연구실이다.
장비를 검토하고 예산을 집행하는 주체는 연구책임자(교신저자)이므로,
논문을 연구실 단위로 접어야 영업 대상 목록이 된다.

이 단계는 집계만 한다. 점수도 제품 결정도 하지 않는다.
- 점수: 여러 지표를 하나로 합치려면 임의 가중치가 필요하고, 그 근거가 없다.
- 제품 결정: 처리량(4단계 AI 판정 결과)을 반영해야 정확하므로 5단계로 미룬다.
우선순위와 제품은 5단계에서 정한다.

사용법:
    python 03_labs.py
입력:
    data/papers_normalized.csv
출력:
    data/labs.csv
"""

import os
import numpy as np
import pandas as pd

IN = "data/papers_normalized.csv"
OUT = "data/labs.csv"


def main():
    if not os.path.exists(IN):
        raise SystemExit(f"{IN} 이 없습니다. 02_normalize.py 를 먼저 실행하세요.")

    df = pd.read_csv(IN)
    df["email"] = df["email"].fillna("")
    df["institution"] = df["institution"].fillna("")
    df["last_author"] = df["last_author"].fillna("")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # 기관을 식별하지 못한 건은 제외한다. 어디의 누구인지 모르면 영업할 수 없다.
    df = df[(df["institution"] != "") & (df["last_author"] != "")].copy()

    # 연구실 식별키.
    # 이메일이 있으면 그것이 가장 확실한 식별자다.
    # 없으면 (교신저자명 + 소속 원문) 조합을 쓴다.
    #   주의: 여기서 기관 '표준명(institution)'이 아니라 소속 '원문(last_author_aff)'을
    #   쓴다. 표준명은 02의 식별 규칙이 바뀌면 표기가 달라져(예: 한글↔영문) 키가 흔들리고,
    #   그러면 판정 결과와의 연결이 끊긴다. 소속 원문은 논문에 실린 그대로라 불변이므로
    #   02를 어떻게 바꾸든 같은 연구실은 같은 키로 묶인다.
    aff_key = df["last_author_aff"].fillna("").str.split("|||").str[0].str.strip()
    df["lab_key"] = np.where(
        df["email"] != "",
        df["email"],
        df["last_author"] + " @ " + aff_key,
    )

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
    # 정렬 없이 안정적인 id 만 부여한다. 순위·제품은 5단계에서 만든다.
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