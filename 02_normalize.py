"""
논문 기반 액체 핸들링 자동화 타겟 리드 발굴 파이프라인
2단계: 기관 식별 및 연락처 추출

PubMed 소속 문자열에서 기관을 식별하고 유형(대학/대학병원/정부출연연/기업)을 분류한다.
리드의 단위는 논문이 아니라 연구실이므로, 소속에서 기관을 뽑아 유형을 나눠야
이후 세그먼트 분석이 성립한다.

기관을 어떻게 식별하는가.
  소속 문자열은 일정한 구조를 가진다.
      "Department of X, OO University, City, Republic of Korea"
  즉 '부서, 기관, 도시, 국가' 순으로 좁은 소속에서 넓은 소속으로 나열되고,
  기관명은 University/Hospital/Institute 같은 핵심어로 끝나는 구(phrase)다.
  이 구조를 이용해 규칙으로 기관을 추출한다. 수작업 사전을 쓰지 않는다.

핵심 판단은 '어느 핵심어를 우선하는가'다.
  한 소속에 병원명과 대학명이 함께 나오는 경우가 39%에 이른다.
      "Samsung Medical Center, Sungkyunkwan University School of Medicine"
  여기서 실제 소속은 더 구체적인 병원(삼성서울병원)이지 대학이 아니다.
  그래서 병원을 대학보다 먼저 판정한다.
  또한 KAIST처럼 명칭에 Institute가 들어가지만 대학인 기관은 예외로 먼저 처리한다.

이 규칙만으로 8,960건 중 95%의 기관 유형을 식별한다.
표준 한글명으로 통일하지는 않는다. 연구실 집계 키는 이메일 또는 (교신저자명+소속)이라
기관명 표기가 영문이어도 집계에 영향이 없고, 유형 분류만 정확하면 충분하기 때문이다.

사용법:
    python 02_normalize.py
입력:
    data/raw_papers.csv
출력:
    data/papers_normalized.csv
"""

import os
import re
import pandas as pd

IN = "data/raw_papers.csv"
OUT = "data/papers_normalized.csv"

# ---------------------------------------------------------------------------
# 기관 식별 규칙
# ---------------------------------------------------------------------------
# 명칭에 Institute가 들어가지만 대학인 기관. University 규칙보다 먼저 걸러야 한다.
# 정식명과 약칭(KAIST 등)을 모두 인식한다. 소속란에 약칭만 적히는 경우가 많다.
SPECIAL_UNIV = {
    r"\bKAIST\b|Korea Advanced Institute of Science and Technology": "KAIST",
    r"\bPOSTECH\b|Pohang University of Science(?:\s+and\s+Technology)?": "POSTECH",
    r"\bUNIST\b|Ulsan National Institute of Science(?:\s+and\s+Technology)?": "UNIST",
    r"\bGIST\b|Gwangju Institute of Science(?:\s+and\s+Technology)?": "GIST",
    r"\bDGIST\b|Daegu Gyeongbuk Institute of Science(?:\s+and\s+Technology)?": "DGIST",
    r"University of Science and Technology,?\s*Daejeon": "UST",
}

# 기관명 추출 패턴. 핵심어로 끝나는 구를 잡는다.
# Institute/University는 뒤따르는 'of X (and Y)'까지 포함해야 기관명이 잘리지 않는다.
# (예: 'Korea Institute of Toxicology'가 'Korea Institute'로 잘리면 서로 다른
#  출연연들이 한 덩어리로 뭉친다.)
PAT_HOSPITAL = re.compile(
    r"([A-Z][A-Za-z.'\-]*(?:\s+[A-Za-z.'\-]+){0,5}?\s+(?:Hospital|Medical Center))"
)
PAT_COMPANY = re.compile(
    r"([A-Z][A-Za-z.'\-]*(?:\s+[A-Za-z.'\-]+){0,4}?\s+(?:Co\.?,?\s*Ltd|Inc\.?|Corporation))"
)
PAT_GOV = re.compile(
    r"((?:Korea|National|Korean)\s+(?:[A-Za-z]+\s+){0,3}?Institute"
    r"(?:\s+of\s+[A-Za-z]+(?:\s+(?:and|&|of)\s+[A-Za-z]+){0,3})?)",
    re.I,
)
PAT_UNIV = re.compile(
    r"([A-Z][A-Za-z.'\-]*(?:\s+[A-Za-z.'\-]+){0,4}?\s+University(?:\s+of\s+[A-Za-z]+)?)"
)
PAT_INST = re.compile(
    r"([A-Z][A-Za-z.'\-]*(?:\s+[A-Za-z.'\-]+){0,4}?\s+Institute"
    r"(?:\s+of\s+[A-Za-z]+(?:\s+and\s+[A-Za-z]+)?)?)"
)

EMAIL_PAT = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")


def extract_email(aff):
    m = EMAIL_PAT.search(aff or "")
    if not m:
        return ""
    return m.group(0).rstrip(". ").lower()


def resolve_institution(aff):
    """소속 문자열에서 기관명과 유형을 규칙으로 판정한다.

    우선순위:
      1. 특수 약칭 대학 (KAIST 등) - Institute를 포함하지만 대학
      2. 병원 - 대학과 병기될 때 더 구체적인 실제 소속
      3. 기업
      4. 정부출연연 (독립 연구기관)
      5. 대학 - University가 있으면 부설 institute/center는 무시
      6. 그 밖의 Institute - 연구소로 분류
    """
    # 소속란은 여러 기관을 ||| 로 이어 붙인 경우가 있다. 교신저자의 첫 소속만 본다.
    first = str(aff or "").split("|||")[0]

    # 1. 특수 약칭 대학
    for pat, name in SPECIAL_UNIV.items():
        if re.search(pat, first, re.I):
            return name, "대학", "규칙"

    # 2. 병원
    m = PAT_HOSPITAL.search(first)
    if m:
        return m.group(1).strip(), "대학병원", "규칙"

    # 3. 기업
    m = PAT_COMPANY.search(first)
    if m:
        return m.group(1).strip(), "기업", "규칙"

    # 4. 정부출연연 (단, 'University ... Institute' 형태는 대학이므로 제외)
    m = PAT_GOV.search(first)
    if m and "University" not in m.group(1):
        return m.group(1).strip(), "정부출연연", "규칙"

    # 5. 대학 (University가 있으면 부설 institute/center 무시하고 대학으로)
    m = PAT_UNIV.search(first)
    if m:
        return m.group(1).strip(), "대학", "규칙"

    # 6. 그 밖의 독립 연구소
    m = PAT_INST.search(first)
    if m:
        return m.group(1).strip(), "정부출연연", "규칙"

    return "", "미분류", "실패"


def main():
    if not os.path.exists(IN):
        raise SystemExit(f"{IN} 이 없습니다. 01_collect.py 를 먼저 실행하세요.")

    df = pd.read_csv(IN)
    print(f"입력: {len(df)}건")

    df = df[df["last_author_is_korea"] == True].copy()
    print(f"교신저자 한국 소속: {len(df)}건")

    df["last_author_aff"] = df["last_author_aff"].fillna("")
    df["email"] = df["last_author_aff"].apply(extract_email)

    resolved = df["last_author_aff"].apply(resolve_institution)
    df["institution"] = [r[0] for r in resolved]
    df["inst_type"] = [r[1] for r in resolved]
    df["resolve_method"] = [r[2] for r in resolved]

    df["email_domain"] = df["email"].apply(lambda e: e.split("@")[-1] if e else "")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    ok = (df["institution"] != "").sum()
    print(f"\n기관 식별 성공: {ok}/{len(df)} = {ok / len(df) * 100:.1f}%")
    print(f"교신저자 이메일 확보: {(df['email'] != '').sum()} ({(df['email'] != '').mean() * 100:.1f}%)")

    print("\n[기관 유형 분포]")
    print(df["inst_type"].value_counts().to_string())

    print("\n[기관 상위 25]")
    print(df[df["institution"] != ""]["institution"].value_counts().head(25).to_string())

    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()