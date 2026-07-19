"""
논문 기반 액체 핸들링 자동화 타겟 리드 발굴 파이프라인
4단계: LLM 판정 — 실제 수작업 강도

왜 필요한가.
  키워드 검색은 '그 논문이 어떤 실험을 언급했는가'까지만 알려준다.
  그러나 영업에 필요한 정보는 '그 연구실이 직접 피펫을 잡았는가, 얼마나 많이 잡았는가'다.
  실제로 RNA-seq을 수행했다고 쓴 논문의 상당수는 시퀀싱을 외부 업체에 위탁하며,
  이 경우 라이브러리 준비라는 액체 핸들링 작업은 그 연구실에서 일어나지 않는다.
  규칙으로는 이 구분이 불가능하다. 초록을 읽어야 하고, 5,796개를 사람이 읽을 수는 없다.

판정 항목
  in_house    : 액체 핸들링을 직접 수행했는가 (외주 여부)  → 5단계의 관문
  manual_load : 수작업 반복 강도 0~5                      → 5단계의 정렬 기준
  throughput  : 처리 규모                                → 세그먼트 참고
  pain_point  : 자동화로 대체 가능한 구체적 수작업          → 영업 대화의 출발점

사용법:
    pip install requests pandas
    # 둘 중 하나만 설정하면 된다
    $env:GEMINI_API_KEY="..."      또는      $env:ANTHROPIC_API_KEY="..."
    python 04_classify.py
입력:
    data/labs.csv, data/papers_normalized.csv
출력:
    data/labs_judged.csv  (중간 저장: data/cache_llm/)
"""

import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd

IN_LABS = "data/labs.csv"
IN_PAPERS = "data/papers_normalized.csv"
OUT = "data/labs_judged.csv"
CACHE_DIR = "data/cache_llm"
os.makedirs(CACHE_DIR, exist_ok=True)

BATCH = 5           # 한 번의 호출에 담을 연구실 수
RPM = 14            # 분당 요청 수. Gemini 무료 티어가 약 15 RPM 이므로 여유를 둔다
MAX_ABS = 2         # 연구실당 참고할 초록 수
ABS_CHARS = 900     # 초록당 최대 길이
MAX_RETRY = 4

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

if ANTHROPIC_KEY:
    PROVIDER = "anthropic"
elif GEMINI_KEY:
    PROVIDER = "gemini"
else:
    raise SystemExit(
        "ANTHROPIC_API_KEY 또는 GEMINI_API_KEY 중 하나를 환경변수로 설정하세요.\n"
        '  PowerShell:  $env:GEMINI_API_KEY="..."'
    )

SYSTEM = """당신은 실험실 자동화 장비(액체 핸들링 로봇) 영업을 지원하는 분석가다.
논문 초록을 읽고, 그 연구실이 액체 핸들링 수작업을 얼마나 하는지 판정한다.

판정 기준:
- in_house: 액체 핸들링(분주, 희석, 플레이트 세팅, 시약 조제)을 연구실이 직접 수행했으면 true.
  시퀀싱·질량분석·합성 등을 외부 업체나 코어 시설에 위탁한 정황이 뚜렷하면 false.
  공개 데이터 재분석, 임상 관찰연구, 계산연구, 리뷰 논문도 false.
  판단 근거가 부족하면 true로 둔다(보수적으로 기회를 남긴다).
- manual_load: 수작업 반복 강도 0~5.
  0=액체 핸들링 없음   1=소량 단발 실험   2=일반적 분자생물학 실험 수준
  3=플레이트 단위 반복(96웰 ELISA/qPCR 등)이 명확
  4=다수 플레이트·다수 조건·연속 희석 반복
  5=대규모 스크리닝, 수백~수천 샘플 처리
- throughput: low / mid / high
- pain_point: 자동화로 대체 가능한 구체적 수작업을 한국어 25자 이내로. 없으면 빈 문자열.
- confidence: 0~1

반드시 JSON 배열만 출력한다. 설명, 서론, 마크다운 코드펜스를 붙이지 않는다.
입력에 준 id 를 그대로 포함하고, 입력 개수와 출력 개수를 일치시킨다."""

SCHEMA_HINT = """출력 형식:
[{"id":"123","in_house":true,"manual_load":3,"throughput":"mid","pain_point":"96웰 ELISA 반복 분주","confidence":0.8}]"""


# ---------------------------------------------------------------------------
def call_anthropic(prompt):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={"model": "claude-haiku-4-5-20251001", "max_tokens": 1500,
              "system": SYSTEM, "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"])


_GEMINI_MODEL = None


def pick_gemini_model():
    """사용 가능한 모델을 실행 시점에 조회해 고른다.
    모델명을 코드에 박아두면 구글이 이름을 바꾸거나 무료 티어에서 내릴 때 그대로 깨진다.
    (실제로 Pro 계열은 2026년 4월 무료 티어에서 제외됐다)"""
    global _GEMINI_MODEL
    if _GEMINI_MODEL:
        return _GEMINI_MODEL
    r = requests.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        params={"key": GEMINI_KEY}, timeout=30,
    )
    r.raise_for_status()
    names = [
        m["name"].split("/")[-1] for m in r.json().get("models", [])
        if "generateContent" in m.get("supportedGenerationMethods", [])
    ]
    # 무료 티어에서 쓸 수 있는 것은 flash 계열이다.
    # 분류 작업이므로 가벼운 flash-lite 를 우선한다.
    for want in ("flash-lite", "flash"):
        cand = [n for n in names if want in n and "preview" not in n and "exp" not in n]
        if cand:
            _GEMINI_MODEL = sorted(cand)[-1]
            print(f"  선택된 모델: {_GEMINI_MODEL}")
            return _GEMINI_MODEL
    raise SystemExit(f"쓸 수 있는 flash 모델이 없습니다. 조회된 목록: {names[:10]}")


def call_gemini(prompt):
    model = os.environ.get("GEMINI_MODEL") or pick_gemini_model()
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": GEMINI_KEY},
        json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
              "contents": [{"parts": [{"text": prompt}]}],
              "generationConfig": {"temperature": 0, "maxOutputTokens": 1500}},
        timeout=120,
    )
    r.raise_for_status()
    js = r.json()
    return "".join(p.get("text", "") for p in js["candidates"][0]["content"]["parts"])


_last_call = [0.0]


def call_llm(prompt):
    # 무료 티어의 분당 한도를 넘기면 429가 돌아온다.
    # 재시도로 처리할 수도 있지만, 애초에 간격을 두는 편이 전체 시간이 짧다.
    gap = 60.0 / RPM
    wait = gap - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()

    fn = call_anthropic if PROVIDER == "anthropic" else call_gemini
    last = None
    for attempt in range(MAX_RETRY):
        try:
            return fn(prompt)
        except Exception as e:
            last = e
            wait = 2 ** attempt
            print(f"    ...재시도 {attempt + 1}/{MAX_RETRY} ({str(e)[:70]}) {wait}초")
            time.sleep(wait)
    raise RuntimeError(f"LLM 호출 실패: {last}")


def parse_json(text):
    """모델이 코드펜스나 잡담을 붙이는 경우가 있어 배열 부분만 추출한다."""
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    i, j = t.find("["), t.rfind("]")
    if i == -1 or j == -1:
        raise ValueError(f"JSON 배열 없음: {t[:120]}")
    return json.loads(t[i:j + 1])


def build_prompt(batch):
    blocks = [
        f"### id: {it['id']}\n"
        f"기관: {it['institution']} ({it['inst_type']})\n"
        f"검색에 걸린 프로토콜: {it['protocols']}\n"
        f"논문 수: {it['n_papers']}\n"
        f"초록:\n{it['abstracts']}"
        for it in batch
    ]
    return f"{SCHEMA_HINT}\n\n아래 {len(batch)}개 연구실을 각각 판정하라.\n\n" + "\n\n".join(blocks)


# ---------------------------------------------------------------------------
def main():
    labs = pd.read_csv(IN_LABS)
    papers = pd.read_csv(IN_PAPERS)

    papers["email"] = papers["email"].fillna("")
    papers["institution"] = papers["institution"].fillna("")
    papers["last_author"] = papers["last_author"].fillna("")
    papers["year"] = pd.to_numeric(papers["year"], errors="coerce")
    papers["lab_key"] = np.where(
        papers["email"] != "",
        papers["email"],
        papers["last_author"] + " @ " + papers["institution"],
    )
    papers = papers.sort_values("year", ascending=False)

    # 연구실별 대표 초록 (최신 논문 우선)
    abs_map = {}
    for key, g in papers.groupby("lab_key"):
        parts = []
        for _, row in g.head(MAX_ABS).iterrows():
            y = int(row["year"]) if pd.notna(row["year"]) else "?"
            a = str(row.get("abstract") or "")[:ABS_CHARS]
            parts.append(f"- ({y}) {row['title']}\n  {a}")
        abs_map[key] = "\n".join(parts)

    items = [{
        "id": str(r["lab_id"]),
        "lab_key": r["lab_key"],
        "institution": r["institution"],
        "inst_type": r["inst_type"],
        "protocols": r["protocols"],
        "n_papers": int(r["n_papers"]),
        "abstracts": abs_map.get(r["lab_key"], "(초록 없음)"),
    } for _, r in labs.iterrows()]

    n_batches = (len(items) + BATCH - 1) // BATCH
    print(f"판정 대상: {len(items)}개 연구실 / 배치 {BATCH}개씩 → {n_batches}회 호출")
    print(f"사용 API: {PROVIDER}")
    if PROVIDER == "gemini":
        print(f"  분당 {RPM}회 제한 → 예상 소요 약 {n_batches / RPM:.0f}분")
    print()

    results = {}
    consecutive_fail = 0
    for bi in range(n_batches):
        cache = os.path.join(CACHE_DIR, f"b{bi:05d}.json")
        if os.path.exists(cache):
            for row in json.load(open(cache, encoding="utf-8")):
                results[str(row.get("id"))] = row
            continue

        batch = items[bi * BATCH:(bi + 1) * BATCH]
        try:
            out = parse_json(call_llm(build_prompt(batch)))
        except Exception as e:
            consecutive_fail += 1
            print(f"  [배치 {bi}] 실패, 건너뜀: {str(e)[:80]}")
            # 연속으로 계속 실패하면 대개 일일 한도 소진이다.
            # 남은 배치를 헛되이 두드리지 말고 멈춰서, 지금까지의 결과를 저장한다.
            if consecutive_fail >= 5:
                print(f"\n  연속 {consecutive_fail}회 실패 → 중단하고 여기까지 저장합니다.")
                print("  (일일 한도라면 내일 다시 실행하면 캐시 덕에 이어서 진행됩니다)")
                break
            continue

        consecutive_fail = 0
        json.dump(out, open(cache, "w", encoding="utf-8"), ensure_ascii=False)
        for row in out:
            results[str(row.get("id"))] = row

        if bi % 20 == 0:
            print(f"  {bi}/{n_batches} 배치 ({len(results)}개 판정)")

    # 루프를 빠져나온 뒤, 이미 저장돼 있으나 아직 안 읽은 캐시까지 모두 취합한다
    for fn_cache in os.listdir(CACHE_DIR):
        if fn_cache.endswith(".json"):
            for row in json.load(open(os.path.join(CACHE_DIR, fn_cache), encoding="utf-8")):
                results[str(row.get("id"))] = row

    # -----------------------------------------------------------------------
    key = labs["lab_id"].astype(str)
    labs["in_house"] = key.map(lambda i: results.get(i, {}).get("in_house", None))
    labs["manual_load"] = key.map(lambda i: results.get(i, {}).get("manual_load", None))
    labs["throughput"] = key.map(lambda i: results.get(i, {}).get("throughput", ""))
    labs["pain_point"] = key.map(lambda i: results.get(i, {}).get("pain_point", ""))
    labs["llm_confidence"] = key.map(lambda i: results.get(i, {}).get("confidence", None))
    labs["manual_load"] = pd.to_numeric(labs["manual_load"], errors="coerce")

    labs.to_csv(OUT, index=False, encoding="utf-8-sig")

    print(f"\n판정 성공: {labs['manual_load'].notna().sum()}/{len(labs)}")
    print("\n[수작업 강도 분포]")
    print(labs["manual_load"].value_counts().sort_index().to_string())
    print("\n[직접 수행 여부]")
    print(labs["in_house"].value_counts(dropna=False).to_string())
    print("\n[처리량]")
    print(labs["throughput"].value_counts().to_string())
    print(f"\n저장: {OUT}")
    print(f"캐시: {CACHE_DIR}/ (다시 판정하려면 이 폴더를 지우세요)")


if __name__ == "__main__":
    main()