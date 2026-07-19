"""
논문 기반 액체 핸들링 자동화 타겟 리드 발굴 파이프라인
1단계: PubMed 수집 (v2)

v1 대비 변경점
  - 저자별 소속을 분리 저장 (관례상 마지막 저자 = 연구책임자, 그 소속이 곧 연구실 주소)
  - 한국 소재 소속만 별도 추출 (외국 논문에 한국인 공저자 1명 끼인 건을 걸러내기 위함)
  - NGS / 연속희석 검색어 확장 (v1에서 각각 7건, 0건에 그침)
  - 프로토콜당 수집 상한 500 → 2000 (v1에서 4개 프로토콜이 상한에 걸려 잘림)

사용법:
    pip install requests pandas
    python 01_collect.py
출력:
    data/raw_papers.csv
"""

import os
import re
import time
import requests
import pandas as pd
from xml.etree import ElementTree as ET

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EMAIL = "gugu52@naver.com"
TOOL = "able-labs-lead-discovery"
API_KEY = os.environ.get("NCBI_API_KEY")  # 없어도 동작 (초당 3회 → 있으면 10회)

OUT_DIR = "data"
CACHE_DIR = os.path.join(OUT_DIR, "cache")   # 프로토콜별 중간 저장. 중단돼도 이어서 실행된다
os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 타겟 프로토콜 정의
# ---------------------------------------------------------------------------
# 선정 기준: 에이블랩스 제품군(NOTABLE / NOTABLE96 / SUITABLE)이 자동화하는
# 액체 핸들링 워크플로우. 각 프로토콜은 반복적 분주·희석·플레이트 처리를 수반한다.

PROTOCOLS = {
    "ELISA": {
        "query": '("enzyme-linked immunosorbent assay"[Title/Abstract] OR ELISA[Title/Abstract])',
        "product_hint": "NOTABLE96",   # 96웰 플레이트 기반, 고반복
        "workflow": "면역분석 전처리 (희석·분주·세척)",
    },
    "NGS_library_prep": {
        # v1에서 7건에 그침. 라이브러리 준비를 명시하는 논문은 드물지만,
        # 시퀀싱을 수행했다면 라이브러리 준비는 반드시 수반된다. 검색 범위를 확장.
        "query": (
            '("library preparation"[Title/Abstract] OR "sequencing library"[Title/Abstract] '
            'OR "next-generation sequencing"[Title/Abstract] OR NGS[Title/Abstract] '
            'OR "RNA-seq"[Title/Abstract] OR "RNA sequencing"[Title/Abstract] '
            'OR "whole genome sequencing"[Title/Abstract] OR "whole-exome sequencing"[Title/Abstract] '
            'OR "single-cell RNA"[Title/Abstract] OR "amplicon sequencing"[Title/Abstract] '
            'OR "16S rRNA sequencing"[Title/Abstract])'
        ),
        "product_hint": "SUITABLE",    # 다양한 용량, 정밀 분주
        "workflow": "NGS 라이브러리 준비 (시약 분주·정제·정량)",
    },
    "qPCR": {
        "query": '("quantitative PCR"[Title/Abstract] OR qPCR[Title/Abstract] OR "real-time PCR"[Title/Abstract] OR "RT-qPCR"[Title/Abstract])',
        "product_hint": "NOTABLE96",
        "workflow": "PCR 반응액 조제 및 플레이트 세팅",
    },
    "cell_line_development": {
        "query": '("cell line development"[Title/Abstract] OR "cell culture"[Title/Abstract] OR "cell seeding"[Title/Abstract] OR "cell viability assay"[Title/Abstract])',
        "product_hint": "NOTABLE",
        "workflow": "세포 배양 및 계대 (배지 교환·시딩)",
    },
    "drug_screening": {
        "query": '("high-throughput screening"[Title/Abstract] OR "drug screening"[Title/Abstract] OR "compound screening"[Title/Abstract] OR "dose-response"[Title/Abstract])',
        "product_hint": "NOTABLE96",
        "workflow": "약물 스크리닝 (화합물 희석·분주)",
    },
    "protein_purification": {
        "query": '("protein purification"[Title/Abstract] OR "affinity chromatography"[Title/Abstract] OR "protein expression and purification"[Title/Abstract])',
        "product_hint": "SUITABLE",
        "workflow": "단백질 정제 공정 (삼성바이오로직스 도입 레퍼런스 영역)",
    },
    "serial_dilution": {
        # v1의 preservative_efficacy는 0건. 방부력 시험은 산업 시험이라 논문화되지 않는다.
        # 실제로 연속 희석을 대량 반복하는 것은 MIC 측정과 항균 감수성 시험이며,
        # 이쪽이 NOTABLE의 타겟 워크플로우에 정확히 대응한다.
        "query": (
            '("minimum inhibitory concentration"[Title/Abstract] OR "broth microdilution"[Title/Abstract] '
            'OR "antimicrobial susceptibility"[Title/Abstract] OR "serial dilution"[Title/Abstract] '
            'OR "preservative efficacy"[Title/Abstract] OR "antimicrobial efficacy"[Title/Abstract])'
        ),
        "product_hint": "NOTABLE",
        "workflow": "연속 희석 기반 시험 (MIC 측정·방부력 시험)",
    },
}

KOREA_FILTER = '("Korea"[Affiliation] OR "Korea, Republic of"[Affiliation])'
DATE_FILTER = '("2022/01/01"[Date - Publication] : "3000"[Date - Publication])'

RETMAX = 6000   # NGS는 전체 5,247건 -> 상한에 걸리지 않도록 상향
CHUNK = 150     # efetch 1회당 요청 건수. 크면 NCBI가 502를 반환하는 빈도가 올라간다
MAX_RETRY = 5


# ---------------------------------------------------------------------------
# NCBI E-utilities 호출
# ---------------------------------------------------------------------------
def _params(extra):
    p = {"tool": TOOL, "email": EMAIL}
    if API_KEY:
        p["api_key"] = API_KEY
    p.update(extra)
    return p


def _sleep():
    # NCBI 정책: API 키 없으면 초당 3회, 있으면 10회
    time.sleep(0.12 if API_KEY else 0.36)


def _request(method, url, **kw):
    """NCBI는 부하가 걸리면 502/429를 간헐적으로 반환한다.
    수십 분짜리 수집이 일시적 오류 한 번에 무너지지 않도록 지수 백오프로 재시도한다."""
    last = None
    for attempt in range(MAX_RETRY):
        try:
            r = requests.request(method, url, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} from NCBI")
            r.raise_for_status()
            return r
        except Exception as e:
            last = e
            wait = 2 ** attempt
            print(f"    ...재시도 {attempt + 1}/{MAX_RETRY} ({e}) {wait}초 대기")
            time.sleep(wait)
    raise RuntimeError(f"NCBI 요청 실패: {last}")


def esearch(term, retmax=RETMAX):
    """검색어에 해당하는 PMID 목록과 전체 검색 건수를 가져온다."""
    r = _request(
        "GET", f"{BASE}/esearch.fcgi",
        params=_params({
            "db": "pubmed", "term": term, "retmax": retmax,
            "retmode": "json", "sort": "date",
        }),
        timeout=30,
    )
    _sleep()
    js = r.json()["esearchresult"]
    return js.get("idlist", []), int(js.get("count", 0))


def efetch(pmids):
    """PMID 목록의 상세 정보(초록·저자·소속)를 XML로 가져온다."""
    if not pmids:
        return None
    r = _request(
        "POST", f"{BASE}/efetch.fcgi",
        data=_params({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}),
        timeout=120,
    )
    _sleep()
    return ET.fromstring(r.content)


# ---------------------------------------------------------------------------
# 파싱
# ---------------------------------------------------------------------------
KOREA_PAT = re.compile(r"\b(korea|korean)\b", re.I)


def _text(node):
    """중첩 태그(<i>, <sup> 등)를 포함한 전체 텍스트를 뽑는다."""
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def parse_article(art):
    pmid = _text(art.find(".//PMID"))
    title = _text(art.find(".//ArticleTitle"))

    # 초록: Methods/Results 등 라벨이 붙은 경우가 있어 라벨을 보존한다
    abs_parts = []
    for ab in art.findall(".//Abstract/AbstractText"):
        label = ab.get("Label")
        txt = _text(ab)
        abs_parts.append(f"[{label}] {txt}" if label else txt)
    abstract = "\n".join(abs_parts)

    journal = _text(art.find(".//Journal/Title"))

    year = ""
    for path in (".//JournalIssue/PubDate/Year", ".//JournalIssue/PubDate/MedlineDate"):
        v = _text(art.find(path))
        if v:
            m = re.search(r"\d{4}", v)
            year = m.group(0) if m else v
            break

    # 저자별 소속을 분리해 보관한다.
    # PubMed는 교신저자를 명시하지 않지만, 생명과학 논문의 관례상
    # 마지막 저자가 연구책임자(PI)이며 그 소속이 곧 연구실의 위치다.
    # 리드의 단위는 논문이 아니라 연구실이므로 이 구분이 필요하다.
    people = []
    for a in art.findall(".//AuthorList/Author"):
        last = _text(a.find("LastName"))
        fore = _text(a.find("ForeName"))
        name = f"{fore} {last}".strip()
        if not name:
            continue
        affs = [_text(x) for x in a.findall(".//AffiliationInfo/Affiliation")]
        people.append({"name": name, "affs": [x for x in affs if x]})

    all_affs = []
    for p in people:
        all_affs.extend(p["affs"])
    all_affs = list(dict.fromkeys(all_affs))

    first = people[0] if people else {"name": "", "affs": []}
    lastp = people[-1] if len(people) > 1 else {"name": "", "affs": []}

    # 한국 소재 소속만 추출.
    # 외국 연구실이 주도한 논문에 한국인 공저자가 1명 끼인 경우와
    # 한국 연구실이 주도한 논문을 구분하기 위한 근거가 된다.
    korea_affs = [a for a in all_affs if KOREA_PAT.search(a)]

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "year": year,
        "first_author": first["name"],
        "first_author_aff": " ||| ".join(first["affs"]),
        "first_author_is_korea": any(KOREA_PAT.search(a) for a in first["affs"]),
        "last_author": lastp["name"],
        "last_author_aff": " ||| ".join(lastp["affs"]),
        "last_author_is_korea": any(KOREA_PAT.search(a) for a in lastp["affs"]),
        "n_authors": len(people),
        "korea_affiliations": " ||| ".join(korea_affs),
        "n_korea_affs": len(korea_affs),
        "affiliations": " ||| ".join(all_affs),
    }


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def collect_protocol(name, spec):
    """프로토콜 하나를 수집한다. 결과는 캐시에 저장해 재실행 시 건너뛴다."""
    cache = os.path.join(CACHE_DIR, f"{name}.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        print(f"[cache] {name} ... {len(df)}건 (이미 수집됨, 건너뜀)")
        return df

    term = f"{spec['query']} AND {KOREA_FILTER} AND {DATE_FILTER}"
    print(f"[esearch] {name} ...", end=" ", flush=True)
    pmids, total = esearch(term)
    flag = "  <- 상한 도달, 최신순으로 잘림" if total > len(pmids) else ""
    print(f"{len(pmids)}건 수집 / 전체 {total}건{flag}")

    rows = []
    for i in range(0, len(pmids), CHUNK):
        chunk = pmids[i:i + CHUNK]
        root = efetch(chunk)
        if root is None:
            continue
        for art in root.findall(".//PubmedArticle"):
            d = parse_article(art)
            d["protocol"] = name
            d["workflow"] = spec["workflow"]
            d["product_hint"] = spec["product_hint"]
            rows.append(d)
        print(f"  [efetch] {name} {min(i + CHUNK, len(pmids))}/{len(pmids)}")

    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False, encoding="utf-8-sig")
    return df


def main():
    parts = []
    for name, spec in PROTOCOLS.items():
        parts.append(collect_protocol(name, spec))

    df = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    if df.empty:
        print("\n수집된 논문이 없습니다. 검색어 또는 기간 필터를 확인하세요.")
        return

    # 같은 논문이 여러 프로토콜에 걸리는 것은 노이즈가 아니라 신호다.
    # 여러 워크플로우를 함께 돌리는 연구실일수록 수작업 부담이 크므로 합쳐 보존한다.
    meta_cols = ["protocol", "workflow", "product_hint"]
    agg_spec = {c: "first" for c in df.columns if c not in meta_cols + ["pmid"]}
    agg_spec.update({
        "protocol": lambda s: "|".join(sorted(set(s))),
        "workflow": lambda s: " / ".join(sorted(set(s))),
        "product_hint": lambda s: "|".join(sorted(set(s))),
    })
    agg = df.groupby("pmid").agg(agg_spec).reset_index()
    agg["n_protocols"] = agg["protocol"].str.count(r"\|") + 1

    out = os.path.join(OUT_DIR, "raw_papers.csv")
    agg.to_csv(out, index=False, encoding="utf-8-sig")

    print(f"\n수집 완료: 논문 {len(agg)}건 (중복 제거 전 {len(df)}건) -> {out}")
    print(f"교신저자(마지막 저자)가 한국 소속인 논문: {agg['last_author_is_korea'].sum()}건")
    print(f"복수 프로토콜 논문: {(agg['n_protocols'] > 1).sum()}건")
    print("\n[프로토콜 분포]")
    print(agg["protocol"].value_counts().head(20))
    print(f"\n캐시: {CACHE_DIR}/ (다시 수집하려면 이 폴더를 지우세요)")


if __name__ == "__main__":
    main()