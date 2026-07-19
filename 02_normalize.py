"""
논문 기반 액체 핸들링 자동화 타겟 리드 발굴 파이프라인
2단계: 기관 정규화 및 연락처 추출

PubMed 소속 문자열은 표기가 제각각이다.
같은 서울대라도 'Seoul National University', 'Seoul National University College of Medicine',
'Seoul National University Bundang Hospital'로 나뉘고, 앞의 둘과 세 번째는 다른 조직이다.
리드의 단위는 논문이 아니라 연구실이므로, 기관을 하나의 이름으로 묶어야 집계가 성립한다.

처리 내용
  1. 교신저자(마지막 저자) 소속에서 기관을 식별하고 표준명으로 정규화
  2. 기관 유형 분류 (대학 / 대학병원 / 정부출연연 / 기업 / 기타)
  3. 소속 문자열에 노출된 교신저자 이메일 추출

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
# 기관 사전
# ---------------------------------------------------------------------------
# (정규식, 표준명, 유형)
# 순서가 곧 우선순위다. 위에서부터 먼저 매칭된 것을 채택한다.
# 소속 문자열은 대개 '학과, 단과대학, 대학, 도시, 국가' 순으로 좁은 것에서 넓은 것으로 가는데,
# 병원명은 대학명과 함께 등장하므로 병원을 먼저 잡아야 대학으로 잘못 묶이지 않는다.

INSTITUTIONS = [
    # --- 기업 ---------------------------------------------------------------
    (r"Samsung\s+Biologics", "삼성바이오로직스", "기업"),
    (r"Celltrion", "셀트리온", "기업"),
    (r"\bLG\s+Chem", "LG화학", "기업"),
    (r"SK\s+bioscience", "SK바이오사이언스", "기업"),
    (r"Green\s+Cross|GC\s+Biopharma|\bGC\s+Cell", "GC녹십자", "기업"),
    (r"Hanmi\s+Pharm", "한미약품", "기업"),
    (r"Yuhan\s+Corp", "유한양행", "기업"),
    (r"Daewoong", "대웅제약", "기업"),
    (r"Chong\s*Kun\s*Dang", "종근당", "기업"),
    (r"Dong-?A\s+(ST|Pharm)", "동아에스티", "기업"),
    (r"JW\s+(Pharma|Bioscience)", "JW중외제약", "기업"),
    (r"Genesystem", "젠시스템", "기업"),
    (r"Seegene", "씨젠", "기업"),
    (r"Macrogen", "마크로젠", "기업"),
    (r"Theragen", "테라젠이텍스", "기업"),
    (r"\bLegoChem", "리가켐바이오", "기업"),
    (r"Yonsei\s+University\s+Health\s+System", "연세의료원", "대학병원"),

    # --- 병원·의료기관 ------------------------------------------------------
    (r"Asan\s+Medical\s+C", "서울아산병원", "대학병원"),
    (r"Samsung\s+Medical\s+C", "삼성서울병원", "대학병원"),
    (r"Severance\s+Hospital|Yonsei\s+Cancer\s+C", "세브란스병원", "대학병원"),
    (r"Seoul\s+National\s+University\s+Bundang\s+H", "분당서울대병원", "대학병원"),
    (r"Seoul\s+National\s+University\s+Hospital|SNUH", "서울대병원", "대학병원"),
    (r"Seoul\s+St\.?\s*Mary'?s\s+H", "서울성모병원", "대학병원"),
    (r"National\s+Cancer\s+C", "국립암센터", "정부출연연"),
    (r"Korea\s+University\s+(Anam|Guro|Ansan)\s+H", "고려대병원", "대학병원"),
    (r"Ajou\s+University\s+Hospital|Ajou\s+University\s+Medical\s+C", "아주대병원", "대학병원"),
    (r"Gil\s+Medical\s+C", "가천대길병원", "대학병원"),
    (r"Chungnam\s+National\s+University\s+H", "충남대병원", "대학병원"),
    (r"Kyungpook\s+National\s+University\s+(Chilgok\s+)?H", "경북대병원", "대학병원"),
    (r"Pusan\s+National\s+University\s+(Yangsan\s+)?H", "부산대병원", "대학병원"),
    (r"Chonnam\s+National\s+University\s+(Hwasun\s+)?H", "전남대병원", "대학병원"),
    (r"Jeonbuk\s+National\s+University\s+H", "전북대병원", "대학병원"),
    (r"Hallym\s+University\s+.*Sacred|Hallym\s+University\s+Medical", "한림대성심병원", "대학병원"),
    (r"Bundang\s+CHA|CHA\s+Bundang", "분당차병원", "대학병원"),

    # --- 정부출연연·국가기관 ------------------------------------------------
    (r"KRIBB|Korea\s+Research\s+Institute\s+of\s+Bioscience", "한국생명공학연구원(KRIBB)", "정부출연연"),
    (r"KRICT|Korea\s+Research\s+Institute\s+of\s+Chemical", "한국화학연구원(KRICT)", "정부출연연"),
    (r"Korea\s+Institute\s+of\s+Science\s+and\s+Technology\b|\bKIST\b", "한국과학기술연구원(KIST)", "정부출연연"),
    (r"Korea\s+Institute\s+of\s+Toxicology|\bKIT\b", "안전성평가연구소", "정부출연연"),
    (r"Korea\s+Basic\s+Science\s+Institute|\bKBSI\b", "한국기초과학지원연구원", "정부출연연"),
    (r"Korea\s+Food\s+Research", "한국식품연구원", "정부출연연"),
    (r"Korea\s+Disease\s+Control|\bKDCA\b|Center\s+for\s+Disease\s+Control", "질병관리청", "정부출연연"),
    (r"National\s+Institute\s+of\s+Health", "국립보건연구원", "정부출연연"),
    (r"Rural\s+Development\s+Administration|National\s+Institute\s+of\s+(Agricultural|Crop|Horticultural|Animal)", "농촌진흥청", "정부출연연"),
    (r"National\s+Institute\s+of\s+Forest", "국립산림과학원", "정부출연연"),
    (r"National\s+Institute\s+of\s+Fisheries", "국립수산과학원", "정부출연연"),
    (r"Ministry\s+of\s+Food\s+and\s+Drug|\bMFDS\b|National\s+Institute\s+of\s+Food\s+and\s+Drug", "식품의약품안전처", "정부출연연"),

    (r"Korea\s+Institute\s+of\s+Oriental\s+Medicine|\bKIOM\b", "한국한의학연구원", "정부출연연"),
    (r"Animal\s+and\s+Plant\s+Quarantine", "농림축산검역본부", "정부출연연"),
    (r"Korea\s+Polar\s+Research", "극지연구소", "정부출연연"),
    (r"International\s+Vaccine\s+Institute", "국제백신연구소", "정부출연연"),
    (r"Health\s+and\s+Environment\s+Research\s+Institute|Institute\s+of\s+Health\s+and\s+Environment", "보건환경연구원", "정부출연연"),
    (r"Gyeonggido\s+Business\s+and\s+Science|\bGBSA\b", "경기도경제과학진흥원", "정부출연연"),
    (r"Korea\s+Atomic\s+Energy|\bKAERI\b", "한국원자력연구원", "정부출연연"),

    # --- 대학 (특수 명칭 우선) ----------------------------------------------
    (r"Korea\s+Advanced\s+Institute\s+of\s+Science|\bKAIST\b", "KAIST", "대학"),
    (r"Pohang\s+University\s+of\s+Science|\bPOSTECH\b", "POSTECH", "대학"),
    (r"Ulsan\s+National\s+Institute\s+of\s+Science|\bUNIST\b", "UNIST", "대학"),
    (r"Gwangju\s+Institute\s+of\s+Science|\bGIST\b", "GIST", "대학"),
    (r"Daegu\s+Gyeongbuk\s+Institute|\bDGIST\b", "DGIST", "대학"),
    (r"University\s+of\s+Science\s+and\s+Technology,?\s*Daejeon|\bUST\b", "과학기술연합대학원(UST)", "대학"),

    # --- 대학 (일반) --------------------------------------------------------
    (r"Seoul\s+National\s+University", "서울대학교", "대학"),
    (r"Yonsei\s+University", "연세대학교", "대학"),
    (r"Sungkyunkwan\s+University|\bSKKU\b", "성균관대학교", "대학"),
    (r"(The\s+)?Catholic\s+University\s+of\s+Korea", "가톨릭대학교", "대학"),
    (r"Korea\s+University", "고려대학교", "대학"),
    (r"Kyungpook\s+National\s+University", "경북대학교", "대학"),
    (r"Kyung\s*Hee\s+University", "경희대학교", "대학"),
    (r"Pusan\s+National\s+University", "부산대학교", "대학"),
    (r"Chonnam\s+National\s+University", "전남대학교", "대학"),
    (r"Chung-?Ang\s+University", "중앙대학교", "대학"),
    (r"Gyeongsang\s+National\s+University", "경상국립대학교", "대학"),
    (r"University\s+of\s+Ulsan", "울산대학교", "대학"),
    (r"Ajou\s+University", "아주대학교", "대학"),
    (r"Konkuk\s+University", "건국대학교", "대학"),
    (r"Hanyang\s+University", "한양대학교", "대학"),
    (r"Jeonbuk\s+National\s+University|Chonbuk\s+National", "전북대학교", "대학"),
    (r"Chungnam\s+National\s+University", "충남대학교", "대학"),
    (r"Dankook\s+University", "단국대학교", "대학"),
    (r"Kangwon\s+National\s+University", "강원대학교", "대학"),
    (r"Gachon\s+University", "가천대학교", "대학"),
    (r"Chungbuk\s+National\s+University", "충북대학교", "대학"),
    (r"Soonchunhyang\s+University", "순천향대학교", "대학"),
    (r"Hallym\s+University", "한림대학교", "대학"),
    (r"Pukyong\s+National\s+University", "부경대학교", "대학"),
    (r"Ewha\s+Womans\s+University", "이화여자대학교", "대학"),
    (r"Jeju\s+National\s+University", "제주대학교", "대학"),
    (r"Dongguk\s+University", "동국대학교", "대학"),
    (r"Inha\s+University", "인하대학교", "대학"),
    (r"Yeungnam\s+University", "영남대학교", "대학"),
    (r"CHA\s+University", "차의과학대학교", "대학"),
    (r"Sejong\s+University", "세종대학교", "대학"),
    (r"Sookmyung", "숙명여자대학교", "대학"),
    (r"Kookmin\s+University", "국민대학교", "대학"),
    (r"Kangnung|Gangneung-?Wonju", "강릉원주대학교", "대학"),
    (r"Korea\s+Maritime\s+and\s+Ocean\s+University", "한국해양대학교", "대학"),
    (r"Sunchon\s+National\s+University", "순천대학교", "대학"),
    (r"Kongju\s+National\s+University", "공주대학교", "대학"),
    (r"Andong\s+National\s+University", "안동대학교", "대학"),
    (r"Kyonggi\s+University", "경기대학교", "대학"),
    (r"Wonkwang\s+University", "원광대학교", "대학"),
    (r"Keimyung\s+University", "계명대학교", "대학"),
    (r"Inje\s+University", "인제대학교", "대학"),
    (r"Kosin\s+University", "고신대학교", "대학"),
    (r"Daegu\s+Haany|Daegu\s+Catholic", "대구한의대·대구가톨릭대", "대학"),
    (r"Kyungsung\s+University", "경성대학교", "대학"),
    (r"Silla\s+University", "신라대학교", "대학"),
    (r"Hoseo\s+University", "호서대학교", "대학"),
    (r"Hankyong", "한경대학교", "대학"),
    (r"Kwangwoon", "광운대학교", "대학"),
    (r"Myongji\s+University", "명지대학교", "대학"),
    (r"Sangji\s+University", "상지대학교", "대학"),
    (r"Semyung", "세명대학교", "대학"),
    (r"Woosuk\s+University", "우석대학교", "대학"),
    (r"Kunsan\s+National", "군산대학교", "대학"),
    (r"Mokpo\s+National", "목포대학교", "대학"),
    (r"Changwon\s+National", "창원대학교", "대학"),
    (r"Pai\s*Chai|Paichai", "배재대학교", "대학"),
    (r"Hannam\s+University", "한남대학교", "대학"),
    (r"Kyungnam\s+University", "경남대학교", "대학"),
    (r"Dong-?Eui", "동의대학교", "대학"),
    (r"Dong-?A\s+University", "동아대학교", "대학"),
    (r"Ulsan\s+University", "울산대학교", "대학"),
    (r"Seoul\s+Women'?s", "서울여자대학교", "대학"),
    (r"Duksung", "덕성여자대학교", "대학"),
    (r"Sungshin", "성신여자대학교", "대학"),
    (r"Kyungil|Kyungpook\s+Nat'?l", "경일대학교", "대학"),
    (r"Handong", "한동대학교", "대학"),
    (r"Hanbat", "한밭대학교", "대학"),
    (r"Korea\s+Polytechnic|KOREATECH", "한국기술교육대학교", "대학"),
    (r"Seoul\s+National\s+University\s+of\s+Science", "서울과학기술대학교", "대학"),
    (r"Chosun\s+University", "조선대학교", "대학"),
    (r"Incheon\s+National\s+University", "인천대학교", "대학"),
    (r"Sogang\s+University", "서강대학교", "대학"),
    (r"Soongsil\s+University", "숭실대학교", "대학"),
    (r"Hongik\s+University", "홍익대학교", "대학"),
    (r"Eulji\s+University", "을지대학교", "대학"),
    (r"Konyang\s+University", "건양대학교", "대학"),
    (r"Sahmyook\s+University", "삼육대학교", "대학"),
    (r"Sangmyung\s+University", "상명대학교", "대학"),
    (r"Dongduk\s+Women'?s", "동덕여자대학교", "대학"),
    (r"Daejeon\s+University", "대전대학교", "대학"),
    (r"Daegu\s+University", "대구대학교", "대학"),
    (r"Kumoh\s+National\s+Institute", "금오공과대학교", "대학"),
    (r"Gachon\s+University", "가천대학교", "대학"),
    (r"Kyungpook\s+National\s+University\s+of\s+Education", "경북대학교", "대학"),
    (r"Korea\s+National\s+University\s+of\s+Education", "한국교원대학교", "대학"),
    (r"Hankuk\s+University\s+of\s+Foreign", "한국외국어대학교", "대학"),
    (r"Seoul\s+National\s+University\s+of\s+Education", "서울교육대학교", "대학"),
]

COMPILED = [(re.compile(p, re.I), name, kind) for p, name, kind in INSTITUTIONS]

# 사전에 없는 기관을 위한 일반 패턴 (표준명은 원문 그대로 사용)
GENERIC = [
    (re.compile(r"([A-Z][A-Za-z&.'\-]*(?:\s+[A-Za-z&.'\-]+){0,4}?\s+(?:Co\.,?\s*Ltd|Inc\.?|Corporation))", re.I), "기업"),
    (re.compile(r"([A-Z][A-Za-z&.'\-]*(?:\s+[A-Za-z&.'\-]+){0,4}?\s+(?:Hospital|Medical Center))"), "대학병원"),
    (re.compile(r"([A-Z][A-Za-z&.'\-]*(?:\s+[A-Za-z&.'\-]+){0,4}?\s+University)"), "대학"),
    (re.compile(r"((?:Korea|National)\s+[A-Za-z&.'\-]*(?:\s+[A-Za-z&.'\-]+){0,4}?\s+Institute)"), "정부출연연"),
]

EMAIL_PAT = re.compile(r"[\w.+\-]+@[\w\-]+\.[\w.\-]+")


def extract_email(aff):
    m = EMAIL_PAT.search(aff or "")
    if not m:
        return ""
    return m.group(0).rstrip(". ").lower()


def resolve_institution(aff):
    """소속 문자열에서 기관 표준명과 유형을 판정한다."""
    s = aff or ""
    for pat, name, kind in COMPILED:
        if pat.search(s):
            return name, kind, "사전"
    for pat, kind in GENERIC:
        m = pat.search(s)
        if m:
            return m.group(1).strip(), kind, "일반패턴"
    return "", "미분류", "실패"


def main():
    if not os.path.exists(IN):
        raise SystemExit(f"{IN} 이 없습니다. 01_collect.py 를 먼저 실행하세요.")

    df = pd.read_csv(IN)
    print(f"입력: {len(df)}건")

    # 교신저자가 한국 소속인 논문만 남긴다.
    # 외국 연구실 논문에 한국인 공저자가 한 명 참여한 경우는 영업 대상이 아니다.
    # 실험을 설계하고 장비를 구매하는 주체는 연구책임자다.
    df = df[df["last_author_is_korea"] == True].copy()
    print(f"교신저자 한국 소속: {len(df)}건")

    df["last_author_aff"] = df["last_author_aff"].fillna("")
    df["email"] = df["last_author_aff"].apply(extract_email)

    resolved = df["last_author_aff"].apply(resolve_institution)
    df["institution"] = [r[0] for r in resolved]
    df["inst_type"] = [r[1] for r in resolved]
    df["resolve_method"] = [r[2] for r in resolved]

    # 이메일 도메인은 기관 판정의 교차 검증 근거로 남긴다
    df["email_domain"] = df["email"].apply(lambda e: e.split("@")[-1] if e else "")

    df.to_csv(OUT, index=False, encoding="utf-8-sig")

    ok = (df["institution"] != "").sum()
    print(f"\n기관 식별 성공: {ok}/{len(df)} = {ok / len(df) * 100:.1f}%")
    print(f"  - 사전 매칭:   {(df['resolve_method'] == '사전').sum()}")
    print(f"  - 일반 패턴:   {(df['resolve_method'] == '일반패턴').sum()}")
    print(f"  - 실패:        {(df['resolve_method'] == '실패').sum()}")
    print(f"교신저자 이메일 확보: {(df['email'] != '').sum()} ({(df['email'] != '').mean() * 100:.1f}%)")

    print("\n[기관 유형 분포]")
    print(df["inst_type"].value_counts().to_string())

    print("\n[기관 상위 25]")
    print(df[df["institution"] != ""]["institution"].value_counts().head(25).to_string())

    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()