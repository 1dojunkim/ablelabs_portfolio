"""
논문 기반 액체 핸들링 자동화 타겟 리드 발굴 파이프라인
5단계: 관문 통과와 정렬

이 단계의 설계 원칙은 하나다. 임의로 정한 숫자를 쓰지 않는다.

여러 지표를 하나의 점수로 합치려면 가중치가 필요하다.
'활동량 40점 + 니즈 25점'의 40과 25를 정당화할 근거는 없다.
근거가 생기는 유일한 경우는 실제 구매 이력으로 가중치를 학습시킬 때이고,
그 데이터는 에이블랩스 CRM 안에 있으며 외부에서는 접근할 수 없다.

문턱값도 마찬가지다. '수작업 강도 3 이상'의 3에도 근거가 없다.
그래서 문턱값을 고를 필요가 없는 것만 관문으로 남겼다.

  관문 1: 액체 핸들링을 직접 하는가       (참/거짓. 외주 준 곳에는 로봇을 팔 수 없다)
  관문 2: 교신저자에게 닿을 수 있는가      (참/거짓. 연락처가 없으면 파이프라인에 올릴 수 없다)

나머지는 전부 정렬로 처리한다. 정렬에는 문턱값이 필요 없다.
그리고 목록을 어디서 자를지는 데이터가 아니라 영업 capacity가 정한다.
(CAPACITY 변수를 바꾸면 그만큼만 출력된다)

최신성에 대하여.
  '2년 전 논문이면 지금은 안 한다'는 판단도 임의적이다.
  그래서 이 데이터로 직접 측정했다. 어떤 연도에 프로토콜 P를 수행한 연구실이
  2025년 이후 같은 프로토콜로 다시 논문을 낸 비율은 다음과 같다.
      2022년 활동 → 15.0%      2023년 → 17.9%      2024년 → 21.7%
  최신성은 신호이되 절벽이 없다. 잘라낼 지점이 데이터에 없다는 뜻이므로 관문으로 쓰지 않고,
  동점을 가르는 2순위 정렬 기준으로만 쓴다.
  또한 이 수치는 '실험을 그만둔 비율'이 아니라 '논문으로 드러난 비율'이다.
  매주 ELISA를 돌리면서 3년에 한 번 논문을 내는 연구실이 많기 때문에,
  이 값은 지속률의 하한이지 지속률 자체가 아니다.

사용법:
    python 05_rank.py
입력:
    data/labs_judged.csv
출력:
    data/leads.csv           (관문 통과 리드 전체, 세그먼트·정렬 적용)
    data/leads_top.csv       (영업 capacity 만큼 자른 최종 목록)
"""

import os
import pandas as pd

IN = "data/labs_judged.csv"
OUT_ALL = "data/leads.csv"
OUT_TOP = "data/leads_top.csv"

# 프로토콜 → 기본 제품. 실험의 성격으로 1차 결정한다.
PROTOCOL_TO_PRODUCT = {
    "ELISA": "NOTABLE96", "qPCR": "NOTABLE96", "drug_screening": "NOTABLE96",
    "NGS_library_prep": "SUITABLE", "protein_purification": "SUITABLE",
    "cell_line_development": "NOTABLE", "serial_dilution": "NOTABLE",
}
PRODUCT_RANK = {"NOTABLE": 1, "NOTABLE96": 2, "SUITABLE": 3}
RANK_PRODUCT = {v: k for k, v in PRODUCT_RANK.items()}


def decide_product(protocols, throughput, inst_type, n_protocols):
    """제품을 결정한다.
    1차: 프로토콜의 성격으로 기본 제품을 정한다.
         (96웰 반복 → NOTABLE96, 가변 용량 정밀 → SUITABLE, 단순 반복 → NOTABLE)
    2차: AI가 판정한 처리량(throughput)으로 보정한다.
         처리량은 '단품이냐 고처리량 대응이냐'를 가르는 데 쓰고,
         제품의 종류 자체는 프로토콜이 정한 것을 존중한다.
         - 고처리량 + 복수 실험 + 대형 기관 → Lab Automation(장비 연계)
         - 저처리량 + 단일 실험 → 프로토콜이 무엇이든 소형 단품 NOTABLE
         - 그 외 → 프로토콜 기준 유지
    """
    base = [PROTOCOL_TO_PRODUCT[p] for p in protocols if p in PROTOCOL_TO_PRODUCT]
    if not base:
        return "", ""
    product = RANK_PRODUCT[max(PRODUCT_RANK[b] for b in base)]  # 프로토콜 기준 제품

    big_org = inst_type in ("대학병원", "기업", "정부출연연")
    if throughput == "high" and n_protocols >= 3 and big_org:
        return "Lab Automation", "고처리량·복수 실험·대형 기관 → 장비 연계"

    if throughput == "low" and n_protocols == 1:
        return "NOTABLE", "저처리량·단일 실험 → 소형 단품"

    return product, "프로토콜 기준"

# 이번 분기에 접촉 가능한 리드 수.
# 이것은 데이터에서 도출한 값이 아니라 영업 조직의 제약 조건이다.
# 데이터는 순서를 알려주고, 몇 개까지 볼지는 사람이 정한다.
CAPACITY = 100


def main():
    if not os.path.exists(IN):
        raise SystemExit(f"{IN} 이 없습니다. 04_classify.py 를 먼저 실행하세요.")

    labs = pd.read_csv(IN)
    total = len(labs)
    judged = labs["manual_load"].notna().sum()
    print(f"연구실 총계: {total}개")
    print(f"LLM 판정 완료: {judged}개 ({judged/total*100:.0f}%)")
    if judged < total:
        print(f"  (미판정 {total-judged}개는 관문 통과 대상에서 제외됩니다.")
        print("   내일 04를 다시 돌리면 캐시 이후분이 채워집니다)")
    print()

    # -----------------------------------------------------------------------
    # 관문
    # -----------------------------------------------------------------------
    print("[관문 통과]")

    # 판정 자체가 실패한 건은 판단 근거가 없으므로 제외한다
    step = labs[labs["manual_load"].notna()].copy()
    print(f"  판정 성공          {len(step):5d} / {total}")

    # 관문 1. 외주를 준 연구실에서는 액체 핸들링이 일어나지 않는다.
    step = step[step["in_house"] == True]
    print(f"  1. 직접 수행       {len(step):5d}")

    # 관문 2. 닿을 수 없는 대상은 리드가 아니다.
    step = step[step["email"].fillna("") != ""]
    print(f"  2. 연락처 확보     {len(step):5d}")

    leads = step.copy()

    # -----------------------------------------------------------------------
    # 정렬
    # -----------------------------------------------------------------------
    # 1순위 수작업 강도  : 자동화로 해결할 고통의 크기. 제안의 근거가 여기서 나온다.
    # 2순위 최신 논문 연도: 측정된 지속률(15.0/17.9/21.7%)에 근거해 동점을 가른다.
    # 3순위 논문 수      : 실험 활동량.
    # 합산하지 않고 사전식으로 정렬하므로 가중치가 필요 없다.
    leads = leads.sort_values(
        ["manual_load", "latest_year", "n_papers"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    # 세그먼트. 접근 방식이 다른 대상을 한 줄로 세우지 않는다.
    # 제품 결정 (프로토콜 + 처리량 반영)
    prod = leads.apply(
        lambda r: decide_product(
            str(r["protocols"]).split("|"),
            r.get("throughput", ""),
            r["inst_type"],
            r["n_protocols"],
        ),
        axis=1,
    )
    leads["product"] = [p[0] for p in prod]
    leads["product_reason"] = [p[1] for p in prod]

    leads["segment"] = leads["inst_type"] + " · " + leads["product"].fillna("")
    leads.insert(0, "rank", leads.index + 1)

    leads.to_csv(OUT_ALL, index=False, encoding="utf-8-sig")

    top = leads.head(CAPACITY)
    top.to_csv(OUT_TOP, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------------------------
    print(f"\n영업 가능 리드: {len(leads)}개 (전체의 {len(leads) / total * 100:.1f}%)")
    print(f"이번 분기 대상: 상위 {len(top)}개  (CAPACITY={CAPACITY})")

    print("\n[수작업 강도별 리드 수]")
    print(leads["manual_load"].value_counts().sort_index(ascending=False).to_string())

    print("\n[세그먼트별 (전체 리드)]")
    print(leads["segment"].value_counts().head(12).to_string())

    print("\n[세그먼트별 (상위 %d)]" % CAPACITY)
    print(top["segment"].value_counts().to_string())

    print("\n[상위 20]")
    cols = ["rank", "pi_name", "institution", "inst_type", "product",
            "manual_load", "n_papers", "latest_year", "pain_point"]
    cols = [c for c in cols if c in top.columns]
    print(top[cols].head(20).to_string(index=False))

    print(f"\n저장: {OUT_ALL} / {OUT_TOP}")


if __name__ == "__main__":
    main()