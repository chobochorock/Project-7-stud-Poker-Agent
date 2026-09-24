from __future__ import annotations

import copy
import datetime as dt
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


HP_URI = "http://www.hancom.co.kr/hwpml/2011/paragraph"
HP = f"{{{HP_URI}}}"

NAMESPACES = {
    "ha": "http://www.hancom.co.kr/hwpml/2011/app",
    "hp": HP_URI,
    "hp10": "http://www.hancom.co.kr/hwpml/2016/paragraph",
    "hs": "http://www.hancom.co.kr/hwpml/2011/section",
    "hc": "http://www.hancom.co.kr/hwpml/2011/core",
    "hh": "http://www.hancom.co.kr/hwpml/2011/head",
    "hhs": "http://www.hancom.co.kr/hwpml/2011/history",
    "hm": "http://www.hancom.co.kr/hwpml/2011/master-page",
    "hpf": "http://www.hancom.co.kr/schema/2011/hpf",
    "dc": "http://purl.org/dc/elements/1.1/",
    "opf": "http://www.idpf.org/2007/opf/",
    "ooxmlchart": "http://www.hancom.co.kr/hwpml/2016/ooxmlchart",
    "hwpunitchar": "http://www.hancom.co.kr/hwpml/2016/HwpUnitChar",
    "epub": "http://www.idpf.org/2007/ops",
    "config": "urn:oasis:names:tc:opendocument:xmlns:config:1.0",
}

ACTIVITIES = [
    ("2026.07.14", "23:16", "23:54", "38분", "Stackless EV와 net-chip 기준 캐시게임 구조 설계"),
    ("2026.07.15", "17:38", "22:35", "4시간 57분", "상대 핸드 테이블·균등 rollout·정확 EV 데이터 수집 설계"),
    (
        "2026.07.16",
        "20:30",
        "23:18",
        "2시간 48분",
        "Dot-product·k-means·GMM-EM 군집화 구현 및 비교",
    ),
    (
        "2026.07.17",
        "00:08",
        "02:26",
        "2시간 18분",
        "UCT 대 UCT·고정 핸드·descendant 저장·신뢰구간 정지 검토",
    ),
    (
        "2026.07.18",
        "00:09",
        "00:27",
        "18분",
        "GMM Q-learning·SARSA 선행연구와 belief-state 입력 검토",
    ),
    ("2026.07.19", "02:35", "03:09", "34분", "Kuhn CFR 원형 구현과 7-Stud 연결 방식 설계"),
    (
        "2026.07.20",
        "00:02",
        "01:15",
        "1시간 13분",
        "7th→6th+→5th+ 단계형 MCCFR 및 heuristic bucket 실험",
    ),
    (
        "2026.07.21",
        "20:38",
        "22:07",
        "1시간 29분",
        "Bottom-up confidence bound와 toy POMDP 기준선 실험",
    ),
    ("2026.07.23", "13:54", "14:11", "17분", "고정 휴리스틱 상대 POMDP best-response 성능 점검"),
    (
        "2026.07.24",
        "23:10",
        "23:45",
        "35분",
        "Asymmetric perturbation 논문 검토와 KL·entropy 변형 구상",
    ),
    (
        "2026.07.25",
        "00:01",
        "00:14",
        "13분",
        "PED·exploitability descent·다인 saddle-point 확장 검토",
    ),
    (
        "2026.07.27",
        "02:52",
        "06:23",
        "3시간 31분",
        "Betting rules v3·Stud-Leduc·2계층 HRL 실험 설계",
    ),
    ("2026.07.27", "12:26", "16:22", "3시간 56분", "C++ 환경·에이전트 분리와 CFR/MCCFR solver 구축"),
    (
        "2026.07.28",
        "09:09",
        "10:57",
        "1시간 48분",
        "10M MCCFR checkpoint 검증과 ante/chip 단위 정비",
    ),
    (
        "2026.07.29",
        "01:18",
        "03:55",
        "2시간 37분",
        "Policy-aware BR/LBR 구조와 기존 평가기의 차이 분석",
    ),
    (
        "2026.07.29",
        "13:11",
        "18:40",
        "5시간 29분",
        "Hard-to-soft temperature·adaptive split/prune 군집 실험",
    ),
    (
        "2026.07.29",
        "21:00",
        "23:39",
        "2시간 39분",
        "순정 hard cluster 기준선과 residual·centroid 보정 비교",
    ),
    (
        "2026.07.30",
        "17:32",
        "20:26",
        "2시간 54분",
        "Perfect recall 게이트·7th subgame AsymP·다인 평가 계획",
    ),
    ("2026.07.30", "21:46", "23:32", "1시간 46분", "스택 민감도·5인 cash·team-LBR 평가 구조 설계"),
    (
        "2026.07.31",
        "18:46",
        "23:50",
        "5시간 4분",
        "100M 확장 판단·H4 학습·K128 warm start·Deep CFR 계획",
    ),
    (
        "2026.08.01",
        "00:10",
        "05:11",
        "5시간 1분",
        "LBR 캐싱·paired 평가·imitation lambda와 외부 피드백 분석",
    ),
    ("2026.08.01", "21:20", "23:57", "2시간 37분", "Deep CFR 공간 효율·fold 기준선·조건부 참여 봇 실험"),
    (
        "2026.08.02",
        "22:49",
        "23:48",
        "59분",
        "Deep CFR 모델 크기·reservoir memory·iteration 자원 조정",
    ),
    (
        "2026.08.03",
        "00:00",
        "05:58",
        "5시간 58분",
        "초기 regret seed와 K128/K256/K512 atlas 비교",
    ),
    ("2026.08.03", "10:30", "12:45", "2시간 15분", "Half-bot 대전과 평균전략 회계·고정점 원인 진단"),
    (
        "2026.08.03",
        "20:34",
        "22:58",
        "2시간 24분",
        "Betting memory와 bluff-defense·hidden-bluff 정책 풀 구성",
    ),
    ("2026.08.04", "00:40", "01:07", "27분", "Memory16 30M 결과와 평균전략 entropy 진단"),
    ("2026.08.09", "17:25", "17:56", "31분", "Reach 기반 street별 16/64/256 비균등 atlas 설계"),
    (
        "2026.08.10",
        "22:38",
        "00:01(+1)",
        "1시간 23분",
        "7th exact betting/hand history local MCCFR resolver 구현",
    ),
    (
        "2026.08.11",
        "11:40",
        "15:16",
        "3시간 36분",
        "Adaptive split 안정화·RAM·resolver sweep·ReBeL 계획",
    ),
    (
        "2026.08.13",
        "00:02",
        "00:13",
        "11분",
        "Particle PBS·V7 value target 기반 ReBeL 모듈 구현",
    ),
    (
        "2026.08.17",
        "23:35",
        "00:29(+1)",
        "54분",
        "ReBeL 240-particle LBR 검증과 checkpoint 비교",
    ),
    (
        "2026.08.21",
        "22:24",
        "00:43(+1)",
        "2시간 19분",
        "원 ReBeL 대비 구현 차이·병목·seminar discussion 검토",
    ),
    (
        "2026.08.22",
        "17:05",
        "19:05",
        "2시간",
        "Online PBS search·CFV target·실시간 학습 구조 실험",
    ),
    (
        "2026.08.22",
        "21:39",
        "00:08(+1)",
        "2시간 29분",
        "핸드별 sampled BR-gap 지표와 log 그래프 설계",
    ),
    (
        "2026.08.23",
        "00:08",
        "02:39",
        "2시간 31분",
        "Heuristic/K512 BR-gap 곡선 생성과 평가 속도 개선",
    ),
    ("2026.08.29", "21:01", "21:34", "33분", "전체 실험·개념 일지 작성과 연구 성과 회고"),
    ("2026.08.30", "16:16", "16:28", "12분", "학부연구생 최종보고서·활동일지 초안 작성"),
    ("2026.08.31", "14:03", "14:38", "35분", "공식 HWPX 양식 이관과 채팅 기반 활동시각 복원"),
]

for prefix, uri in NAMESPACES.items():
    ET.register_namespace(prefix, uri)


def cell_map(table: ET.Element) -> dict[tuple[int, int], ET.Element]:
    cells: dict[tuple[int, int], ET.Element] = {}
    for cell in table.iter(HP + "tc"):
        addr = cell.find(HP + "cellAddr")
        if addr is not None:
            cells[(int(addr.attrib["rowAddr"]), int(addr.attrib["colAddr"]))] = cell
    return cells


def paragraphs(cell: ET.Element) -> list[ET.Element]:
    sub_list = cell.find(HP + "subList")
    assert sub_list is not None
    return sub_list.findall(HP + "p")


def set_paragraph_text(paragraph: ET.Element, text: str) -> None:
    runs = paragraph.findall(HP + "run")
    if not runs:
        runs = [ET.SubElement(paragraph, HP + "run", {"charPrIDRef": "11"})]
    first = runs[0]
    for run in runs:
        for node in list(run):
            if node.tag == HP + "t":
                run.remove(node)
    ET.SubElement(first, HP + "t").text = text
    for run in runs[1:]:
        paragraph.remove(run)
    line_segments = paragraph.find(HP + "linesegarray")
    if line_segments is not None:
        paragraph.remove(line_segments)


def set_cell_text(cell: ET.Element, text: str) -> None:
    set_paragraph_text(paragraphs(cell)[0], text)


def serialize(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def write_hwpx(source: Path, target: Path, replacements: dict[str, bytes]) -> None:
    with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(target, "w") as dst:
        for info in src.infolist():
            dst.writestr(info, replacements.get(info.filename, src.read(info.filename)))


def load_section(path: Path) -> ET.Element:
    with zipfile.ZipFile(path) as archive:
        return ET.fromstring(archive.read("Contents/section0.xml"))


def fill_report(source: Path, target: Path) -> None:
    root = load_section(source)
    table = next(root.iter(HP + "tbl"))
    cells = cell_map(table)

    replacements = {
        (2, 1): "[대학교 입력]",
        (2, 2): "[학부·전공 입력]",
        (3, 1): "[성명 입력]",
        (3, 3): "[학번 입력]",
        (4, 2): "해당 없음",
        (5, 1): "[휴대전화 입력]",
        (5, 3): "[전자메일 입력]",
        (7, 1): "해당 없음 (개인 연구)",
        (7, 3): "해당 없음",
        (8, 1): "해당 없음",
    }
    for address, text in replacements.items():
        set_cell_text(cells[address], text)

    body = paragraphs(cells[(9, 0)])
    activity_lines = [
        "연구주제: 불완전정보 7-Stud 포커의 정보 추상화와 CFR 기반 전략 학습",
        "• 환경 구축: Python 초기 환경을 C++로 이전하고 5구 betting rules v3, UCT, MCCFR, LBR 평가기를 구현함.",
        "• 초기 진단: 146만 hands의 정확 상태 rollout에서 재방문율이 0.000384%에 불과해 함수근사와 정보 추상화의 필요성을 확인함.",
        "• 군집 실험: MLP, spherical k-means, GMM-EM을 비교했으며 밀도 유사성이 전략 유사성을 보장하지 않음을 실제 대전으로 확인함.",
        "• 이론 검증: Kuhn·Stud-Leduc에서 exact CFR+와 MCCFR을 구현하고 exact exploitability로 solver의 건전성을 확인함.",
        "• 7-Stud 기준선: hard power bucket과 external-sampling MCCFR을 학습해 휴리스틱과 통계적 무승부 수준의 정책을 구축함.",
        "• 취약성 평가: policy-aware LBR과 sampled BR-gap을 구현해 추상 게임의 낮은 regret과 원 게임 취약성의 차이를 발견함.",
        "• 확장·세미나: resolver, ReBeL, Deep CFR를 비교하고 CFR부터 ReBeL까지 핵심 논문 9편을 검토하여 Joint Seminar 발표자료로 종합함.",
        "• 세미나 준비: CFR, MCCFR, CFR+, Cepheus, decomposition, DeepStack, Libratus, Deep CFR, ReBeL 논문을 검토하고 Joint Seminar 발표자료로 종합함.",
    ]
    for paragraph, text in zip(body[2:11], activity_lines):
        set_paragraph_text(paragraph, text)

    reflection = [
        "불완전정보 게임에서는 특정 휴리스틱에게 이기는 성능과 균형에 가까운 방어 성능을 구분해야 함을 배웠다. 낮은 training regret도 정보 추상화가 서로 다른 상황을 합치면 원 게임의 낮은 exploitability를 보장하지 않았다.",
        "특히 매우 좋아 보였던 posterior resolver 결과에서 대규모 blueprint fallback을 발견하고 결론을 철회하면서, 점수보다 평가 정책과 실제 실행 정책의 일관성이 우선이라는 연구 원칙을 체감했다.",
        "향후에는 동일 node budget의 MCCFR/MCCFR+ 비교, reach와 regret conflict 기반 선택적 refinement, joint-PBS safe resolving을 우선 검토하고자 한다. 프로그램 운영 시 실제 연구사진과 중간발표 기록을 정기적으로 남길 수 있는 공통 안내가 제공되면 보고서 작성에 도움이 될 것이다.",
    ]
    set_paragraph_text(body[12], reflection[0])
    sub_list = cells[(9, 0)].find(HP + "subList")
    assert sub_list is not None
    for text in reflection[1:]:
        paragraph = copy.deepcopy(body[12])
        set_paragraph_text(paragraph, text)
        sub_list.append(paragraph)

    set_paragraph_text(body[10], "• 활동사진: [실험 수행·발표 장면 사진을 이 위치에 삽입]")

    closing = paragraphs(cells[(10, 0)])
    set_paragraph_text(closing[3], "2026년 8월 31일")
    set_paragraph_text(closing[5], "참여자        [성명 입력]        (인/서명)")

    preview = """<SW·AI학부연구생 프로그램 활동 최종보고서>
<참여자><성명 입력><학번 입력><대학교·학부·전공 입력>
<활동기간><2026년 7월 1일 ~ 2026년 8월 31일>
<주요활동내용><7-Stud 환경, 정보 추상화, CFR/MCCFR, LBR, resolver, ReBeL, Deep CFR 구현 및 비교>
<소감><활용 성능과 방어 성능, 추상화 오차와 최적화 오차를 구분하는 연구 절차를 학습함>
<주의><활동사진과 개인정보, 서명을 제출 전에 입력해야 함>
""".encode(
        "utf-8"
    )
    write_hwpx(
        source,
        target,
        {"Contents/section0.xml": serialize(root), "Preview/PrvText.txt": preview},
    )


def set_cell_style(cell: ET.Element, char_id: int, para_id: int) -> None:
    for paragraph in paragraphs(cell):
        paragraph.set("paraPrIDRef", str(para_id))
        for run in paragraph.findall(HP + "run"):
            run.set("charPrIDRef", str(char_id))


def set_cell_margin(
    cell: ET.Element, left: int, right: int, top: int, bottom: int
) -> None:
    margin = cell.find(HP + "cellMargin")
    assert margin is not None
    margin.attrib.update(
        {"left": str(left), "right": str(right), "top": str(top), "bottom": str(bottom)}
    )


def compact_activity_table(
    table: ET.Element,
    activities: list[tuple[str, str, str, str, str]],
    include_total: bool,
) -> None:
    """Build a compact page table, optionally ending with the grand total."""
    rows = table.findall(HP + "tr")
    assert len(rows) == 26
    body_template = copy.deepcopy(rows[1])
    total_template = copy.deepcopy(rows[-1])
    for row in rows[1:]:
        table.remove(row)

    for _ in activities:
        table.append(copy.deepcopy(body_template))
    if include_total:
        table.append(total_template)

    rows = table.findall(HP + "tr")
    table.set("rowCnt", str(len(rows)))
    for row_index, row in enumerate(rows):
        for cell in row.findall(HP + "tc"):
            address = cell.find(HP + "cellAddr")
            assert address is not None
            address.set("rowAddr", str(row_index))

            size = cell.find(HP + "cellSz")
            assert size is not None
            if 0 < row_index <= len(activities):
                size.set("height", "2015")
                if address.get("colAddr") == "6":
                    set_cell_margin(cell, 180, 180, 50, 50)
                else:
                    set_cell_margin(cell, 120, 120, 50, 50)
            elif include_total and row_index == len(rows) - 1:
                size.set("height", "2600")

    table_size = table.find(HP + "sz")
    assert table_size is not None
    header_height = max(
        int(cell.find(HP + "cellSz").get("height"))
        for cell in rows[0].findall(HP + "tc")
    )
    total_height = 2600 if include_total else 0
    table_size.set("height", str(header_height + len(activities) * 2015 + total_height))


def style_activity_table(
    table: ET.Element, activity_count: int, include_total: bool
) -> None:
    cells = cell_map(table)
    for column in range(7):
        set_cell_style(cells[(0, column)], char_id=11, para_id=24)

    for row in range(1, activity_count + 1):
        for column in range(6):
            set_cell_style(cells[(row, column)], char_id=20, para_id=24)
        set_cell_style(cells[(row, 6)], char_id=20, para_id=11)

    if include_total:
        total_row = activity_count + 1
        set_cell_style(cells[(total_row, 5)], char_id=11, para_id=24)
        set_cell_style(cells[(total_row, 6)], char_id=20, para_id=11)
        set_cell_margin(cells[(total_row, 6)], 180, 180, 80, 80)


def fill_activity_table(
    table: ET.Element,
    activities: list[tuple[str, str, str, str, str]],
    start_number: int,
    total_text: str | None = None,
) -> None:
    include_total = total_text is not None
    compact_activity_table(table, activities, include_total)
    cells = cell_map(table)
    weekdays = "월화수목금토일"
    for offset in range(len(activities)):
        row = offset + 1
        set_cell_text(cells[(row, 0)], str(start_number + offset))
        date_text, start, end, duration, description = activities[offset]
        date = dt.datetime.strptime(date_text, "%Y.%m.%d").date()
        values = {
            1: f"{date.month}월 {date.day}일",
            2: weekdays[date.weekday()],
            3: start,
            4: end,
            5: duration,
            6: description,
        }
        for column, text in values.items():
            set_cell_text(cells[(row, column)], text)
    if include_total:
        total_row = len(activities) + 1
        set_cell_text(cells[(total_row, 5)], total_text)
        set_cell_text(cells[(total_row, 6)], "채팅 turn 시각 기반 복원; 무인 학습 시간 제외")
    style_activity_table(table, len(activities), include_total)


def fill_journal(source: Path, target: Path) -> None:
    root = load_section(source)
    top_level = list(root)
    info_paragraph = top_level[2]
    table_paragraph = top_level[3]
    info_table = next(info_paragraph.iter(HP + "tbl"))
    info_cells = cell_map(info_table)
    info_values = {
        (0, 1): "[대학교 입력]",
        (0, 3): "[학부(과) 입력]",
        (1, 1): "[성명 입력]",
        (1, 3): "[학번 입력]",
        (2, 3): "[활동장소 입력]",
    }
    for address, text in info_values.items():
        set_cell_text(info_cells[address], text)

    activities = ACTIVITIES
    assert len(activities) == 39
    first_page = activities[:25]
    second_page = activities[25:]

    table_template = copy.deepcopy(table_paragraph)
    first_table = next(table_paragraph.iter(HP + "tbl"))
    fill_activity_table(first_table, first_page, 1)

    second_table_paragraph = copy.deepcopy(table_template)
    second_table_paragraph.set("pageBreak", "1")
    second_table = next(second_table_paragraph.iter(HP + "tbl"))
    fill_activity_table(second_table, second_page, 26, "81시간 59분")
    signature_index = list(root).index(top_level[4])
    root.insert(signature_index, second_table_paragraph)

    set_paragraph_text(top_level[6], "2026년 8월 31일")
    set_paragraph_text(top_level[8], "학부연구생 : [성명 입력]        (인/서명)")

    preview_lines = [
        "『SW·AI학부연구생』 활동일지",
        "<성명><성명 입력><학번><학번 입력>",
        "<활동기간><2026.07.01 ~ 2026.08.31>",
        "<복원기준><Codex 채팅 turn 시각; 45분 이내 연속 turn 병합; 무인 학습 제외>",
    ]
    for number, (date_text, start, end, duration, description) in enumerate(
        activities, 1
    ):
        preview_lines.append(
            f"<{number}><{date_text}><{start}><{end}><{duration}><{description}>"
        )
    write_hwpx(
        source,
        target,
        {
            "Contents/section0.xml": serialize(root),
            "Preview/PrvText.txt": ("\n".join(preview_lines) + "\n").encode("utf-8"),
        },
    )


def validate(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        assert archive.infolist()[0].filename == "mimetype"
        assert archive.infolist()[0].compress_type == zipfile.ZIP_STORED
        ET.fromstring(archive.read("Contents/section0.xml"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    experiment = Path(__file__).resolve().parents[2]
    report_source = experiment / "성명_⌈SW·AI학부연구생⌋ 프로그램 활동 최종보고서.hwpx"
    journal_source = experiment / "성명_⌈SW·AI학부연구생⌋ 활동일지.hwpx"
    report_target = experiment / "성명_⌈SW·AI학부연구생⌋ 프로그램 활동 최종보고서_작성초안.hwpx"
    journal_target = experiment / "성명_⌈SW·AI학부연구생⌋ 활동일지_작성초안.hwpx"

    fill_report(report_source, report_target)
    fill_journal(journal_source, journal_target)
    validate(report_target)
    validate(journal_target)
    print(report_target)
    print(journal_target)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise
