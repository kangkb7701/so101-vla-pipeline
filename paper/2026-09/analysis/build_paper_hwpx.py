"""Build the autumn-conference paper (HWPX) from the spring paper's file, keeping its page/column/paragraph/character
styles: every new paragraph is a copy of a template paragraph from the original with only the text replaced.
Line-layout caches (hp:linesegarray) are dropped so Hancom re-lays out the text. Figure 1 keeps the original
figure box (same size) with the image replaced by figs/paper_fig1.png.
Input : ~/Desktop/한국정보통신학회 2026 춘계종합학술대회_발표문_강경빈.hwpx (not modified)
Output: ~/Desktop/한국정보통신학회 2026 추계종합학술대회_발표문_강경빈.hwpx
"""
import re, zipfile
from pathlib import Path
from xml.sax.saxutils import escape
import xml.dom.minidom

SRC = Path.home() / "Desktop" / "한국정보통신학회 2026 춘계종합학술대회_발표문_강경빈.hwpx"
DST = Path.home() / "Desktop" / "한국정보통신학회 2026 추계종합학술대회_발표문_강경빈.hwpx"
FIG = Path(__file__).resolve().parent / "figs" / "paper_fig1.png"

zin = zipfile.ZipFile(SRC)
s = zin.read("Contents/section0.xml").decode("utf-8")
P = [m.start() for m in re.finditer(r'<hp:p id="[^"]*" paraPrIDRef="\d+"', s)]


def open_tag(i):
    return s[P[i]:s.index(">", P[i]) + 1]


def own_end(i):
    return s.index("</hp:p>", P[i]) + len("</hp:p>")


def para(tag, runs):
    body = "".join(f'<hp:run charPrIDRef="{c}"><hp:t>{escape(t)}</hp:t></hp:run>' if t else f'<hp:run charPrIDRef="{c}"/>'
                   for c, t in runs)
    return tag + body + "</hp:p>"


def raw(a, b):
    return s[P[a]:P[b]]


ABS_KO = ("본 연구는 저비용 로봇 SO-101에서 행동 표현이 사전 학습된 VLA(Vision-Language-Action) 모델의 파인튜닝 성능에 미치는 "
    "영향을 분석한다. 절대 관절각, 상대 관절각, 절대 엔드이펙터(EE) 자세, 상대 EE 자세의 네 가지 표현으로 SmolVLA를 같은 조건에서 "
    "파인튜닝하고, 표현마다 20회씩 실제 로봇으로 평가하며 모델 출력과 모터 명령, 관절 측정값을 함께 분석했다. 학습 데이터에 대한 "
    "예측 정확도는 네 표현이 비슷했으나, 실제 로봇에서 바나나를 잡은 횟수는 절대 관절각이 6회, 나머지 표현이 각각 1회였다. 상대 "
    "표현과 EE 표현은 모터가 반응하지 않을 만큼 작은 이동을 반복해서 지시하여 로봇이 멈추거나 물체에 다가가지 못했고, 네 표현 모두 "
    "그리퍼와 손목 명령을 현재 관절 값에 크게 의존하여 정했다.")
ABS_EN = ("This study analyzes how the action representation affects the fine-tuning performance of a pretrained "
    "Vision-Language-Action (VLA) model on the low-cost SO-101 robot. SmolVLA was fine-tuned with absolute joint angles, "
    "relative joint angles, absolute end-effector (EE) poses, and relative EE poses under identical conditions and evaluated in "
    "20 real-robot trials per representation. Prediction accuracy on the training data was similar across representations, "
    "whereas the absolute joint representation grasped the object in 6 trials and the others in only one each. The relative "
    "and EE representations repeatedly commanded motions too small for the servo motors to respond to, which stalled the "
    "robot, and all representations relied heavily on the current joint values for "
    "their gripper and wrist commands.")
INTRO = [
    "사전 학습된 VLA(Vision-Language-Action) 모델을 소규모 시연 데이터로 파인튜닝하여 저비용 로봇에 적용하려는 연구가 "
    "활발하다[1-3]. VLA 모델의 행동 표현으로는 절대 관절각, 현재 상태 기준의 상대 변위, 엔드이펙터(EE) 자세 등이 쓰이며, "
    "최근에는 현재 상태 기준의 상대 표현이 절대 표현보다 파인튜닝 성공률이 높다는 결과가 시뮬레이션[4]과 실제 "
    "로봇[5]에서 보고됐다.",
    "그러나 이러한 결과는 주로 정밀한 구동기를 갖춘 로봇에서 얻어졌다. 저가 서보 모터를 쓰는 로봇[6]은 목표 위치가 조금만 바뀌는 "
    "명령에 잘 반응하지 않는데, 이 특성이 행동 표현에 따라 성능에 다르게 작용하는지는 검증되지 않았다.",
    "본 연구에서는 6자유도 저비용 로봇 SO-101[7]에서 SmolVLA[3]를 네 가지 행동 표현으로 같은 조건에서 파인튜닝하고, 실제 로봇 평가와 "
    "함께 모델 출력, 모터 명령, 관절 측정값을 분석하여 실패 원인을 밝힌다.",
]
SEC2 = [
    ("sub", "2.1 행동 표현"),
    ("body", "네 표현 모두 1.5초 분량인 15개의 행동을 한 번에 예측하고, 이를 10Hz로 모두 실행한 뒤 다음 15개를 예측한다. 이 묶음을 "
             "행동 청크라 한다. 절대 관절각 표현은 관절 5개와 그리퍼의 목표 각도를, 상대 관절각 표현은 청크를 시작할 때 측정한 관절각에서의 "
             "변화량을 출력한다. 절대 EE 표현은 순기구학으로 계산한 그리퍼 위치와 6D 회전 표현[8], 그리퍼 열림을 출력하며, 상대 EE 표현은 "
             "청크 시작 시점의 EE 값과의 차이를 성분별로 출력한다. EE 표현은 역기구학(IK)으로 관절각을 구해 실행한다."),
    ("blank", ""),
    ("sub", "2.2 학습 및 평가"),
    ("body", "SmolVLA는 LeRobot[9]으로 전체 파인튜닝했다. 학습 데이터는 바나나를 지시한 색의 바구니로 옮기는 원격조작 시연 180개이며, "
             "상단과 측면에 고정한 카메라 2대의 영상을 입력으로 사용했다. 학습은 네 표현 모두 24에폭, 배치 크기 64, 학습률 1e-4로 같게 "
             "했다. 로봇의 각 관절은 1/345 감속비의 Feetech STS3215 서보 모터로 구동되며, 모터는 0.088° 단위로 위치를 측정하고 "
             "제어한다."),
    ("body", "평가는 표현마다 20회 수행했고, 바나나 위치 6곳과 방향, 방해물인 사과와 당근의 유무, 목표 바구니 색을 조합한 같은 "
             "시나리오를 사용했다. 녹화 영상을 판독하여 바나나가 목표 바구니 안에 들어가면 성공, 바구니 경계에 걸치면 부분 성공으로 "
             "분류했다. 또한 녹화된 영상은 그대로 두고 관절 값 하나만 바꾸어 모델을 다시 실행함으로써, 모델이 어떤 입력을 근거로 행동을 "
             "정하는지 분석했다."),
]
SEC3 = [
    ("sub", "3.1 표현별 평가 결과"),
    ("body", "그림 1(a)와 같이 절대 관절각 표현은 20회 중 6회 바나나를 잡아 2회는 목표 바구니 안에, 3회는 바구니 경계에 "
             "놓았고 1회는 바구니 밖에 떨어뜨렸다. 나머지 세 표현은 각각 1회만 바나나를 잡았으며, 상대 관절각은 바구니 안에, 절대 EE와 "
             "상대 EE는 바구니 경계에 놓았다."),
    ("figure", ""),
    ("body", "실패 양상도 표현에 따라 달랐다. 절대 관절각 표현은 실패 15회 중 12회에서 그리퍼가 바나나까지 갔으나, 바나나를 밀어내거나 "
             "옆을 잡는 등 마지막 잡기 단계에서 실패했다. 반면 상대 관절각, 절대 EE, 상대 EE 표현은 실패 19회 중 바나나까지 간 경우가 "
             "각각 4회, 5회, 8회에 불과했다. 이들 표현에서는 바나나에 다가가기 전에 그리퍼를 닫아 빈 그리퍼로 바구니로 간 경우가 각각 "
             "9회, 8회, 8회였고, 처음 자세에서 끝까지 움직이지 않은 경우가 각각 6회, 6회, 3회였다."),
    ("blank", ""),
    ("sub", "3.2 학습 데이터 정확도와 하드웨어 정확도"),
    ("body", "학습 데이터의 장면을 하나씩 입력해 예측하게 하면 네 표현 모두 잡을 위치를 약 1cm 오차로 맞혔다. 명령한 관절 각도와 "
             "실제 각도의 차이는 0.2초의 지연을 빼면 1° 이내로, 원격조작으로 학습 데이터를 모을 때와 같은 수준이었다. 즉 네 표현은 "
             "한 번의 예측과 명령 수행에서 같은 정확도를 보였으며, 표현 간 성능 차이는 모델의 출력이 로봇을 움직이고 그 결과가 다시 "
             "입력이 되는 반복 실행 과정에서 나타났다."),
    ("blank", ""),
    ("sub", "3.3 실패 원인 분석"),
    ("body", "첫 번째 원인은 너무 작은 이동 지시이다. 바나나를 잡기 전 구간에서 한 청크 동안 지시한 그리퍼 끝 이동 거리의 중앙값은 "
             "절대 관절각이 5.8cm였으나 상대 관절각 0.8cm, 절대 EE 0.7cm, 상대 EE 1.0cm였고, 이 세 표현은 청크의 절반 이상에서 1cm "
             "미만을 지시했다. SO-101의 서보 모터는 목표와 현재 각도의 차이가 0.4° 미만일 때 0.2초 안에 움직인 비율이 23~44%였고, "
             "팔의 무게를 받치는 어깨와 팔꿈치는 차이가 1.3° 이상일 때 대부분 움직였다. 이 값은 모터의 위치 단위인 0.088°보다 크며, "
             "차이에 비례하는 힘으로 위치를 맞추는 서보 제어에서 작은 차이로 생기는 힘이 기어 마찰과 팔의 무게를 넘기 어렵기 때문으로 "
             "보인다. 상대 표현은 매 청크 현재 위치를 기준으로 이동을 지시하므로, 모터가 반응하지 "
             "않으면 같은 자리에서 다시 작은 이동을 지시하며 제자리에 머문다. 반면 절대 표현은 같은 목표를 계속 지시하므로 목표와 현재 "
             "위치의 차이가 쌓여 결국 모터가 움직인다. 그림 1(b)에서 1cm 미만의 지시는 상대 관절각, 절대 EE, 상대 EE에서 지시한 "
             "거리의 17~44%만 실제로 움직였으나 절대 관절각에서는 51~65%가 움직였으며, 이는 절대 목표가 유지되는 효과를 보여 준다. 처음 자세에서 한 청크 안에 3cm 이상 움직일 확률도 절대 관절각이 28%로 가장 높고 상대 관절각이 9%로 가장 "
             "낮았다."),
    ("body", "두 번째 원인은 모델이 그리퍼와 손목 명령을 현재 관절 값에 크게 의존하여 정한다는 점이다. 시연 데이터에서는 바로 다음의 "
             "그리퍼와 손목 값이 현재 값과 거의 같아, 현재 값을 따라 하기만 해도 학습 오차가 줄어든다. 모방학습에서 인과 혼동[10]이라 "
             "부르는 현상이다. 이를 확인하기 위해 바나나에 다가가기 전에 그리퍼를 닫은 28회의 시행에서, 닫기 명령이 나온 순간의 영상은 "
             "그대로 두고 그리퍼 값만 시연 시작 때처럼 열린 값으로 바꾸어 모델을 다시 실행했다. 원래 입력에서는 96%의 경우 다시 닫기 "
             "명령이 나왔지만, 그리퍼 값만 바꾸자 한 번도 나오지 않았다. 화면이 같은 상태에서 결정이 바뀌었으므로, 모델은 "
             "그리퍼가 이미 거의 닫혀 있다는 값을 근거로 닫기를 결정했다. 반면 바나나 바로 위에서 잡은 순간에는 같은 변경에도 모두 닫기 "
             "명령이 나와, 물체가 가까우면 영상을 근거로 판단했다. 손목도 마찬가지로, 손목 각도 값만 30° 바꾸어 넣으면 손목 명령도 "
             "25~31° 따라 바뀌었다. 즉 모델은 현재 손목 각도를 유지하는 방향으로 손목 명령을 "
             "출력했다. 상대 관절각 표현은 잡기 직전 3초 동안 1° 미만의 작은 손목 명령을 반복하여 손목을 세밀하게 조정하는 움직임을 보였으며, "
             "이 기간의 실제 회전량은 약 3°였다."),
]
CONCL = [
    "본 연구에서는 저비용 로봇 SO-101에서 네 가지 행동 표현에 따른 SmolVLA의 파인튜닝 성능을 비교했다. 학습 데이터에 대한 예측 "
    "정확도는 네 표현이 비슷했으나, 실제 로봇에서는 절대 관절각 표현이 가장 많이 바나나를 잡았다. 상대 표현과 EE 표현은 모터가 "
    "반응하지 않을 만큼 작은 이동을 반복 지시하여 로봇이 멈추었고, 네 표현 모두 그리퍼와 손목 명령을 현재 관절 값에 의존했다.",
    "따라서 저가 서보 모터를 쓰는 로봇에 상대 표현을 적용할 때는 모터가 반응할 수 있는 최소 이동량을 보장하거나 절대 목표를 유지하는 "
    "보완이 필요하다. 다만 표현마다 한 번의 학습과 20회의 평가로 얻은 결과이므로, 향후 평가 횟수를 늘리고 학습 시 관절 상태 입력에 "
    "잡음을 더하거나 다양한 초기 자세의 시연을 포함하여 관절 상태 의존을 줄이는 방법을 함께 검증할 예정이다.",
]
CAPTION = "그림 1. 행동 표현별 평가 결과와 지시한 이동 거리별 실제 이동 비율"

# ---------------- title block (inside the 1x1 table) ----------------
TITLE = {
    1: [(26, "저비용 로봇 SO-101에서 행동 표현에 따른 ")],
    2: [(26, "VLA 파인튜닝 성능 비교")],
    3: [(27, "강경빈"), (24, "1"), (27, " · 박현준"), (24, "1,*"), (27, "")],
    5: [(12, "Comparing VLA Fine-Tuning Performance across Action Representations ")],
    6: [(12, "on the Low-Cost SO-101 Robot")],
    7: [(13, "Kyeong-bin Kang"), (29, "1"), (13, " · Hyun-jun Park"), (29, "1,*")],
    9: [(14, "E-mail : kangkb7701@naver.com / hyunjun@kmou.ac.kr")],
    14: [(10, ABS_KO)],
    19: [(32, ABS_EN)],
    22: [(33, "Robot Manipulation, Vision-Language-Action, Action Representation, Fine-tuning, SO-101")],
}

# ---------------- body ----------------
T_INTRO, T_BODY, T_SUB, T_H = open_tag(24), open_tag(38), open_tag(37), open_tag(35)
T_H4 = open_tag(61)
BLANK = para(open_tag(36), [(10, "")])
GAP_A = raw(33, 35)        # two empty paragraphs before section II (original spacing)
GAP_B = raw(45, 47)        # empty paragraphs before section III
GAP_C = raw(59, 61)        # empty paragraphs before the conclusion
AFTER_H4 = raw(62, 63)
FIGURE = raw(30, 33).replace("그림 1. 시스템 아키텍처", CAPTION)


intro, sec2, sec3, concl = INTRO, SEC2, SEC3, CONCL


def items(lst):
    out = []
    for kind, text in lst:
        if kind == "figure":
            out.append(FIGURE)
            continue
        out.append(BLANK if kind == "blank" else para(T_SUB if kind == "sub" else T_BODY, [(10, text)]))
    return "".join(out)


body = ("".join(para(T_INTRO, [(28, t)]) for t in intro) + GAP_A
        + para(T_H, [(16, "Ⅱ. 실험 방법")]) + BLANK + items(sec2) + GAP_B
        + para(T_H, [(16, "Ⅲ. 실험 결과 및 분석")]) + BLANK + items(sec3) + GAP_C
        + para(T_H4, [(27, "Ⅳ. 결  론")]) + AFTER_H4 + "".join(para(T_BODY, [(10, t)]) for t in concl))

REFS = [
    [(17, 'K. Black et al., "π0: A Vision-Language-Action Flow Model for General Robot Control," arXiv:2410.24164, 2024.')],
    [(17, 'Physical Intelligence et al., "π0.5: a Vision-Language-Action Model with Open-World Generalization," arXiv:2504.16054, 2025.')],
    [(17, 'M. Shukor et al., "SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics," arXiv:2506.01844, 2025.')],
    [(17, 'A. Chen, Y. Yang, Z. Zhu, K. Xu, Z. Zhou, R. Xiong, and Y. Wang, "Toward Embodiment Equivariant Vision-Language-Action Policy," arXiv:2509.14630, 2025.')],
    [(17, 'W. Wu et al., "From Foundation to Application: Improving VLA Models in Practice," arXiv:2607.06403, 2026.')],
    [(17, 'T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware," in'),
     (37, " Proc. Robotics: Sci. Syst. (RSS)"), (17, ", 2023.")],
    [(17, 'TheRobotStudio and Hugging Face, "SO-ARM100: Standard Open Arm 100," GitHub, 2025. [Online]. Available: https://github.com/TheRobotStudio/SO-ARM100')],
    [(17, 'Y. Zhou, C. Barnes, J. Lu, J. Yang, and H. Li, "On the Continuity of Rotation Representations in Neural Networks," in'),
     (37, " Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)"), (17, ", pp. 5745-5753, 2019.")],

    [(17, 'R. Cadene et al., "LeRobot: An Open-Source Library for End-to-End Robot Learning," arXiv:2602.22818, 2026.')],
    [(17, 'P. de Haan, D. Jayaraman, and S. Levine, "Causal Confusion in Imitation Learning," in'),
     (37, " Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)"), (17, ", 2019.")],]
refs = "".join(para(open_tag(73), r) for r in REFS)

# assemble: [0..23] title block with edited paragraphs, new body, [65..72] ack + reference heading, new references
out, pos = [], 0
for i in sorted(TITLE):
    out.append(s[pos:P[i]]); out.append(para(open_tag(i), TITLE[i])); pos = own_end(i)
out.append(s[pos:P[24]]); out.append(body); out.append(s[P[65]:P[73]]); out.append(refs); out.append("</hs:sec>")
new = "".join(out)
OLD_ACK = "본 과제(결과물)는 2026년도 교육부 및 부산시의 재원으로 부산RISE혁신원의 지원을 받아 수행된 지역혁신중심 대학지원체계(RISE)의 결과입니다.(2026-RISE-02-002-001)"
NEW_ACK = ("This research was supported by the Commercialization Promotion Agency for R&D Outcomes(COMPA) funded by the Ministry "
           "of Science and ICT(MSIT).(RS-2025-25416427, University Technology Management Promotion Project(TLO Innovation "
           "Type)_NATIONAL KOREA MARITIME & OCEAN UNIVERSITY)")
assert new.count(OLD_ACK) == 1
new = new.replace(OLD_ACK, escape(NEW_ACK))
new = re.sub(r"<hp:linesegarray>.*?</hp:linesegarray>", "", new, flags=re.S)
xml.dom.minidom.parseString(new.encode("utf-8"))        # well-formedness check

prv = re.sub(r"<[^>]+>", "", re.sub(r"</hp:p>", "\n", new))
with zipfile.ZipFile(DST, "w") as zout:
    for info in zin.infolist():
        data = zin.read(info.filename)
        if info.filename == "Contents/section0.xml":
            data = new.encode("utf-8")
        elif info.filename == "Contents/header.xml":
            h = data.decode("utf-8")
            for pid in ("2", "20"):
                m = re.search(r'<hh:paraPr id="%s".*?</hh:paraPr>' % pid, h, flags=re.S)
                h = h[:m.start()] + m.group(0).replace('keepWithNext="0"', 'keepWithNext="1"') + h[m.end():]
            data = h.encode("utf-8")
        elif info.filename == "BinData/image1.png":
            data = FIG.read_bytes()
        elif info.filename == "Preview/PrvText.txt":
            data = "\n".join(l for l in prv.split("\n") if l.strip())[:1000].encode(zin.read(info.filename)[:2] == b"\xff\xfe" and "utf-16" or "utf-8")
        zout.writestr(info, data, compress_type=zipfile.ZIP_STORED if info.filename == "mimetype" else zipfile.ZIP_DEFLATED)


def text_of(x):
    return "".join(re.findall(r"<hp:t>([^<]*)</hp:t>", x))


print("written:", DST)
print("body chars:", len(text_of(body)), "| abstract ko:", len(text_of("".join(para('', TITLE[14])))),
      "| en:", len(TITLE[19][0][1]), "| refs:", len(text_of(refs)))
