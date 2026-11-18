# OpenVLA 프로젝트 구조 및 설계 분석

AI 공부를 하는 대학생들이 참고하기 좋은 OpenVLA(Vision-Language-Action) 라이브러리의 구조와 설계 원칙을 정리한 문서입니다.

## 1. 전문적인 AI 라이브러리 구축 시 고려해야 할 핵심 요소

이 프로젝트처럼 확장 가능하고 유지보수가 쉬운 AI 라이브러리를 만들려면 다음 3가지를 특히 신경 써야 합니다.

*   **모듈화 (Modularity):** 모델 아키텍처, 데이터 로딩, 학습 전략을 완전히 분리해야 합니다. 예를 들어, Vision Encoder를 CLIP에서 SigLIP으로 바꿔도 다른 코드는 수정할 필요가 없도록 설계해야 합니다.
*   **레지스트리 패턴 (Registry Pattern):** 다양한 모델(Llama2, Mistral 등)과 데이터셋을 문자열 ID로 관리하여, 코드 수정 없이 설정값만으로 실험을 바꿀 수 있게 합니다. (`prismatic/models/materialize.py` 참고)
*   **재현성 (Reproducibility):** `pyproject.toml`을 통한 환경 관리, 고정된 시드 설정, 그리고 학습/전처리 과정을 자동화한 스크립트(`scripts/`)를 제공해야 합니다.

## 2. OpenVLA 폴더 구성 및 역할

프로젝트는 크게 **핵심 로직(`prismatic/`)**, **실행 스크립트(`scripts/`)**, **실험 환경(`experiments/`)**으로 나뉩니다.

*   **`prismatic/` (Core Library):** 라이브러리의 심장부입니다.
    *   `models/`: Vision Backbone, LLM Backbone, 그리고 이들을 연결하는 VLM/VLA 모델이 정의되어 있습니다.
    *   `vla/`: 로봇 액션을 토큰화하거나 특수한 데이터셋(RLDS)을 처리하는 로직이 담겨 있습니다.
    *   `conf/`: 데이터셋과 모델의 하이퍼파라미터를 관리하는 설정 파일들이 모여 있습니다.
    *   `training/`: 분산 학습(DDP, FSDP) 등 실제 학습을 제어하는 전략이 구현되어 있습니다.
*   **`scripts/` & `vla-scripts/` (Entry Points):**
    *   모델 학습(`train.py`), 미세 조정(`finetune.py`), 가중치 변환(`convert_...`) 등 사용자가 직접 실행하는 파일들입니다.
*   **`experiments/` (Downstream Tasks):**
    *   실제 로봇 환경(Bridge, Libero 등)에서 모델을 평가하고 데이터를 분석하는 코드가 들어 있습니다.

---

## 3. MLP Projector 코드 분석

MLP Projector는 **시각 인코더(Vision Encoder)가 뽑아낸 특징을 언어 모델(LLM)이 이해할 수 있는 차원으로 변환**해 주는 다리 역할을 합니다.

### 코드 위치
`prismatic/util/nn_utils.py`

### 주요 구현 방식
OpenVLA는 실험의 복잡도에 따라 세 가지 타입의 Projector를 제공합니다.

1.  **LinearProjector:** 가장 단순한 형태로, 하나의 선형 레이어(`nn.Linear`)만 사용합니다.
2.  **MLPProjector (GELU-MLP):** 
    *   `Linear -> GELU 활성화 함수 -> Linear` 구조입니다.
    *   비선형성을 추가하여 시각-언어 공간 사이의 복잡한 매핑을 학습합니다.
3.  **FusedMLPProjector:**
    *   여러 시각 특징(예: DINOv2 + SigLIP)을 합쳐서 사용할 때 쓰는 더 깊은 MLP입니다.
    *   차원을 일시적으로 크게 키웠다가(`* 4`) 다시 줄이는 방식을 사용합니다.

### 코드 발췌 (`prismatic/util/nn_utils.py`)
```python
class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        # ... (생략)

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)
```

### 왜 중요한가요?
멀티모달 학습에서 Vision Encoder와 LLM은 이미 잘 학습된(Pre-trained) 모델을 가져다 쓰는 경우가 많습니다. 이때 두 거대 모델을 연결하는 **유일하게 학습 가능한 부분**이 바로 이 Projector입니다. 따라서 효율적인 변환을 위해 단순하지만 강력한 MLP 구조를 선택하는 것이 트렌드입니다.
