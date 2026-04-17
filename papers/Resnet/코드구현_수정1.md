## 5. 코드 분석 - 논문이 코드로 어떻게 구현되는가

구현 코드: [resnet-implementation](https://github.com/9dadada/resnet-implementation)

### 5.1 전체 구조

코드는 2개의 Class로 구성된다.

```
ResNet18 (전체 모델)
  └── BasicBlock (기본 블록) × 8개
```

| Class | 역할 |
|---|---|
| BasicBlock | 논문의 Residual Block 하나를 구현. conv 2개 + shortcut |
| ResNet18 | BasicBlock을 8개 쌓아서 전체 모델을 구성 |

ResNet-18에서 18은 레이어 수다. BasicBlock 8개 × conv 2개 = 16개 conv + 맨 앞 conv1 + 맨 뒤 fc = 18층.

---

### 5.2 BasicBlock — 논문의 핵심이 담긴 블록

논문의 핵심 아이디어인 **잔차 학습**과 **shortcut connection**이 이 블록 안에 구현되어 있다.

BasicBlock에는 두 개의 함수가 있다:
- `__init__`: 블록에서 사용할 부품(conv, bn, relu, shortcut)을 정의
- `forward`: 정의한 부품을 순서대로 실행

#### 5.2.1 `__init__` — 부품 정의

__init__ 안에는 크게 **두 경로**의 부품이 정의되어 있다:

```
┌─ 메인 경로 (F(x) 학습) ─────────────┐
│  conv1 → bn1 → relu → conv2 → bn2   │
└──────────────────────────────────────┘

┌─ 지름길 경로 (x 보존) ──────────────┐
│  shortcut (빈 통로 or 1x1 conv+bn)  │
└──────────────────────────────────────┘
```


```python
def __init__(self, in_channels, out_channels, stride=1):
    super(BasicBlock, self).__init__()

    # 메인 경로 (F(x)를 학습하는 부분)
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                           stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_channels)

    # 지름길 경로 (x를 보존하는 부분)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
```



각 부품이 하는 일:

**conv1 (첫 번째 컨볼루션)**
- 채널 수를 바꾸고(in → out), stride=2일 때 특징맵 크기를 절반으로 줄인다
- 블록에서 크기 변환을 담당하는 핵심 부분이다

**conv2 (두 번째 컨볼루션)**
- conv1에서 이미 크기를 바꿨으므로, 같은 크기에서 한 번 더 특징을 뽑는다
- 그래서 입출력 채널이 같고 stride도 항상 1이다

| | conv1 | conv2 |
|---|---|---|
| 입력 채널 | in_channels | out_channels |
| 출력 채널 | out_channels | out_channels |
| stride | 1 또는 2 | 항상 1 |
| 역할 | 채널 수 변환 + 크기 축소 | 같은 크기에서 특징 추출 |

**bn1, bn2 (배치 정규화)**
- 컨볼루션을 통과한 값들은 레이어를 거칠수록 점점 커지거나 작아질 수 있다
- 배치 정규화는 이 값들을 평균 0, 표준편차 1로 맞춰서 학습을 안정시킨다
- conv 하나당 bn 하나가 붙는다. conv1 → bn1, conv2 → bn2
- bias=False인 이유: bn이 평균을 빼는 과정에서 bias가 사라지므로 의미가 없다

**relu (활성화 함수)**
- `relu(x) = max(0, x)` — 음수는 0으로, 양수는 그대로 통과
- 중요하지 않은 특징은 꺼버리고, 중요한 특징만 살리는 역할

**shortcut (지름길)** — 논문의 shortcut connection 구현
- 크기가 같을 때: `nn.Sequential()` 빈 통로. 입력 x가 그대로 통과
- 크기가 다를 때: 1x1 conv로 채널 수와 크기를 맞춤

1x1 conv가 필요한 이유: skip connection은 `F(x) + x`로 더하기 연산이다. 더하려면 크기가 같아야 한다.

```
F(x) = 128채널, 16x16
x    = 64채널, 32x32    ← 크기가 다르면 더할 수 없음

→ 1x1 conv(stride=2)로 x를 128채널, 16x16으로 변환하면 더할 수 있음
```

1x1 conv는 주변 픽셀은 보지 않고 채널 수만 변환하는 가장 가벼운 컨볼루션이다. shortcut은 원본을 최대한 그대로 전달하는 게 목적이므로 무거운 3x3 대신 1x1을 쓴다.

#### 5.2.2 `forward` — 논문의 핵심 수식 실행

__init__에서 부품을 정의했으면, forward에서 실제로 순서대로 실행한다.
논문의 핵심 수식 `y = F(x) + x`가 여기서 구현된다.

```python
def forward(self, x):
    identity = self.shortcut(x)       # x를 보존 (지름길)

    out = self.conv1(x)               # ─┐
    out = self.bn1(out)               #  │ F(x): 잔차를 학습하는 메인 경로
    out = self.relu(out)              #  │
    out = self.conv2(out)             #  │
    out = self.bn2(out)               # ─┘

    out += identity                   # F(x) + x ← 논문의 핵심 (잔차 학습)
    out = self.relu(out)
    return out
```

흐름을 그림으로 보면:

```
입력 x
  │
  ├──────────────────┐
  │                  │ identity = shortcut(x)
  ▼                  │
conv1 (3x3)          │
bn1                  │
relu                 │
conv2 (3x3)          │
bn2                  │
  │                  │
  + ←────────────────┘  F(x) + x
  │
relu
  ▼
출력
```

`out += identity` 이 한 줄이 논문의 핵심이다. 이게 없으면 평범한 plain network가 된다.

---

### 5.3 ResNet18 — BasicBlock을 쌓아서 전체 모델 구성

ResNet18에도 `__init__`과 `forward`가 있다.

#### 5.3.1 `__init__` — 전체 모델 구조 정의

```python
def __init__(self, num_classes=1000):
    super(ResNet18, self).__init__()

    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)

    self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)     # 32x32 유지
    self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)    # 32→16
    self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)   # 16→8
    self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)   # 8→4

    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                    # 4x4→1x1
    self.fc = nn.Linear(512, num_classes)                           # 분류
```

전체 흐름:

```
이미지 (3채널, 32x32)
  │
  ▼
conv1 + bn1 + relu          → 64채널, 32x32 (입구)
  │
  ▼
layer1: BasicBlock × 2      → 64채널, 32x32
layer2: BasicBlock × 2      → 128채널, 16x16
layer3: BasicBlock × 2      → 256채널, 8x8
layer4: BasicBlock × 2      → 512채널, 4x4
  │
  ▼
avgpool                      → 512채널, 1x1 (전체 평균)
fc                           → 10개 클래스 확률 (CIFAR-10)
```

논문의 설계 규칙 두 가지를 따른다:
1. 같은 크기의 특징맵에서는 같은 수의 필터 사용
2. 특징맵 크기가 반으로 줄면 필터 수를 2배로 (64 → 128 → 256 → 512)

#### 5.3.2 `_make_layer` — BasicBlock을 묶어주는 함수

```python
def _make_layer(self, in_channels, out_channels, blocks, stride):
    layers = []
    layers.append(BasicBlock(in_channels, out_channels, stride))
    for _ in range(1, blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)
```

예를 들어 `self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)`이면:
- 첫 번째 블록: 64채널 → 128채널, stride=2 (크기 절반으로 줄임)
- 두 번째 블록: 128채널 → 128채널, stride=1 (크기 유지)

첫 번째 블록에서만 크기를 바꾸고, 나머지는 같은 크기에서 특징을 더 뽑는다.

#### 5.3.3 `forward` — 이미지가 모델을 통과하는 순서

```python
def forward(self, x):
    x = self.conv1(x)      # 입구
    x = self.bn1(x)
    x = self.relu(x)

    x = self.layer1(x)     # BasicBlock × 2
    x = self.layer2(x)     # BasicBlock × 2
    x = self.layer3(x)     # BasicBlock × 2
    x = self.layer4(x)     # BasicBlock × 2

    x = self.avgpool(x)    # 전체 평균
    x = x.view(x.size(0), -1)  # 1차원으로 펼침
    x = self.fc(x)         # 분류
    return x
```

---

### 5.4 논문의 ResNet 전체 모델 구조 (논문 Table 1)

모든 ResNet은 같은 뼈대를 공유한다. 다른 건 **layer1~4 안에 블록을 몇 개 쌓느냐, 어떤 블록을 쓰느냐**뿐이다.

```
입력 (224x224)
  → conv1: 7x7, 64채널, stride 2     → 112x112
  → maxpool: 3x3, stride 2           → 56x56
  → layer1 (conv2_x)                 → 56x56
  → layer2 (conv3_x)                 → 28x28
  → layer3 (conv4_x)                 → 14x14
  → layer4 (conv5_x)                 → 7x7
  → avgpool                          → 1x1
  → fc → 1000 클래스
```

#### 블록 타입 2가지

**BasicBlock** (18, 34층용) — conv 2개 = 2층
```
3x3 conv → BN → ReLU → 3x3 conv → BN → (+shortcut) → ReLU
```

**Bottleneck** (50, 101, 152층용) — conv 3개 = 3층
```
1x1 conv → BN → ReLU → 3x3 conv → BN → ReLU → 1x1 conv → BN → (+shortcut) → ReLU
```

Bottleneck은 1x1로 채널을 줄이고 → 3x3으로 특징 뽑고 → 1x1로 채널을 다시 늘린다. 연산량을 줄이면서 더 깊게 쌓을 수 있는 구조다.

#### 모델별 블록 구성

| 구간 | 출력 크기 | ResNet-18 | ResNet-34 | ResNet-50 | ResNet-101 | ResNet-152 |
|---|---|---|---|---|---|---|
| conv1 | 112x112 | 7x7, 64, stride 2 | ← 동일 | ← 동일 | ← 동일 | ← 동일 |
| maxpool | 56x56 | 3x3, stride 2 | ← 동일 | ← 동일 | ← 동일 | ← 동일 |
| layer1 | 56x56 | BasicBlock ×2 | BasicBlock ×3 | Bottleneck ×3 | Bottleneck ×3 | Bottleneck ×3 |
| layer2 | 28x28 | BasicBlock ×2 | BasicBlock ×4 | Bottleneck ×4 | Bottleneck ×4 | Bottleneck ×8 |
| layer3 | 14x14 | BasicBlock ×2 | BasicBlock ×6 | Bottleneck ×6 | Bottleneck ×23 | Bottleneck ×36 |
| layer4 | 7x7 | BasicBlock ×2 | BasicBlock ×3 | Bottleneck ×3 | Bottleneck ×3 | Bottleneck ×3 |
| | | avgpool → fc | ← 동일 | ← 동일 | ← 동일 | ← 동일 |

ResNet-18과 34는 BasicBlock을 쓰고, 50부터는 Bottleneck을 쓴다. 34와 50은 블록 수(3+4+6+3=16개)가 같지만, Bottleneck이 conv를 3개 쓰니까 층 수가 다르다.

---

## 6. PyTorch 공식 ResNet과의 비교

### 6.1 입력부 차이

| | 논문 (ImageNet) | PyTorch 공식 | 내 구현 (CIFAR-10) |
|---|---|---|---|
| conv1 | 7x7, stride 2 | 7x7, stride 2 | **3x3, stride 1** |
| maxpool | 3x3, stride 2 | 3x3, stride 2 | **없음** |
| 입력 크기 | 224x224 | 224x224 | **32x32** |

PyTorch 공식 코드는 논문의 ImageNet 설정을 그대로 따른다. 내 구현은 CIFAR-10(32x32)에 맞게 바꿨다. 32x32 이미지에 7x7 conv + maxpool을 적용하면 특징맵이 8x8로 줄어들어 정보 손실이 크기 때문이다.

### 6.2 구조적 차이

| | PyTorch 공식 | 내 구현 |
|---|---|---|
| Shortcut 처리 | downsample 파라미터로 외부에서 주입 | 블록 내부에서 자체 판단 |
| expansion | BasicBlock/Bottleneck 구분하여 채널 확장 지원 | BasicBlock만 사용 (expansion 없음) |
| 가중치 초기화 | Kaiming normal 명시적 초기화 | PyTorch 기본값 사용 |
| Bottleneck 블록 | 있음 (ResNet-50/101/152용) | 없음 (ResNet-18만 구현) |
| dilation, groups | 지원 (ResNeXt 등 변형 대응) | 미지원 |

PyTorch 공식 코드는 ResNet-18부터 152까지, ResNeXt, Wide ResNet 등 다양한 변형을 하나의 클래스로 처리하는 범용 구현이다. 내 구현은 ResNet-18/CIFAR-10에 집중한 최소 구현이다.

### 6.3 비교 실험

같은 조건(CIFAR-10, 동일 하이퍼파라미터)에서 PyTorch 공식 ResNet18과 내 구현을 학습한 비교 실험은 [experiments/](https://github.com/9dadada/resnet-implementation/tree/main/experiments) 에서 확인할 수 있다.
