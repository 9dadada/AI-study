# ResNet Paper Review

> 논문 원본: [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
> 구현 코드: [resnet-implementation](https://github.com/9dadada/resnet-implementation)

## 왜 이 논문을 선택했는가

AlexNet에서 레이어가 8층으로 제한되는 문제가 있었다. 더 깊은 레이어를 학습할 수 있는 모델이 궁금해서 선택했다.

## 1. 논문 개요

| 항목 | 내용 |
|------|------|
| 논문 제목 | Deep Residual Learning for Image Recognition |
| 저자 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun |
| 발표 | CVPR 2016 |
| 데이터셋 | ImageNet, CIFAR-10, COCO |

더 깊은 네트워크의 학습을 용이하게 하기 위해 residual learning 프레임워크를 제시한 논문이다. ImageNet에서 최대 152개 층의 잔차 네트워크를 평가했는데, VGG 네트워크보다 8배 더 깊으면서도 복잡도는 더 낮다.

---

## 2. 문제 제기 - 레이어를 깊게 쌓으면 성능이 좋아질까?

그렇지 않다. 네트워크 깊이가 증가하면 정확도가 포화 상태에 이르렀다가 급격히 저하된다.

중요한 점은 이 열화(degradation)가 과적합과 다르다는 것이다.
- **열화**: 훈련 오류 자체가 높음 (학습이 안 됨)
- **과적합**: 훈련 오류는 낮지만 테스트 오류가 높음 (외워버림)

레이어가 너무 많아지면 최적화(optimization) 자체가 어려워지기 때문에 발생하는 문제다.

ResNet은 이 열화 문제를 해결하기 위해 심층 잔차 학습(deep residual learning) 프레임워크를 도입했다.

---

## 3. 핵심 기술

### 3.1 잔차 학습 (Residual Learning)

기존 방식은 H(x)를 직접 학습한다. "정답을 처음부터 맞혀라"라는 방식이다. 네트워크가 깊어질수록 이 직접 학습이 어려워진다.

ResNet은 발상을 바꿨다. "현재 답에서 뭘 고쳐야 하는지만 찾아라"

- 기존: H(x)를 직접 학습 → 깊어질수록 어려움
- ResNet: F(x) = H(x) - x (잔차)만 학습 → 출력은 F(x) + x

### 3.2 숏컷 연결 (Shortcut Connection)

입력 x를 레이어를 건너뛰어 출력에 그대로 더해주는 연결이다.

중요한 점은 숏컷 연결이 파라미터를 전혀 추가하지 않는다는 것이다. 그냥 x를 그대로 더해주기만 하는 거라서, Plain Network와 ResNet의 연산량이 완전히 동일하다.

### VGG-19, Plain-34, ResNet-34 비교

| | VGG-19 | Plain-34 | ResNet-34 |
|---|---|---|---|
| 층 수 | 19층 | 34층 | 34층 |
| 연산량 | 196억 | 36억 | 36억 |
| 숏컷 | 없음 | 없음 | 있음 |
| 열화 문제 | — | 발생 | 해결 |

---

## 4. AlexNet과의 비교

| | AlexNet | ResNet |
|---|---|---|
| 깊이 | 8층 (더 깊게 쌓으면 기울기 소실) | 152층까지 가능 (skip connection으로 해결) |
| 정규화 | LRN (효과 미미) | BatchNorm |
| 활성화 함수 | ReLU 도입 | ReLU 사용 (AlexNet이 열어준 길) |
| FC 레이어 | 파라미터 대부분이 FC에 집중 | Global Average Pooling으로 FC 축소 |

---

## 5. 코드 분석 - 논문이 코드로 어떻게 구현되는가

구현 코드: [resnet-implementation](https://github.com/9dadada/resnet-implementation)

### 5.1 Residual Block 구현 (논문 Figure 2 → BasicBlock)

논문의 핵심 수식 `y = F(x) + x`가 코드에서는 이렇게 구현된다:

```python
def forward(self, x):
    identity = self.shortcut(x)       # x를 보존 (지름길)

    out = self.conv1(x)               # ─┐
    out = self.bn1(out)               #  │ F(x): 잔차를 학습하는 메인 경로
    out = self.relu(out)              #  │
    out = self.conv2(out)             #  │
    out = self.bn2(out)               # ─┘

    out += identity                   # F(x) + x ← 논문의 핵심
    out = self.relu(out)              # 덧셈 후 활성화 (논문 Fig.2의 σ(y))
    return out
```

- `out += identity`가 skip connection이다. 이 한 줄이 없으면 평범한 plain network가 된다.
- Conv-BN-ReLU 순서는 논문 Section 3.4 ("BN right after each convolution and before activation")와 동일하다.
- `bias=False`는 BN이 shift를 대신하기 때문이다. 논문에서도 "biases are omitted"이라고 했다.

### 5.2 Shortcut Connection 구현 (논문 Eqn.1, Eqn.2)

입력과 출력의 차원이 같으면 x를 그대로 더하고(Eqn.1), 다르면 1x1 conv로 맞춘다(Eqn.2):

```python
self.shortcut = nn.Sequential()                    # 기본: identity (Eqn.1)
if stride != 1 or in_channels != out_channels:     # 차원이 다를 때
    self.shortcut = nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=1, stride=stride),    # 1x1 conv로 차원 맞춤 (Eqn.2)
        nn.BatchNorm2d(out_channels),
    )
```

논문의 Option B 방식이다: "projection shortcuts are used for increasing dimensions, and other shortcuts are identity."

### 5.3 전체 구조 (논문 Table 1 → ResNet18)

논문 Table 1의 18-layer 구성을 CIFAR-10에 맞게 구현했다:

```python
self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)  # 입구
self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)         # 32x32 유지
self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)        # 32→16
self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)       # 16→8
self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)       # 8→4
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                        # Global Average Pooling
self.fc = nn.Linear(512, num_classes)                               # 분류
```

논문의 두 가지 설계 규칙을 따른다:
1. 같은 크기의 특징맵에서는 같은 수의 필터 사용
2. 특징맵 크기가 반으로 줄면 필터 수를 2배로 (64 → 128 → 256 → 512)

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

---

## 마무리

잔차학습이라는 아이디어가 인상 깊었다. 기존에는 정답을 직접 학습했다면, ResNet은 "얼마나 틀렸는지"만 학습하는 발상의 전환이다. 단순히 x를 더해주는 것만으로 152층까지 학습이 가능해졌다는 게 놀랍다.
