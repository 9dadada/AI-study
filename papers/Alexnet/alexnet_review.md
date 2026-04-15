# AlexNet Paper Review

> 논문 원본: [ImageNet Classification with Deep Convolutional Neural Networks (NeurIPS 2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

## 1. 논문 개요

<table>
  <tr>
    <th>항목</th>
    <th>내용</th>
  </tr>
  <tr>
    <td>논문 제목</td>
    <td>ImageNet Classification with Deep Convolutional Neural Networks</td>
  </tr>
  <tr>
    <td>저자</td>
    <td>Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton</td>
  </tr>
  <tr>
    <td>발표</td>
    <td>NeurIPS 2012</td>
  </tr>
  <tr>
    <td>데이터셋</td>
    <td>ImageNet LSVRC-2010 / ILSVRC-2012</td>
  </tr>
  <tr>
    <td>총 파라미터 수</td>
    <td>약 6천만 개</td>
  </tr>
</table>

AlexNet은 딥러닝의 시대를 본격적으로 연 논문으로, ILSVRC 2012 대회에서 2위 대비 오류율을 약 10%p 이상 줄이며 압도적인 1위를 차지했다.


---
## 2. 아키텍처 구조

```
Input (224×224×3)
  → Conv1 → MaxPool → LRN
  → Conv2 → MaxPool → LRN
  → Conv3
  → Conv4
  → Conv5 → MaxPool
  → FC1 (4096) → Dropout
  → FC2 (4096) → Dropout
  → FC3 (1000) → Softmax
```

### 레이어별 상세

| 레이어 | 출력 크기 | 필터 수 | 커널 크기 | 특이사항 |
|--------|----------|---------|----------|---------|
| Input | 224×224×3 | — | — | RGB 이미지 |
| Conv1 | 55×55×96 | 96 | 11×11, s=4 | ReLU → MaxPool → LRN |
| Conv2 | 27×27×256 | 256 | 5×5, p=2 | ReLU → MaxPool → LRN |
| Conv3 | 13×13×384 | 384 | 3×3, p=1 | ReLU |
| Conv4 | 13×13×384 | 384 | 3×3, p=1 | ReLU |
| Conv5 | 13×13×256 | 256 | 3×3, p=1 | ReLU → MaxPool |
| FC1 | 4096 | — | — | ReLU, Dropout(0.5) |
| FC2 | 4096 | — | — | ReLU, Dropout(0.5) |
| FC3 | 1000 | — | — | Softmax |

> `s` = stride, `p` = padding

### GPU 분할 구조

속도 향상을 위해 모델을 **GPU 2개**에 절반씩 나눠 학습시켰다.  
Conv3, FC1~3에서만 두 GPU가 서로 통신하고, 나머지 레이어는 독립적으로 연산한다.

```
GPU 1: Conv1(48ch) → Conv2(128ch) → Conv3(192ch) ┐
                                                   ├→ Conv4(192ch) → Conv5(128ch) → FC
GPU 2: Conv1(48ch) → Conv2(128ch) → Conv3(192ch) ┘
```

---

## 3. 핵심 기술 혁신

### 3.1 ReLU 활성화 함수

기존 활성화 함수인 sigmoid, tanh를 버리고 **ReLU**를 사용했다.


### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

- 출력 범위: $(0, 1)$
- 기울기: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- 문제: $x$가 매우 크거나 작으면 기울기 → $0$ (기울기 소실)

---

### tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

- 출력 범위: $(-1, 1)$
- 기울기: $\tanh'(x) = 1 - \tanh^2(x)$
- 문제: sigmoid보다 낫지만 여전히 기울기 소실 발생

---

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

- 출력 범위: $[0, \infty)$
- 기울기: $\text{ReLU}'(x) = \begin{cases} 1 & (x > 0) \\ 0 & (x \leq 0) \end{cases}$
- 문제: 음수 입력 뉴런이 영구적으로 죽는 Dying ReLU

<br>
<br>

**발전 과정:** <br>
처음에 sigmoid와 tanh 함수를 쓴 것은 실제 뇌를 모사하기 위함이었다.
인간의 뉴런은 자극이 쌓이면 천천히 활성화되다가 포화되는데, sigmoid가 딱 그 모양이었다. 생물학적 타당성때문에 sigmoid를 선택한것이다.<br>
기술이 발전된 후에 기울기 소실이 문제가 되어 레이어가 늘어날수록 학습에 문제가 생겼다.

**장점:**
- 학습 속도가 수십 배 향상
- 기울기 소실(Vanishing Gradient) 문제 완화
- 계산이 단순해 GPU 연산에 유리

논문에서는 CIFAR-10 데이터셋 기준, ReLU가 tanh 대비 **6배 빠르게** 25% 오류율에 도달했다.

<br>

**그렇다면 ReLU는 문제가 없을까?:**<br>
그렇지 않다.<br>
학습 중에 어떤 뉴런의 입력이 항상 음수가 되면, 그 뉴런의 출력은 영원히 0이 된다. 기울기도 0이 되니 그 뉴런은 다시 업데이트 되지 않고 죽어버린다. sigmoid나 tanh는 기울기가 0인 구간이 없고 0에 수렴하기 때문에 이러한 문제가 없다. <br>
그래서 지금도 ReLU만 쓰지는 않고 다음과 같은 다양한 활성화 함수를 사용한다.

<br>

| 함수 | 수식 | 해결하려는 문제 |
|------|------|--------------|
| ReLU | `max(0, x)` | 기울기 소실 (Vanishing Gradient) |사용 |
| Leaky ReLU | `x > 0 ? x : 0.01x` | Dying ReLU (뉴런 영구 비활성화) |
| ELU | `x > 0 ? x : α(eˣ−1)` | 음수 구간 출력 허용, 평균 0 근접 |
| GELU | `x · Φ(x)` | 확률 기반 부드러운 활성화 | Transformer 표준 |



### 3.2 Overlapping Pooling

일반적인 풀링은 윈도우 크기와 스트라이드를 같게 설정하였는데,  
AlexNet은 스트라이드(2)보다 큰 커널(3×3)을 사용해 **겹치는 풀링**을 사용하였다.
<br>
겹치는 풀링에서는 경계 픽셀이 인접한 두 윈도우 모두에 참여한다. 그래서 중요한 특징이 살아남을 확률이 더 높아힌다. 또한 이웃한 윈도우들이 서로 겹치니 공간적인 연속성이 보존되어, 모델이 위치 변화에 덜 민감해진다. 이 효과가 과적합을 약간 억제한다.


| 방식 | 커널 | 스트라이드 | 겹침 |
|------|------|----------|------|
| 일반 풀링 | 2×2 | 2 | 없음 |
| Overlapping | 3×3 | 2 | 있음 |

- Top-1 오류율 약 0.4%p, Top-5 오류율 약 0.3%p 개선
- 과적합이 다소 줄어드는 효과

### 3.3 드롭아웃 (Dropout)

FC1, FC2에서 학습 시 **50% 확률로 뉴런을 무작위로 비활성화**한다.
<br>
과적합 방지를 위해 사용한다. 언제 어떤 뉴런이 꺼질지 모르니 각 뉴런이 혼자서도 유용한 특징을 학습해야 한다. 결과적으로 여러 가지 다른 구조의 네트워크를 동시에 학습하는 앙상블 효과가 생긴다.


```
학습 시: 각 뉴런이 50% 확률로 출력 0
추론 시: 모든 뉴런 사용, 출력에 0.5 곱함 (기댓값 보정)
```

**효과:**
- 뉴런 간 공동 적응(co-adaptation) 방지
- 앙상블 효과: 매번 다른 구조의 네트워크를 학습하는 것과 유사
- 과적합 대폭 감소

---

### 3.4 데이터 증강 (Data Augmentation)

기존 데이터를 변형해서 학습 데이터를 인위적으로 늘리는 기법

**왜 필요한가?**
- 딥러닝은 모델 파라미터가 많을수록 더 많은 데이터가 필요하다.
- 새로운 데이터를 수집하는 것은 비용이 크고 시간이 오래 걸린다.
- 그래서 기존 이미지를 조금씩 변형해서 데이터를 늘리는 방법을 쓴다. 

**방법 1: 랜덤 크롭 + 수평 뒤집기**
- 256×256 이미지에서 224×224 크기를 랜덤하게 잘라냄
- 좌우 뒤집기 적용
- 학습 데이터를 **2048배** 늘리는 효과

**방법 2: PCA 색상 변환**
- 학습 이미지 전체의 RGB 픽셀값에 PCA 적용
- 주성분 방향으로 랜덤한 크기의 색상 변화를 추가
- 조명 변화와 색상 불변성 학습에 효과적


---