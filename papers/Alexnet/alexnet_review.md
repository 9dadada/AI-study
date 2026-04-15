# AlexNet Paper Review

> 논문 원본: [ImageNet Classification with Deep Convolutional Neural Networks (NeurIPS 2012)](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)

## 왜 이 논문을 선택했는가

CNN에서 딥러닝으로 AI 학습의 트렌드가 넘어올 수 있었던 이유가 AlexNet의 알고리즘이다. AlexNet은 깊은 신경망에서 학습하는 방법을 처음 제시한 논문이기 때문에 첫 논문 스터디의 논문으로 선택하였다.

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

sigmoid와 tanh의 그래프 모양을 보면, 입력값이 커지거나 작아질수록 기울기가 점점 0에 수렴한다. 이 때문에 신경망이 깊어질수록 가중치 반영이 잘 안되는 **기울기 소실(Vanishing Gradient)** 문제가 발생했다. AlexNet의 핵심은 이 기울기 소실을 방지하기 위해 활성화 함수로 ReLU를 사용했다는 것이다.

$$\text{ReLU}(x) = \max(0, x)$$

- 출력 범위: $[0, \infty)$
- 기울기: $\text{ReLU}'(x) = \begin{cases} 1 & (x > 0) \\ 0 & (x \leq 0) \end{cases}$

**여기서 든 의문:** 
<br>
수식을 보면 ReLU가 sigmoid와 tanh보다 훨씬 간단하다. 그런데 왜 CNN 초기에 ReLU를 사용하지 않았을까? 이는 실제 인간의 뉴런 신경망을 모방한 함수로 sigmoid와 tanh가 제시되었기 때문이다. 인간의 뉴런은 자극이 쌓이면 천천히 활성화되다가 포화되는데, sigmoid가 딱 그 모양이었다. 생물학적 타당성 때문에 sigmoid를 선택한 것이다. 이후 기술이 발전하면서 기울기 소실이 문제가 되었고, 그제서야 단순하지만 효과적인 ReLU가 주목받게 되었다.

논문에서는 CIFAR-10 데이터셋 기준, ReLU가 tanh 대비 **6배 빠르게** 25% 오류율에 도달했다.

**그렇다면 ReLU는 문제가 없을까?**

그렇지 않다. 학습 중에 어떤 뉴런의 입력이 항상 음수가 되면, 그 뉴런의 출력은 영원히 0이 된다. 기울기도 0이 되니 그 뉴런은 다시 업데이트 되지 않고 죽어버린다(Dying ReLU). 그래서 지금도 ReLU만 쓰지는 않고 Leaky ReLU, GELU 등 다양한 활성화 함수가 사용되고 있다.

---


### 3.2 드롭아웃 (Dropout)

FC1, FC2에서 학습 시 **50% 확률로 뉴런을 무작위로 비활성화**한다.

과적합 방지를 위해 사용한다. 언제 어떤 뉴런이 꺼질지 모르니 각 뉴런이 혼자서도 유용한 특징을 학습해야 한다. 결과적으로 여러 가지 다른 구조의 네트워크를 동시에 학습하는 앙상블 효과가 생긴다.

```
학습 시: 각 뉴런이 50% 확률로 출력 0
추론 시: 모든 뉴런 사용, 출력에 0.5 곱함 (기댓값 보정)
```

직접 구현하면서 dropout 유무에 따라 학습 결과가 어떻게 달라지는지 실험해보고 싶다.

---

### 3.3 데이터 증강 (Data Augmentation)

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

## 4. AlexNet의 한계

1. **깊이의 한계** - 8개 레이어밖에 없어서 더 깊게 쌓으면 기울기 소실이 다시 발생한다. 이후 ResNet이 잔차 연결(skip connection)로 이 문제를 해결했다.
2. **파라미터가 너무 많음** - 약 6천만 개의 파라미터 중 대부분이 FC 레이어에 집중되어 메모리 비효율적이다.
3. **LRN의 효과 미미** - 논문에서는 LRN이 도움된다고 했지만, 이후 VGGNet 논문에서 성능 향상에 기여하지 않는다고 밝혀졌다. 현재는 BatchNorm으로 대체되었다.
4. **큰 커널 사이즈** - Conv1에서 11×11 커널을 사용해 연산량이 크다. 이후 VGGNet에서 3×3 커널을 여러 개 쌓는 것이 더 효율적이라는 것이 밝혀졌다.

---

## 마무리

이 논문을 읽으면서 느낀 점:

지금은 당연하게 쓰이는 ReLU, Dropout 같은 기법들이 2012년에는 혁신이었다는 게 인상 깊었다.
직접 구현하면서 dropout 유무에 따른 성능 차이를 실험해볼 예정이다.
다음 논문으로 ResNet을 읽을 예정인데, AlexNet의 한계를 어떻게 극복했는지 비교해보고 싶다.
