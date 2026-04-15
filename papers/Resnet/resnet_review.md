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

## 마무리

잔차학습이라는 아이디어가 인상 깊었다. 기존에는 정답을 직접 학습했다면, ResNet은 "얼마나 틀렸는지"만 학습하는 발상의 전환이다. 단순히 x를 더해주는 것만으로 152층까지 학습이 가능해졌다는 게 놀랍다.
