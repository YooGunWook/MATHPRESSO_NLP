# MATHPRESSO_NLP_Project

## Contribution of Project

- [김희진](https://github.com/gimmizz)
- [안영진](https://github.com/snoop2head)
- [김지환](https://github.com/sopogen)
- [유건욱](https://github.com/YooGunWook)
- [최정윤](https://github.com/yuna-102)

## 산학 주제

NLP 파트이며 수학 문제에 대한 데이터가 주어질 때 문제의 유형을 예측하는 Task입니다.

**Input Example.**

```
`x`에 대한 이차방정식 `x^2-6x+k+1=0`은 실근을 갖고, `x`에 대한 이차방정식 `kx^2-8x+5=0`은 서로 다른 두 허근을 갖도록 하는 자연수 `k`의 개수는?
```

**Output Example.**

`H1S1-05`

------

## Modeling To-do

- [ ] QTID 전체 기준 training: 중단원이 아니라 소단원으로 연결시키면 더 좋을 것 같다. 어떤 걸 예측하냐에 따라서 모델 파일을 따로따로 올리자.
  - [ ] 중단원 기준(qtid 일부)이 아니라 소단원 기준(qtid 전체) 592개 class를 갖고 multi-label classification하자. 나중에 소단원 qtid 라벨 7번째 글자까지 자르면 중단원이니까.
  - [x] 소단원 기준으로 stratified random sampling을 진행하는 게 좋을 것 같다.
  - [x] **Random Sampling Seed가 컴퓨터 별로 다르게 나오는 경우가 있다. Train File, Validation File을 각각 csv파일로 하나 정해놓고 표준화시키자.** 
- [ ] 각 실험 별 어떤 걸 바꿨는지 정리를 해야 할 것 같다.  **Slack에 표를 만들어서 공유할 수 있다면, Slack을 사용해보자.** 
  - Model Name
  - Preprocessing: white space 추가 등. Github commit url을 첨부
  - HyperParameter Tuning
  - Validation F1 score
  - Model에서 변화: Early Stopping 등
  - 몇 Epoch 돌렸는지
  - Colab Notebook URL
- [ ] 불균등한 Evaluation 데이터셋 예측 시 해결방법
  - [ ] Prediction할 때 weight을 곱해주는 거 어때요? training dataset이 게시물의 그림처럼 불균등하고, hidden dataset이 같은 분포를 따른다고 했죠… 즉 training dataset의 분포 비율 그대로를 weight로 곱해주면 더 정확하지 않을까요 ㅎㅎ. 카테고리가 37개 잖음? softmax function 들어가게 되면 tensor 안에 37개에 대한 예측값이 나옴. 그걸 갖고 있어야 weight을 prediction에다가 곱할 수 있음. 예측값을 저장해야 함.
  - [ ] [Training 할 때 loss를 계산할때 weight를 줘서 하는 방식으로 진행. cross entropy에 weight를 줘서 한번 해보자!](https://towardsdatascience.com/pytorch-tabular-multiclass-classification-9f8211a123ab) 별로 복잡하지 않고, 식 그대로 코드를 써서 넣으면 될 것 같다.
- [ ] Electra 경량화 모델을 사용하면서, 이 정도 성능이 나온다는 걸 보여주는 게 좋지 않을까?
- [ ] RNN, LSTM은 사이즈가 작은 모델이니까 이걸로 비교해보는 것도 나쁘지 않을 것 같다.
- [ ] Word2vec 혹은 Tfidf으로 만들어진 벡터를 Random forest로 사용할 수 있지 않을까?


## Training Dataset Preprocessing To-do

- [ ] **각 preprocessing version 별로 나눠서 모델 성능을 평가하는 게 좋겠다.** 깃헙 commit log를 첨부하면 무엇을 바꿨는지 파악하기 더 쉬울 듯
  - Roman Letters in english (alpha, beta...) -> ɑ, β
  - 영문 -> 한글
  - 영문 -> 특수문자
  - White Space
  - Data Augmentation
    * Python Regex으로 noise 발생
    * TFIDF or Word2vec으로 noise 발생
    * BERT으로 noise 발생
- [ ] **중의성을 해소하는 것에 목적을 둔 Preprocess module for Deep NLP 만들어보려고 합니다.** 
  - [ ] ()랑 ^의 중복처리는 아직임. 해야 한다. 이를 Evaluation dataset에 적용
  	- [ ] () : 점? 좌표? 순서쌍? 정규분포 N(0, 1)
- [ ] Pipe ( | ) : 정의역치역? 절댓값? 조건부확률?: 정의역, 치역 집합 조건 |, 조건부확률의 |, 절대값 || 다르게 처리해야 한다. -> 물론, '정의역', '치역'이라는 단어로 분리되긴 할 듯.
  - [ ] CSV에서 replace all해서 "pipe"로 바꾼 다음에 regex 처리 하겠음.
- [ ] bar은 기호 하나로 바꾸면 더 좋을 것 같음.
- [x] combination이나 permutation 등으로 용어를 바꿔준 부분들을 그럼 순열, 조합 등의 한글로 바꿔 주기로 함. 


## Evaluation Dataset Preprocessing To-do

- [ ] evaluation dataset을 기반으로 Preprocess module for Deep Learning NLP 만들기.
  - [ ] evaluation dataset: "다음 중 무슨 함수의 그래프인가?", "다음 중 ..." "다음 중 옳지 않은 걸 고르시오" 등 20자 밑의 질문들이 있음. -> Koelectra에서 이걸 빼던데, 어디서 추가해주는 거에용? 결국에 제출물에 추가를 해주긴 해야 하는데 Preprocessing에서 함수를 하나 만들어 놓자. 

## Training Dataset Size-up To-do
- [ ] Nlp Data Augmentation: 데이터셋 100개 미만 챕터: H1S2-01, H1S1-01, H1S1-06, HSTA-04, H1S1-03, HSTA-01, HSTA-02, H1S1-09, HSU1-11
  - [ ] Proportion 맞춰서 키우는 것
  
  - [ ] Even하게 데이터셋 키우는 것
  
  - [ ] **Training dataset만 augmentation을 해야 validation dataset에 같은 문제들이 섞이지 않을 것. 즉 모듈로 만들어야겠네.**
  
  - [x] Random Deletion이 수학 기호와 한글 사이의 경계를 날려버리기도 함 ㅠ ->
  
    ````
    `3+5` 의 답은?
    ````
  
    {A: "`3+5`"...} 로 놓고 dict에다가 저장. 특정 부분만 사라지는 문제는 일단 해결을 함. A에서 3+5라는 string으로 다시 복구하는 것
  
  - [x] [논문에 다르면 5000개 정도의 데이터에서 EDA로 20000개로 늘리는 게 최적이라고 함.  ](https://catsirup.github.io/ai/2020/04/21/nlp_data_argumentation.html)
- [ ] KoELECTRA에 수학 데이터셋을 학습시키면 좋지 않을까? 수학 기호들이 [UNK] 토큰으로 표시 될까봐 두렵다.
  - [ ] [Deep Mind's Mathematical Q&A Pairs](https://github.com/deepmind/mathematics_dataset)과의 수학 notation 통일. 라벨링은 진행하지 않기. 
  - [ ] [Deep mind dataset training 활용 사례](https://github.com/mandubian/pytorch_math_dataset)
  - [ ] [Deep mind 논문 2019](https://github.com/andrewschreiber/hs-math-nlp)
  - [ ] [Automatic Generation of Headlines for Online Math Questions](https://github.com/yuankepku/MathSum)

-------

## Done for Modeling
- [x] Koelectra나 bert에서 training 하는 과정에서 loss, accuracy등을 그래프로 그리기. Preprocessed Data나 NLP Augmented Data를 넣었을 때 성능 비교를 해야 한다. -> Loss를 list로 저장해서 matplotlib으로 그리면 된다. 굳이 tensorboard처럼 실시간으로 볼 필요는 없으니까.
- [x] **Train하고 Test를 Split해서 시험해봐야 할 것 같다.** Stratified Random Sampling해서 0.8:0.2로 나누면 될 듯.
- [x] [참고자료: Early Stopping for PyTorch](https://github.com/Bjarten/early-stopping-pytorch)
- [x] BERT 모델에서는 tokenized된 결과를 확인해볼 수 있는 방법이 있을까요?? math word 처리 부분만 하고, data_preprocessed.csv, data_josa_removed.csv 등 기존에 tokenized된 결과는 이용하지 않는 것 같은데, tfidf / word2vec에서는 잘 예측하지만 BERT에서만 엉뚱하게 예측하는 케이스를 어떻게 해석해야 하는지 궁금합니다!!
  - [x] -> kobert 모듈을 다운로드 받고, 다음과 같이 코드를 쓰면 됩니다.
  ```python
  from kobert_transformers import get_tokenizer
  tokenizer = get_tokenizer()
  tokenizer.tokenize("합집합 A∪B")
  ```
  - [x] 수학 기호들이 [UNK] 토큰으로 표시 되지는 않나?: ∉, ≠ 기호 등은 안 됨. 하지만 prediction에 크게 상관 없을 듯.
- [x] 개발환경(구글 드라이브 폴더) 통일
  - [x] [📂 우리 팀 KoElectra & BERT 코드](https://drive.google.com/drive/u/2/folders/1dKjz7dVv9NVpCnoGSyfzDd4V7zU4W4EI)
    * ID: ybigtavision@gmail.com
    * PW: ybigta!best!
  - [x] [Colab - Github Private Repo](https://colab.research.google.com/drive/1ZQxKM9Dt1fJPsp4DDkEtJx9Nl7DT0bWk#scrollTo=HCa-uo_kK_Nz)
  - [x] [Open in Colab 크롬 익스텐션 깔기.](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo?hl=en)
- [x] TFIDF, Word2vec은 버리자 ㅠㅠ
  - [x] 혼합하는 기준을 세우는 것도 시간인 듯.
  - [x] Old Folder로 옮기기 ㅠㅠ
  - [x] <>는 txt로 정리를 함.
  - [x] evaluation dataset: `</span>` 등 태그는 이미 제거를 함.
- [x] 혹시 valid set을 사용하시면 valid set를 예측한 label 을 꼭 올려주세요. valid set 중 어떤 문제를 못 맞췄는지 살펴보고 모델을 보완하기 위한 EDA 및 FE를 진행할 계획입니다.
  - [x] KoElectra: ㅠㅠ qplay_id: 챕터 형태가 더 좋을 듯 해요
  - [x] KoBERT (epoch 80)
  - [x] KoBERT (epoch 15)
  
- [x] Optimizer는 아니지만 제가 지환님이랑 early stopping 관해서 보고 있는데요, 지환님이 지금 early stopping 기준이 loss로 되어있는데 이 기준을 f1 score로 바꾸는게 좋을 것 같다고 하시는데 다들 어떻게 생각하시나요?? -> convention이 loss를 따르는 거라, loss로 하는 걸로 결정.


## Done for Preprocessing 

  - [x] `{::}`는 함수식 내 분수를 표현하는 기호. 없애는 것보다 따로 처리해주면 성능이 올라갈까?
  - [x] sin^`, `cos^`, `tan^` 를 따로 분류하면?
    - [x] ^은 제곱을 의미함.
  - [x] 곱하기 기호가 `x` 문자로 표현되는 경우가 있는 것 같다. (tan1˚xxtan2˚xxcdotsxxtan88˚xxtan89˚=c, Mxxm 등) -> `xx`로 분리되고 있음
    - [x] **xx와 *는 곱하기 notation인데, 이걸 통일시켜줘야 함.**
    - [x] cdots는 continous dots
  - [x] 아래첨자 _ 는 그냥 그대로 놔두기로 했음
    - [x] A_1 P_(n-1) 아래첨자 표기법도 Combination, Permutation 처럼 하나로 인식하면 좋을 것 같다. (A, _, 1 로 분리되어 인식되고 있음)
    - [x] log_ 쪽도 포함해보기 밑이 있는 log 방정식이 다른 단원으로 분류가 되기 때문에, 이걸 처리해줘야 한다.
  - [x] `oo` infinity 기호 -> `∞`로 표시하기
  - [x] `sum`도 math_terms에 -> ex) `lim_(n->oo)1/nsum_(k=1)^(n-1)1/S_k` -> 'lim', '_', '(', 'n', '->', 'oo', ')', '1', '/', 'ns', 'um', '_', '(', 'k', '=', '1', ')', '^', '(', 'n', '-1', ')', '1', '/', 'S', '_', 'k' 로 인식되고 있음. `ns와 um으로 분리되어 인식되고 있음`
  - [x] `squareABCD`가 하나로 인식되고 있음(data 개수 2개) -> 큰 상관은 없을 듯하다
  - [x] 의미중복 ABCD: 사각형? 정사면체? 
  - [x] random sampling으로 분리해낸 valid set에서 확인해야 할 것들. 흔하게 쓰이지만, BERT가 이걸 잘 넘기는지, 못 넘기는지 확인 필요.

--------

## Modeling So Far

* [📂 우리 팀 KoElectra & BERT 코드](https://drive.google.com/drive/u/2/folders/1dKjz7dVv9NVpCnoGSyfzDd4V7zU4W4EI)
* KoELECTRA 참고 자료들
  * [Koelectra base-v3 epoch 20으로 돌려서 제출했는데 결과가 좋게 꽤 좋게 나왔음. 이걸 기본적으로 사용하자.](https://github.com/monologg/KoELECTRA)
  * [2주 간의 KoELECTRA 개발기 - 1부](https://monologg.kr/2020/05/02/koelectra-part1/)
  * [2주 간의 KoELECTRA 개발기 - 2부](https://monologg.kr/2020/05/02/koelectra-part2/#Finetuning-%EC%BD%94%EB%93%9C-%EC%A0%9C%EC%9E%91)
  * [KoElectra Multiple Label Classification - Emotion](https://github.com/monologg/GoEmotions-Korean)
  * [KoElectra Binary Label Classification - Hate Speech](https://github.com/monologg/korean-hate-speech-koelectra) 
  * [KoElectra Binary Label Classification - NSMC](https://github.com/monologg/KoBERT-nsmc)
  * [Fine Tuning Transformer for MultiClass Text Classification](https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multiclass_classification.ipynb)
* BERT 참고 자료들
  * [BERT Classification 코드 참조할 만한 naver_review_classifications_pytorch_kobert](https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb#scrollTo=qAKJvJGY5z1D)
  * [Multi-Label, Multi-Class Text Classification with BERT, Transformer and Keras](https://link.medium.com/x40Sa1aCBbb)
  * [당근마켓 BERT로 게시글 필터링하기](https://medium.com/daangn/딥러닝으로-동네생활-게시글-필터링하기-263cfe4bc58d)
  * [KOBERT Score](https://github.com/lovit/KoBERTScore)
  * [KoBERT-KorQUAD](https://github.com/monologg/KoBERT-KorQuAD)
  * [네이버 댓글 기반으로 학습한 BERT: KCBERT](https://beomi.github.io/2020/02/26/Train-BERT-from-scratch-on-colab-TPU-Tensorflow-ver/)

---------



## Feedback

* Task에 매몰되지 않았으면 좋겠다. 성능보다는 어떤 식으로 의사결정을 하고, 어떤 근거를 보고 개발하고 나아가는지를 볼 것 같다. Tuning에 시간을 많이 쓰지는 마라.
* Slide에 실험에 대해서 요약을 했으면 좋겠다. 시행착오에 대해서 얘기를 해줬으면 좋겠다. Baseline부터 시작해서 왜 이걸 선택을 하게 됐는지가 드러났으면 좋겠다.
* 최종 발표에 대해서 데이터셋을 더 공개를 하려고 한다. 연립방정식 -> 연랍방정식 형태로 noisy한 형태임. 실제 production 환경과 비슷한 test validation을 할 것 같다. 지금 regex 방식이 아니라 noise에 대응할 수 있는 방식을 보고 싶다. 논리적인 과정을 보고 싶다. 
* 제 예상으로는 Open source를 잘 활용하고 있는 것 같다. 지금까지 진행한 건에 팀이 가장 크게 contribution한 게 어떤 거라고 생각하나?
  * 마지막 최종 발표 때 preprocessing에 대한 분석이 있으면 좋겠다. 어떤 preprocessing을 했을 때 "어느 정도의 성능향상을 가져왔다"는 주장이 더 좋을 것 같다.
  * 울 회사에서도 내부 BERT 모델을 제작해서 1000개 ~ 2000개를 분류하는 classification을 하고 있음.
* 분류할 수 없는 문제가 아니라, 실제로 class를 나눠 떨어지지는 않는다. 충분히 문제를 풀어보고 one label로 했었는데, multi labeling은 내부에서 논의가 나오고 있음. 
* 높은 평가를 드리는 게 아니라 점수 기반이 아니라, 문제 상황을 listup하고 해결하는 과정을 보고 싶다. 성능을 높이기 위해서 어떤 문제가 있었는지, 어떤 게 요인이 제일 크고, 이걸 어떻게 해결한 건지를 보고 싶다.
* 어떤 쪽을 유도하기 위해서가 아니라 본인이 중요하다고 생각한 부분을 집중적으로 파면 되고, 그게 왜 중요한 지만 설득할 수 있으면 됨. Presentation을 할 때 그 부분이 부각될 수 있도록 하면 좋겠다. 결과를 보여준 것들은 Model 별로 보여준 느낌이 있어서... 어떤 방식으로 의사결정을 하고, 어떤 방향으로 나갔는지, 어떤 로직으로 방향을 결정했는지를 보여주면 좋겠다. 

--------

### Others

#### Notation Parsing & Markup

* [Library parsing mathematical equations in strings](https://github.com/gunthercox/mathparse)
* Markup languages for mathmatics are: [Latex](https://www.math.ubc.ca/~pwalls/math-python/jupyter/latex/), [AsciiMath](http://asciimath.org/), Mathml

#### Korean Corpora

* [Available Korean Corpora](https://github.com/songys/AwesomeKorean_Data)
