# STL → STEP 변환기

STL 메시 파일을 STEP(AP214) 솔리드로 바꾸는 작은 도구입니다. [OpenCascade](https://dev.opencascade.org/)(OCP 파이썬 바인딩)를 씁니다.
명령줄로도 쓸 수 있고, 내 PC에서 도는 웹페이지(끌어다 놓기)로도 쓸 수 있습니다. 파일은 외부로 전송되지 않습니다.

## 하는 일
1. STL 삼각형을 읽어 면으로 꿰맵니다 (sewing, 허용 오차 `--tol`).
2. 닫힌 껍질은 솔리드로 만듭니다 (덩어리가 여러 개여도 각각 처리).
3. 같은 평면에 있는 삼각형들을 큰 면 하나로 합칩니다 (`--no-merge`로 끔).
4. STEP 파일로 저장하고 삼각형 수·면 수·부피·형상 검사 결과를 알려 줍니다.

> **주의:** 곡면은 매끈한 NURBS 곡면으로 복원되지 않고 삼각형(평면) 조각 그대로 들어갑니다.
> 다른 CAD에서 열기·배치·불리언 정도는 되지만 곡면을 편집하기는 어렵습니다.
> 삼각형이 20만 개를 넘으면 수십 분 걸릴 수 있으니, 메시를 먼저 줄이는 것을 권합니다.

## 설치
Python 3.10 이상.
```bash
pip install -r requirements.txt
```

## 사용법

### 웹페이지
```bash
python stl_to_step_web.py              # http://127.0.0.1:8765 접속
python stl_to_step_web.py --host 0.0.0.0 --port 8765   # 같은 네트워크의 다른 기기(폰 등)에서 접속
```
STL을 끌어다 놓으면 변환되고 **STEP 받기** 버튼이 나타납니다. 여러 파일을 한꺼번에 넣을 수 있습니다(차례로 처리).

### 명령줄
```bash
python stl_to_step.py part.stl                    # part.step 생성
python stl_to_step.py a.stl b.stl -o out/         # 여러 개를 out/ 폴더로
python stl_to_step.py "parts/*.stl" -o out/       # 글롭 패턴
python stl_to_step.py part.stl -o result.step --tol 0.005
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `-o, --out` | 입력과 같은 폴더 | 출력 파일(`.step`) 또는 폴더 |
| `--tol` | `0.01` | 꿰매기 허용 오차 (mm) |
| `--no-merge` | 끔 | 면 합치기 생략 (빠르지만 면 수 = 삼각형 수) |

출력 예:
```
OK  ail_L1.stl -> ail_L1.step | 삼각형 6744 -> 면 5283 | 덩어리 1 | 부피 71.1 cm3 | 4.2s
```
변환 실패가 있으면 종료 코드 1을 돌려줍니다.

## 경고 메시지
- **열린 껍질 N개**: STL에 구멍이 있어 닫힌 솔리드가 안 됐습니다. 껍질(면 모음)로 저장됩니다. 메시 복구 도구로 구멍을 메운 뒤 다시 변환하세요.
- **형상 검사 불통과**: OpenCascade 형상 검사를 통과하지 못했습니다. `--tol` 값을 바꿔 보세요.

## 라이선스
MIT
