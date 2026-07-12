# Phase 2 결정 — 시각 모달리티가 LM에 무엇을 더하는가 (종결)

브랜치 `phase2_visual_modality`. 상세 시간순: [phase2_log.md](phase2_log.md), 사전 등록:
[phase2_plan.md](phase2_plan.md). seed 11 스크리닝으로 종결(3-seed 미진행 — 사용자 결정).

> **잠정(provisional) 상태**: 아래 결과는 전부 **seed 11 단독**이다. 사전 등록(phase2_plan §9b)은
> H-Q3 통과 시 seed 17·23으로 3-seed 확정을 요구하며, 그 확정은 미실시다. 따라서 관측된
> seed-11 델타는 유효한 증거이나 프로토콜상 **확정된 발견이 아니다**. seeds 17·23 공급 전까지
> "확정"이 아니라 "seed-11 잠정"으로 읽어야 한다.

## 물음

원래 의도(AGENTS.md "modality alignment with an LLM"): 8×8 공간 체스 인코더를 frozen LM에
이식해, LM이 판을 "보고" 근거 있는 답을 하게 한다. 구체 물음: **64칸 공간 토큰이 FEN 문자열
대비 더 나은/보완적인 LM 입력인가.**

## 방법 (사전 등록, 반증 지향)

- frozen Qwen3-4B-Instruct-2507 위, 학습하는 것은 작은 projection/soft-token뿐(LM·인코더 동결).
- 5+1 arm 동일 기판: **board**(공간 토큰), **rawboard**(원시 plane), **fen_soft**(FEN 텍스트),
  **fen_zs**(FEN 무학습), **blind**(판 없음=바닥), **hybrid**(FEN+공간 토큰).
- 프로그램 생성·독립 검증 QA(정적 T1 지각, T2 관계, 동역학 T3), 답 균형, held-out 템플릿,
  shuffled-board null. 판정선·사살 기준 결과 열람 전 커밋.

## 결과

**1) 공간 단독 vs FEN → FEN 우위 (P1, seed-11 반증 방향).** LM은 FEN 텍스트를 native하게 읽음.
pooled 공간 토큰은 판독에서 열세. 사전학습 인코더는 원시 plane과 **동률**(raw 판독엔 무이득).

**2) FEN+시각 vs FEN 단독 → 시각이 보강 (P1b, seed-11 잠정).** hybrid 0.886 > fen_soft 0.866.
이득의 소재 = **정적 파생 상태**: is_check +0.15, pin +0.09. (수렴 교란 제거 위해 4ep,
board도 학습량↑에 val은 fen과 동률이나 test는 −0.03로 판독 일반화 열세 지속.)

**3) 보강의 경계 (P1c).**
- **정적 파생 상태 보강은 진짜** — 다른 코퍼스(qa_v2)에서 재현, **blind 검증**: is_check
  blind 0.504(우연) vs board/hybrid 0.995. 누출·용량 아티팩트 아님(board 단독도 동일
  projection으로 0.998).
- **동역학(T3)은 미판정** — blind(판 없음)이 T3=0.833으로, 판 보는 어느 arm도 못 넘음.
  T3 과제가 수의 기하만으로 풀리는 **보드-독립적 설계 결함**(illegal move가 기하로 판별
  가능). 시각이 동역학을 돕는지 판정 불가. blind arm이 검출(T1/T2 누출 검출과 동일).

## 결론

**시각/인코더 채널의 보강 가치는 "인코더가 지도학습(state-probe)으로 계산한 정적 현재-국면
파생 상태(체크·핀)를 텍스트에 더하는 것"에 한정되는 것으로 보인다(seed-11 잠정, 3-seed 확정 전).**
- seed 11에서 이 델타는 견고했다(blind 대조로 누출·용량 아티팩트 배제). 다만 사전 등록
  프로토콜상 seeds 17·23 확정 전에는 "진짜"로 단정하지 않는다.
- 그 밖으로 확장 안 됨: raw 판독은 열세(FEN 우위), 동역학은 미판정.
- 정직한 프레이밍: "raw 시각이 판을 더 잘 본다"가 아니라 **"인코더가 학습한 특정 정적 특징이
  텍스트를 보강한다"**(학습된 특징의 전이). in-check는 입력이 아니라 인코더 표현으로 전달
  → 가드레일 누출 아님.

## 한계 / 미결

- **seed 11 단독** (3-seed 미확정) — 헤드라인 주장엔 3-seed 필요(연구 관례).
- **동역학 미판정** — 누출 없는 T3 재설계(illegal move를 합법 move와 기하 매칭, check/non-check
  move를 기하 매칭) 필요.
- **QA 누출** — blind이 king_square·square_attacked·piece_defended(정적)과 T3 전반에서 누출
  검출. 정적 핵심 발견(체크·핀)은 blind이 우연 수준이라 영향 없으나, 다른 지표의 절대값은
  누출 정화 후 재측정 필요.
- 용량 대칭 통제(fen_soft 큰 head) 미실시 — is_check는 정보-내용 논증으로 무관하나 명시.

## 재사용 자산

- `src/modalchess/fusion/`: arm 모듈(board/rawboard/blind/fen_*/hybrid), 시퀀스 조립,
  후보 logprob 채점, shuffled null, epoch 체크포인트/재개.
- `qa_generator`/`qa_verifier`(독립 경로), qa_v1(T1+T2)·qa_v2(+T3) 코퍼스.
- 인프라 교훈(phase2_log): GPU shared-spill 진단(Get-Counter), grad accumulation,
  분리 실행(Start-Process), 멱등 재실행.
