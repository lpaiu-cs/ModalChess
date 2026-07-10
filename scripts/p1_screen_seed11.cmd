@echo off
rem P1 스크리닝 (seed 11, 5 arm 순차) — Claude 세션과 분리 실행용 (마커: SCREEN_DONE.marker)
cd /d E:\lab\modalchess
set PYTHONUNBUFFERED=1
rem 메모리 스필은 fusion_run eval 수정(답 위치에만 log_softmax)으로 해결.
rem expandable_segments는 Windows 미지원이라 설정하지 않음.
set LOGDIR=outputs\phase2\p1_v1\screen
if not exist %LOGDIR% mkdir %LOGDIR%
for %%A in (board fen_soft fen_zs blind rawboard) do (
  echo === arm %%A start === >> %LOGDIR%\screen.log
  C:\Users\lpaiu\AppData\Local\Programs\Python\Python312\python.exe -u scripts\train_fusion_arm.py --config configs\fusion\p1_v1.yaml --arm %%A --seed 11 --output-dir %LOGDIR%\%%A_seed11 >> %LOGDIR%\screen.log 2>&1
  if errorlevel 1 echo ARM_FAILED %%A >> %LOGDIR%\screen.log
)
echo done > %LOGDIR%\SCREEN_DONE.marker
