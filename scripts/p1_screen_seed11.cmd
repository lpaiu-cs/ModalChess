@echo off
rem P1 스크리닝 (seed 11, 5 arm 순차) — Claude 세션과 분리 실행용 (마커: SCREEN_DONE.marker)
cd /d E:\lab\modalchess
set PYTHONUNBUFFERED=1
rem 단편화 완화 — eval peak가 dedicated 경계에 눌러앉아 shared로 스필되던 문제 대응
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set LOGDIR=outputs\phase2\p1_v1\screen
if not exist %LOGDIR% mkdir %LOGDIR%
for %%A in (board fen_soft fen_zs blind rawboard) do (
  echo === arm %%A start === >> %LOGDIR%\screen.log
  C:\Users\lpaiu\AppData\Local\Programs\Python\Python312\python.exe -u scripts\train_fusion_arm.py --config configs\fusion\p1_v1.yaml --arm %%A --seed 11 --output-dir %LOGDIR%\%%A_seed11 >> %LOGDIR%\screen.log 2>&1
  if errorlevel 1 echo ARM_FAILED %%A >> %LOGDIR%\screen.log
)
echo done > %LOGDIR%\SCREEN_DONE.marker
