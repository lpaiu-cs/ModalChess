@echo off
rem P1c (seed 11): T3 심화 — fen_soft, hybrid, board, blind @4ep on qa_v2. 세션 독립.
cd /d E:\lab\modalchess
set PYTHONUNBUFFERED=1
set LOGDIR=outputs\phase2\p1c_v1
if not exist %LOGDIR% mkdir %LOGDIR%
for %%A in (fen_soft hybrid board blind) do (
  echo === arm %%A start === >> %LOGDIR%\p1c.log
  C:\Users\lpaiu\AppData\Local\Programs\Python\Python312\python.exe -u scripts\train_fusion_arm.py --config configs\fusion\p1c_v1.yaml --arm %%A --seed 11 --output-dir %LOGDIR%\%%A_seed11 >> %LOGDIR%\p1c.log 2>&1
  if errorlevel 1 echo ARM_FAILED %%A >> %LOGDIR%\p1c.log
)
echo done > %LOGDIR%\P1C_DONE.marker
