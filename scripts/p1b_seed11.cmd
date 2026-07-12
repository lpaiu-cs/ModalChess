@echo off
rem P1b (seed 11): 수렴(epochs 4) + hybrid — board, fen_soft, hybrid 순차. 세션 독립.
cd /d E:\lab\modalchess
set PYTHONUNBUFFERED=1
set LOGDIR=outputs\phase2\p1b_v1
if not exist %LOGDIR% mkdir %LOGDIR%
del /q %LOGDIR%\P1B_DONE.marker %LOGDIR%\P1B_FAILED.marker 2>nul
for %%A in (fen_soft hybrid board) do (
  echo === arm %%A start === >> %LOGDIR%\p1b.log
  C:\Users\lpaiu\AppData\Local\Programs\Python\Python312\python.exe -u scripts\train_fusion_arm.py --config configs\fusion\p1b_v1.yaml --arm %%A --seed 11 --output-dir %LOGDIR%\%%A_seed11 >> %LOGDIR%\p1b.log 2>&1
  if errorlevel 1 ( echo ARM_FAILED %%A >> %LOGDIR%\p1b.log & echo failed >> %LOGDIR%\P1B_FAILED.marker )
)
if exist %LOGDIR%\P1B_FAILED.marker ( exit /b 1 )
echo done > %LOGDIR%\P1B_DONE.marker
