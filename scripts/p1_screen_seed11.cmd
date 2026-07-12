@echo off
rem P1 스크리닝 (seed 11, 5 arm 순차) — Claude 세션과 분리 실행용 (마커: SCREEN_DONE.marker)
cd /d E:\lab\modalchess
set PYTHONUNBUFFERED=1
rem 메모리 스필은 fusion_run eval 수정(답 위치에만 log_softmax)으로 해결.
rem expandable_segments는 Windows 미지원이라 설정하지 않음.
set LOGDIR=outputs\phase2\p1_v1\screen
if not exist %LOGDIR% mkdir %LOGDIR%
rem 실패 추적은 파일시스템 플래그로(cmd 지연 확장 함정 회피). 스테일 마커 정리 후 시작.
del /q %LOGDIR%\SCREEN_DONE.marker %LOGDIR%\SCREEN_FAILED.marker 2>nul
for %%A in (board fen_soft fen_zs blind rawboard) do (
  echo === arm %%A start === >> %LOGDIR%\screen.log
  C:\Users\lpaiu\AppData\Local\Programs\Python\Python312\python.exe -u scripts\train_fusion_arm.py --config configs\fusion\p1_v1.yaml --arm %%A --seed 11 --output-dir %LOGDIR%\%%A_seed11 >> %LOGDIR%\screen.log 2>&1
  if errorlevel 1 ( echo ARM_FAILED %%A >> %LOGDIR%\screen.log & echo failed >> %LOGDIR%\SCREEN_FAILED.marker )
)
rem arm이 하나라도 실패하면 DONE 마커를 쓰지 않고 non-zero 종료 — 미완 스크린을 성공으로 오인 방지
if exist %LOGDIR%\SCREEN_FAILED.marker ( exit /b 1 )
echo done > %LOGDIR%\SCREEN_DONE.marker
