[로봇 서버 업데이트 안내 - 2026-08-31]

이 zip의 파일 3개를 로봇 PC의 so101-vla-pipeline-edge-deployment 폴더에
같은 경로 그대로 덮어쓰세요:

  so101_pipeline/interfaces/command_bridge.py   (한국어/영어 명령 정규화 추가)
  so101_pipeline/interfaces/edge_app_backend.py (이해 못한 명령 응답 추가)
  so101_pipeline/runtime/main_edge.py           (정규화된 명령 비교로 변경)

덮어쓴 뒤 실행 중인 서버(main_edge.py 또는 main_real2.py)를 재시작하면 적용됩니다.

새 앱(robot_control.apk)은 한국어 명령을 보내므로, 이 서버 업데이트 없이는
로봇이 명령을 인식하지 못합니다. 반드시 같이 적용하세요.
