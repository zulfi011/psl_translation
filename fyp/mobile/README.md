# Mobile app (Flutter)

Place the Flutter project here (the folder containing `pubspec.yaml`).

It talks to the API in [`../server/app.py`](../server/app.py):
`POST /process` with a multipart `video` field, `GET /health`.

Before publishing, make sure the API base URL is a build-time setting
(e.g. `--dart-define=API_URL=...`), not a hard-coded IP, and that no keys are
committed.
