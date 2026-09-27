import os

import mlflow


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{PROJECT_ROOT}/mlruns.db",
)


def configure_mlflow():
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment("BOxCrete")

    return TRACKING_URI