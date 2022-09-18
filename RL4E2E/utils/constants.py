import os
import pathlib
PROJECT_PATH = ((pathlib.Path(__file__).parent.resolve()).parent.resolve()).parent.resolve()
PROJECT_NAME = "RL4E2E"

FRAMEWORK_PATH = os.path.join(PROJECT_PATH,PROJECT_NAME)
MODELS_PATH = os.path.join(PROJECT_PATH,"Models")
GALAXY_PATH = os.path.join(MODELS_PATH,"GALAXY")
PPTOD_PATH = os.path.join(MODELS_PATH,"pptod/E2E_TOD")

BASELINE_NAME  = "PickNPlug"
BASELINE_PATH = os.path.join(FRAMEWORK_PATH,BASELINE_NAME)