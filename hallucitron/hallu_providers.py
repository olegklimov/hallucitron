import os

import yaml

from hallucitron.hallu_structs import HalluStructuredRequest


# Credentials come from env vars (see .test_api_keys.example); test_models.yaml maps
# each model to a provider, endpoint, the env var holding its key, and optional prices.


def models_path():
    return os.environ.get("HALLU_MODELS") or os.path.join(_repo_root(), "test_models.yaml")


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_models(path=None):
    path = path or models_path()
    with open(path) as f:
        return yaml.safe_load(f)["models"]


def prefill_request_with_model(model, path=None):
    m = load_models(path).get(model)
    if m is None:
        raise RuntimeError("model %r not in %s" % (model, path or models_path()))
    api_key = os.environ.get(m["api_key_env"], "")
    if not api_key:
        raise RuntimeError("env var %s not set (try: source .test_api_keys)" % m["api_key_env"])
    return HalluStructuredRequest(
        prov_name=m["provider"],
        prov_endpoint=m["endpoint"],
        prov_api_key=api_key,
        provm_name=model,
        provm_prices=m.get("prices") or {},
    )
