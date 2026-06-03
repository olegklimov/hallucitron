import os

import yaml

from hallucitron.hallu_structs import HalluStructuredRequest


# A hallucitron config is a set of providers, each owning a list of models, plus a
# per-model price/capability table. See providers_default.yaml for the on-disk shape.
#
# A real application using this library supplies its own config -- in a multi-tenant
# system each tenant has a separate config with its own providers and api keys.
# load_default_config() loads the built-in providers_default.yaml, with an explicit
# choice of whether api keys come from the environment or straight from the config.


_DEFAULT_CONFIG = "providers_default.yaml"


class HalluConfig:
    """Parsed providers + models. `providers` maps provider-id -> provider dict;
    `models` maps model-name -> price/capability dict (with the owning provider's
    `more_prices` already merged in)."""

    def __init__(self, providers, models):
        self.providers = providers
        self.models = models
        # model-name -> provider-id, built once for fast lookup.
        self._owner = {}
        for pid, prov in providers.items():
            for model in prov.get("models") or []:
                self._owner[model] = pid

    def provider_for_model(self, model):
        pid = self._owner.get(model)
        if pid is None:
            raise RuntimeError("model %r not owned by any provider" % model)
        return pid, self.providers[pid]

    def prices_for_model(self, model):
        """Model price table with the owning provider's more_prices merged in."""
        _, prov = self.provider_for_model(model)
        prices = dict(self.models.get(model) or {})
        prices.update(prov.get("more_prices") or {})
        return prices


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_config_path():
    return os.path.join(_repo_root(), _DEFAULT_CONFIG)


def _read_test_keys():
    """Parse the repo's .test_api_keys (shell `export NAME=VALUE` lines) into a
    {NAME: value} dict. Dev/test convenience; missing file yields {}."""
    keys = {}
    try:
        with open(os.path.join(_repo_root(), ".test_api_keys")) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return keys
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        name, sep, value = line.partition("=")
        if sep:
            keys[name.strip()] = value.strip().strip('"').strip("'")
    return keys


def parse_config(data, use_env_keys=False, use_test_keys=False):
    """Build a HalluConfig from already-parsed YAML/dict data.

    use_env_keys=False/use_test_keys=False: api keys are taken verbatim from the config
        (the multi-tenant case -- the caller injected the keys it wants).
    use_env_keys=True:  each provider's api_key is filled from the env var it names in
        `api_key_env`, overriding whatever the config held. Use this only for local
        runs where keys live in the environment, not the config.
    use_test_keys=True: same, but keys come from the repo's .test_api_keys file rather
        than the environment (the dev/test convenience). Wins if both are set.
    """
    providers = dict(data.get("providers") or {})
    models = dict(data.get("models") or {})
    if use_env_keys or use_test_keys:
        src = _read_test_keys() if use_test_keys else os.environ
        providers = {pid: dict(prov) for pid, prov in providers.items()}
        for prov in providers.values():
            env_name = prov.get("api_key_env")
            if env_name:
                prov["api_key"] = src.get(env_name, prov.get("api_key", ""))
    return HalluConfig(providers, models)


def load_config(path, use_env_keys=False, use_test_keys=False):
    """Load a config from a YAML file. See parse_config for key-source semantics."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return parse_config(data, use_env_keys=use_env_keys, use_test_keys=use_test_keys)


def load_default_config(use_env_keys=False, use_test_keys=False):
    """Load the built-in providers_default.yaml. By default api keys come from the
    config as-is; pass use_env_keys=True to fill them from the environment, or
    use_test_keys=True to fill them from the repo's .test_api_keys file."""
    return load_config(default_config_path(), use_env_keys=use_env_keys, use_test_keys=use_test_keys)


def prefill_request_with_model(config, model):
    """Build a request for `model`, resolving its owning provider from `config`."""
    _, prov = config.provider_for_model(model)
    api_key = prov.get("api_key", "")
    if not api_key:
        raise RuntimeError(
            "no api_key for provider of model %r "
            "(set it in the config, or load with use_env_keys=True)" % model
        )
    return HalluStructuredRequest(
        prov_name=prov["kind"],
        prov_endpoint=prov["endpoint"],
        prov_api_key=api_key,
        provm_name=model,
        provm_prices=config.prices_for_model(model),
    )
