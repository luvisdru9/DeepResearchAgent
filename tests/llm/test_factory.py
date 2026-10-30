import pytest

from deepresearch.config import LLMConfig, load_config, load_env_file
from deepresearch.llm import MockLLMClient, OpenAICompatibleClient, create_llm_client


def test_factory_creates_configured_mock_client():
    client = create_llm_client(
        LLMConfig(
            provider="mock",
            model="mock-test",
            timeout_seconds=5,
            max_retries=1,
        )
    )

    assert isinstance(client, MockLLMClient)
    assert client.model == "mock-test"
    assert client.timeout_seconds == 5
    assert client.max_retries == 1
    assert client.retry_base_delay_seconds == 0.0
    assert client.retry_max_delay_seconds == 0.0


def test_named_profiles_resolve_active_and_explicit_override():
    config = LLMConfig(
        active_profile="mock",
        profiles={
            "mock": LLMConfig(provider="mock", model="mock-default"),
            "other": LLMConfig(provider="mock", model="mock-other"),
        },
    )

    assert create_llm_client(config).model == "mock-default"
    assert create_llm_client(config, profile_name="other").model == "mock-other"
    assert create_llm_client(config, profile_name="other").profile_name == "other"
    assert create_llm_client(config, profile_name="other").provider_name == "mock"


def test_flat_llm_config_remains_backward_compatible():
    config = LLMConfig(provider="mock", model="legacy-mock")

    assert config.resolve_profile() is config
    assert create_llm_client(config).model == "legacy-mock"


def test_default_yaml_enables_agent_specific_deepseek_profiles():
    config = load_config("configs/default.yaml")

    assert config.llm.active_profile == "planner_deepseek"
    assert config.llm.resolve_profile().provider == "openai_compatible"
    assert config.planning.provider == "llm"
    assert config.reader.provider == "llm"
    assert config.synthesis.provider == "llm"
    assert config.planning.llm_profile == "planner_deepseek"
    assert config.reader.llm_profile == "reader_deepseek"
    assert config.synthesis.llm_profile == "synthesizer_deepseek"
    assert config.reader.max_docs_per_call == 4
    assert config.reader.max_evidence_per_doc == 2
    assert config.reader.max_chars_per_doc == 2500
    assert config.reader.target_evidence_per_subtask == 3
    assert config.reader.target_unique_sources_per_subtask == 2
    assert config.reader.max_empty_docs_before_stop == 2
    assert config.reader.enable_gap_recovery is True
    assert config.reader.gap_recovery_max_docs == 2
    assert config.runtime.save_llm_io is True
    assert config.runtime.llm_trace_subdir == "llm_traces"
    assert config.tool_calling.enabled is False
    assert config.tool_calling.max_steps == 3
    assert config.tool_calling.max_calls_per_turn == 1
    planner = config.llm.resolve_profile("planner_deepseek")
    reader = config.llm.resolve_profile("reader_deepseek")
    synthesizer = config.llm.resolve_profile("synthesizer_deepseek")
    redteam = config.llm.resolve_profile("redteam_deepseek")
    evaluator_pro = config.llm.resolve_profile("evaluator_deepseek_pro")
    evaluator_pro_compact = config.llm.resolve_profile("evaluator_deepseek_pro_compact")
    evaluator_flash_secondary = config.llm.resolve_profile(
        "evaluator_deepseek_flash_secondary"
    )
    assert planner.provider == "openai_compatible"
    assert planner.model == "deepseek-flash"
    assert planner.base_url == "https://api.deepseek.com"
    assert planner.api_key_env == "DEEPSEEK_API_KEY"
    assert planner.temperature is None
    assert planner.retry_base_delay_seconds == 0.5
    assert planner.retry_max_delay_seconds == 8.0
    assert planner.retry_jitter_ratio == 0.2
    assert planner.request_options == {
        "reasoning_effort": "high",
        "max_tokens": 2000,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert reader.request_options == {
            "reasoning_effort": "none",
            "max_tokens": 1600,
        "response_format": {"type": "json_object"},
    }
    assert synthesizer.request_options == {
        "reasoning_effort": "none",
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
    }
    assert reader.model == "deepseek-flash"
    assert synthesizer.model == "deepseek-flash"
    assert redteam.model == "deepseek-flash"
    assert redteam.request_options == {
        "reasoning_effort": "none",
        "max_tokens": 3000,
        "response_format": {"type": "json_object"},
    }
    assert evaluator_pro.provider == "openai_compatible"
    assert evaluator_pro.model == "deepseek-v4-pro"
    assert evaluator_pro.base_url == "https://api.deepseek.com"
    assert evaluator_pro.api_key_env == "DEEPSEEK_API_KEY"
    assert evaluator_pro.timeout_seconds == 180
    assert evaluator_pro.temperature is None
    assert evaluator_pro.request_options == {
        "reasoning_effort": "high",
        "max_tokens": 12000,
        "response_format": {"type": "json_object"},
        "extra_body": {"thinking": {"type": "enabled"}},
    }
    assert evaluator_pro_compact.model == "deepseek-v4-pro"
    assert evaluator_pro_compact.timeout_seconds == 180
    assert evaluator_pro_compact.max_retries == 1
    assert evaluator_pro_compact.request_options == {
        "reasoning_effort": "none",
        "max_tokens": 5000,
        "response_format": {"type": "json_object"},
    }
    assert evaluator_flash_secondary.model == "deepseek-flash"
    assert evaluator_flash_secondary.request_options == {
        "reasoning_effort": "none",
        "max_tokens": 5000,
        "response_format": {"type": "json_object"},
    }
    assert config.evaluation.judge_profiles == [
        "evaluator_deepseek",
        "evaluator_deepseek_flash_secondary",
    ]
    assert config.evaluation.judge_protocol_version == "5.0"
    assert config.evaluation.judge_aggregation_policy == (
        "hierarchical_model_consensus"
    )
    assert config.evaluation.judge_min_successful_repetitions_per_profile == 2
    assert config.redteam.enable_blue_revision is True
    assert config.redteam.verify_after_revision is True


def test_unknown_profile_has_clear_error():
    config = LLMConfig(
        active_profile="missing",
        profiles={"mock": LLMConfig()},
    )

    with pytest.raises(ValueError, match="Unknown LLM profile 'missing'"):
        create_llm_client(config)


def test_factory_requires_api_key_for_network_profile(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    config = LLMConfig(
        provider="openai_compatible",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        api_key_env="DEEPSEEK_API_KEY",
    )

    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        create_llm_client(config)


def test_factory_constructs_real_async_sdk_without_sending_request(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "placeholder-not-sent")
    config = load_config("configs/default.yaml")

    client = create_llm_client(config.llm)

    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "deepseek-flash"
    assert client.base_url == "https://api.deepseek.com"
    assert client.profile_name == "planner_deepseek"
    assert client.provider_name == "openai_compatible"
    assert config.reader.provider == "llm"
    assert config.reader.llm_profile == "reader_deepseek"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_client(LLMConfig(provider="unknown"))


def test_load_env_file_sets_missing_values_without_overriding_existing(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "DEEPSEEK_API_KEY=from_file",
                "EXISTING_KEY=from_file",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("EXISTING_KEY", "from_shell")

    load_env_file(env_path)

    assert __import__("os").environ["DEEPSEEK_API_KEY"] == "from_file"
    assert __import__("os").environ["EXISTING_KEY"] == "from_shell"
