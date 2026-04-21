from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    host: str = "0.0.0.0"
    port: int = 8889
    log_level: str = "INFO"

    model_path: str = "zai-org/GLM-OCR"
    device_map: str = "auto"
    torch_dtype: Literal["auto", "bfloat16", "float16", "float32"] = "auto"
    trust_remote_code: bool = True

    max_new_tokens: int = 8192
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    no_repeat_ngram_size: int = 12
    skip_special_tokens: bool = False

    max_concurrent_inferences: int = 1

    pdf_dpi: int = 200
    pdf_max_pages: int = 32

    max_upload_bytes: int = Field(default=50 * 1024 * 1024)
    max_batch_size: int = 16

    default_prompt: str = "Text Recognition:"
    hf_home: Optional[str] = None

    cors_allow_origins: list[str] = ["*"]
    background_model_load: bool = True
    fail_fast_on_load_error: bool = False


settings = Settings()
