from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "GNSS Forecasting System"
    data_path: str = "./data/"
    actual_data_file: str = "clock_bias_correction_combined_1_7_jan_2024.csv"
    predicted_data_file: str = "suyash_gandu.csv"

    google_api_key: str
    llm_model_name: str
    llm_temperature: float

    model_config = SettingsConfigDict(env_file=".env", extra="allow") 

settings = Settings()
