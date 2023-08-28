
# Local Testing

# Update the API Keys and Base Folder Path

This example utilizes two external APIs. These should be updated in the API_keys_config.py file.

This example also utilizes an assets folder that should be explicitly set to circumvent reference issues.

## DALL-E API key
openai.api_key = API_keys_config.openai_api_key

## NASA API Keys
DEMO_KEY = API_keys_config.NASA_api_key

## Base Folder Path in spaceSimulator.py
basePath = <base_path_to_spaceSimulator_folder>

e.g. --> basePath = "/home/jovyan/shared/notebooks/spaceSimulator"

# Dependencies

This dashboard requires the following dependencies of which the first four are embedded on the image directly as they are a core component for the utilization of Python Shiny. The rest need to be loaded with a module file.

```
pip3 install shiny==0.4.0 uvicorn==0.23.1 shinyswatch==0.2.4 shinywidgets==0.2.1 numpy==1.25.2 matplotlib==3.7.2 astropy==5.3.2 openai==0.27.9 plotly==5.16.1 pandas==2.0.3
```