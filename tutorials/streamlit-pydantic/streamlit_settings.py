from typing import List

import streamlit as st
from pydantic import BaseModel, Field, SecretStr

import streamlit_pydantic as sp


class SubModel(BaseModel):
    things_i_like: List[str]


class MySettings(sp.StreamlitSettings):
    username: str = Field(..., description="The username for the database.")
    password: SecretStr
    my_cool_secrets: SubModel


st.json(MySettings().dict())


# Does not work with v1 - streamlit-settings.py
# Does not work with v2 - streamlit-settings.py, overwrite_streamlit_args.py, data_validation.py, dataclass_form.py, complex_showcase.py, complex_disabled_showcase.py, complex_instance_model.py
