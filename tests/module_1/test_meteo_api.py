import pandas as pd
import requests
import json
import pytest
from unittest.mock import Mock

from src.module_1.module_1_meteo_api import (
    request_API,
    get_data_meteo_api,
    data_API_to_df,
    mensualizar_datos,
)




def test_request_API_exito(monkeypatch):
    mockedResponse = Mock(spec=requests.models.Response,status_code=200,text= "data_ejemplo")
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mockedResponse)
    respuesta = request_API("url",params="parametros")
    assert respuesta.status_code == 200
    assert respuesta.text == "data_ejemplo"    
    
def test_request_API_fallo(monkeypatch):
    mockedResponse = Mock(spec=requests.models.Response,status_code=404)
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mockedResponse)
    respuesta = request_API("url",params="parametros")
    assert respuesta.status_code == 404
   


def test_data_API_to_df():
    mockedResponse = Mock(spec=requests.models.Response)
    mockedResponse.text = json.dumps(
        {"daily": {"temperature_2m_mean": [20], "precipitation_sum": [0]}}
    )

    esperados = {"temperature_2m_mean": [20], "precipitation_sum": [0]}

    pd.testing.assert_frame_equal(
        data_API_to_df(mockedResponse), pd.DataFrame(esperados), check_dtype=False
    )


def test_mensualizar_datos():
    datos = {
        "time": ["2010-01-01", "2010-01-02", "2010-01-03"],
        "temperature_2m_mean": [20, 21, 22],
        "precipitation_sum": [1, 2, 3],
        "wind_speed_10m_max": [10, 11, 12],
    }

    esperados = {
        "year": [2010],
        "month": [1],
        "temperature_2m_mean": [21],
        "precipitation_sum": [6],
        "wind_speed_10m_max": [12],
        "date": [pd.to_datetime("2010-01-01")],
    }
    funciones = {
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
        "wind_speed_10m_max": "max",
    }

    pd.testing.assert_frame_equal(
        mensualizar_datos(pd.DataFrame(datos), funciones),
        pd.DataFrame(esperados),
        check_dtype=False,
    )
