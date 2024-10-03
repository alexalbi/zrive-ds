# Importancion librerias
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "https://archive-api.open-meteo.com/v1/archive?"
COORDINATES = {
    "Madrid": {"latitude": 40.416775, "longitude": -3.703790},
    "London": {"latitude": 51.507351, "longitude": -0.127758},
    "Rio": {"latitude": -22.906847, "longitude": -43.172896},
}
VARIABLES = ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max"]


def request_API(urlAPI: str, **kwargs: dict) -> requests.models.Response:
    """
    Funcion que realiza una consulta a cualquier API, devolviendo la
    respuesta de la misma.
    Adaptable a añadirle parametros a la consulta.

    Args:
        urlAPI (str): API key para la consulta
        **kwargs (dict): Parametros de la consulta

    Returns:
        r (requests.models.Response): Respuesta de la API

    Raises:
    Exception : Error en la consulta
    """
    # Consulta a la API
    r = requests.get(urlAPI, **kwargs)
    if r.status_code != 200:
        raise Exception("Error en la consulta")
    if r.status_code == 200:
        print("Consulta exitosa")
    return r


def get_data_meteo_api(ciudad: str, variables_temp: list) -> requests.models.Response:
    """
    Funcion que realiza una consulta a la API de Open Meteo, según los argumentos
    introducidos.

    Para hacer la consulta a la API utiliza la función request_API, creada
    anteriormente.

    Args:
        ciudad (str): Ciudad de la que se quiere obtener los datos.
        variables_temp (list): Lista de variables de temperatura que se
        quieren consultar.
    Returns:
        r (requests.models.Response): Respuesta de la API.

    """
    # Creación de la API
    parametros = {
        "latitude": COORDINATES[ciudad]["latitude"],
        "longitude": COORDINATES[ciudad]["longitude"],
        "start_date": "2010-01-01",
        "end_date": "2019-12-31",
        "daily": variables_temp,
        "timezone": "Europe/London",
    }
    r = request_API(API_URL, params=parametros)
    return r


def data_API_to_df(data: requests.models.Response) -> pd.DataFrame:
    """
    Funcion que convierte la respuesta de la API a un DataFrame.
    Coge el texto de la respuesta, lo convierte a un diccionario y
    posteriormente en un DataFrame.

    Args:
        data (requests.models.Response): Respuesta de la API.

    Returns:
        df (pd.DataFrame): DataFrame con los datos extraidos de la API.
    """
    datos = data.text
    datos_json = json.loads(datos)
    df = pd.DataFrame(datos_json["daily"])
    return df


def mensualizar_datos(df: pd.DataFrame, funciones: dict) -> pd.DataFrame:
    """
    Funcion que mensualiza los datos de un DataFrame.
    Agrupa los datos por mes y hace el cálculo correspondiente para cada variable
    a partir del diccionario añadido.

    Args:
        df (pd.DataFrame): DataFrame con los datos diarios.

    Returns:
        df (pd.DataFrame): DataFrame con los datos mensualizados.
    """
    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month
    df["year"] = df["time"].dt.year
    df = df.groupby(["year", "month"]).agg(funciones).reset_index()
    df["date"] = pd.to_datetime(df[["year", "month"]].assign(DAY=1))
    return df


def graficoVariable(dataframes: list, variable: str, tipo: str, labels: list) -> None:
    """
    Función que realiza una grafico sobre la variable introducida desde diferentes
    dataframes en un solo gráfico.
    Con el lineplot se puede ver la evolución de la variable en el tiempo. Con el
    histplot se puede ver la distribución de la variable.

    Args:
        dataframes (list): Lista de dataframes con los datos mensualizados.
        variable (str): Variable que se quiere graficar.
        tipo (str): Tipo de gráfico ('hist', 'line').
        labels (list): Lista para escoger los valores de la leyenda.
    """
    plt.figure(figsize=(14, 6))

    for df, label in zip(dataframes, labels):
        if tipo == "hist":
            sns.histplot(data=df, x=variable, bins=30, label=label)
        elif tipo == "line":
            sns.lineplot(data=df, x="date", y=variable, marker="o", label=label)

    plt.title(f"{tipo.capitalize()}plot de la variable {variable}")
    plt.xlabel("Fecha" if tipo == "line" else variable)
    plt.ylabel(variable if tipo == "line" else "Frecuencia")
    plt.legend()
    # plt.show()


def main():
    madridData = get_data_meteo_api("Madrid", VARIABLES)
    londonData = get_data_meteo_api("London", VARIABLES)
    rioData = get_data_meteo_api("Rio", VARIABLES)

    madridDF = data_API_to_df(madridData)
    londonDF = data_API_to_df(londonData)
    rioDF = data_API_to_df(rioData)

    funciones = {
        "temperature_2m_mean": "mean",
        "precipitation_sum": "sum",
        "wind_speed_10m_max": "max",
    }

    madridDF_mensual = mensualizar_datos(madridDF, funciones)
    londonDF_mensual = mensualizar_datos(londonDF, funciones)
    rioDF_mensual = mensualizar_datos(rioDF, funciones)

    dfs = [madridDF_mensual, londonDF_mensual, rioDF_mensual]
    labelsLeyenda = ["Madrid", "London", "Rio"]

    # Gráfico lineal para mostrar la evolución de las variables con el tiempo
    graficoVariable(dfs, "temperature_2m_mean", "line", labelsLeyenda)
    graficoVariable(dfs, "precipitation_sum", "line", labelsLeyenda)
    graficoVariable(dfs, "wind_speed_10m_max", "line", labelsLeyenda)

    # Histogramas para mostrar la distribución de las variables
    graficoVariable(dfs, "temperature_2m_mean", "hist", labelsLeyenda)
    graficoVariable(dfs, "precipitation_sum", "hist", labelsLeyenda)
    graficoVariable(dfs, "wind_speed_10m_max", "hist", labelsLeyenda)

    plt.show()


if __name__ == "__main__":
    main()
