import pyprojroot
import numpy as np
import nhanes.load

def make_dir_function(dir_name):
  def dir_function(*args):
    return pyprojroot.here().joinpath(dir_name, *args)
  return dir_function


def load_and_clean_nhanes_data(cycle='2017-2018'):
    """
    Descarga y limpia los datos de NHANES para el ciclo especificado.
    
    Parámetros:
        cycle (str): Año del ciclo de NHANES a cargar. Por defecto, '2017-2018'.
    
    Retorna:
        pd.DataFrame: DataFrame limpio con las columnas seleccionadas y valores faltantes tratados.
    """
    nhanes_raw_df = nhanes.load.load_NHANES_data(cycle).clean_names(case_type='snake')
    
    nhanes_df = (
        nhanes_raw_df.select_columns(
            "general_health_condition",
            "age_in_years_at_screening",
            "gender",
            "current_selfreported_height_inches",
            "current_selfreported_weight_pounds",
            "doctor_told_you_have_diabetes",
            "60_sec_pulse30_sec_pulse2",
            "total_cholesterol_mgdl"
        )
        .rename_columns(
            {
                "age_in_years_at_screening": "age",
                "current_selfreported_height_inches": "height",
                "current_selfreported_weight_pounds": "weight",
                "doctor_told_you_have_diabetes": "diabetes",
                "60_sec_pulse30_sec_pulse2": "pulse",
                "total_cholesterol_mgdl": "total_cholesterol"
            }
        )
        .replace(
            {
                "height": {9999: np.nan, 7777: np.nan},
                "weight": {9999: np.nan, 7777: np.nan},
                "diabetes": {"Borderline": np.nan}
            }
        )
        .missing.sort_variables_by_missingness()
        .dropna(subset=["diabetes"], how="any")
        .transform_column(column_name="diabetes", function=lambda s: s.astype(int), elementwise=False)
    )
    
    nhanes_df = nhanes_df.dropna(
        subset=['pulse', 'total_cholesterol', 'general_health_condition', 'weight', 'height'],
        how='all'
    )
    
    return nhanes_df