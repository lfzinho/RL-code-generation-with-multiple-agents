import pandas as pd

def change_csv(string_code, csv_path):
    df = pd.read_csv(csv_path)
    global_scope = {}
    local_scope = {"df": df}
    exec(string_code, global_scope, local_scope)
    df_updated = local_scope["df"]
    df_updated.to_csv(csv_path, index=False)   