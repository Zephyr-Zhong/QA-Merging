import pandas as pd

def load_eval_data(file_path):
    if isinstance(file_path, dict):
        gsm8k_data = pd.read_json(file_path['gsm8k'], lines=True)
        gsm8k_data = gsm8k_data.drop(columns=["answer","report"])
        math_data = pd.read_json(file_path['math'], lines=True)
        math_data = math_data.drop(columns=["solution","answer","report"])
        aime_data = pd.read_json(file_path['aime'], lines=True)
        aime_data = aime_data.drop(columns=["solution","answer","report"])
        
        data = pd.concat([gsm8k_data, math_data, aime_data], ignore_index=True)
        data = data.rename(columns={"code": "responses"})
        print(gsm8k_data.columns)
        print(math_data.columns)
        print(aime_data.columns)
        print(data.columns)
    else:
        data = pd.read_json(file_path, orient='index')
        data = data.to_dict(orient='records')
    return data