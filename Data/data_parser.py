import json

data = {}
for i in range(415):
    with open(f"Data/{i}", encoding="utf-8") as f:
        str_data = f.read().replace("'", '"')
        dict_data = json.loads(str_data)
    for key, value in dict_data.items():
        data[key] = value

with open("Data/data.json", 'x', encoding='utf-8') as f:
    f.write(str(data).replace("'", '"'))
