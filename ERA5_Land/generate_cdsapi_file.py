import os

# 你的 CDS 账户 UID 和 API Key（从 Profile 页面复制）
UID = "<your-uid>"
API_KEY = "<your-api-key>"

# 文件内容
cdsapirc_content = f"""url: https://cds.climate.copernicus.eu/api
key: {UID}:{API_KEY}
verify: 1
"""

# 获取用户家目录路径
home_dir = os.path.expanduser("~")
cdsapirc_path = os.path.join(home_dir, ".cdsapirc")

# 写入文件
with open(cdsapirc_path, "w") as f:
    f.write(cdsapirc_content)

print(f".cdsapirc 已生成在 {cdsapirc_path}")
