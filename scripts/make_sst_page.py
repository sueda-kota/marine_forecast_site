from pathlib import Path

# Copernicus Marine SST forecast page generator
# このスクリプトで以下を行う予定です。
# 1. Copernicus Marineから海面水温データを取得
# 2. NetCDFを軽量化
# 3. 画像を作成
# 4. public/index.html を生成
# 5. GitHub Pages用artifactとして公開

tmp_dir = Path("tmp")
public_dir = Path("public")

tmp_dir.mkdir(exist_ok=True)
public_dir.mkdir(exist_ok=True)

print("make_sst_page.py started")
print(f"tmp_dir: {tmp_dir}")
print(f"public_dir: {public_dir}")
